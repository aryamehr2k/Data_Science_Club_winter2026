import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.dataset import TP53StructureDataset
from src.hgnn import HGNN
from src.metrics import kfold_indices, spearmanr_np


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

CSV_PATH = "urn_mavedb_00001234-a-1_scores.csv"
PDB_PATH = "data/structures/TP53_RCSB.pdb"
CHAIN_ID = "A"
MSA_PATH = "data/msa/tp53_alignment.aln"

BATCH_SIZE = 1
EPOCHS = 120
LR = 1e-3
WD = 1e-4
PATIENCE = 15
VAL_FRAC = 0.15
DCA_SCALE = 1.0

OUT_PATH = "checkpoints/gnn_entropy_dca_cv_result.json"


def split_train_val(train_idx, seed):
    rng = np.random.default_rng(seed)
    idx = np.array(train_idx, copy=True)
    rng.shuffle(idx)
    n_val = max(1, int(round(VAL_FRAC * len(idx))))
    return idx[n_val:], idx[:n_val]


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            preds.append(float(out.detach().cpu().item()))
            trues.append(float(batch.y.view(-1).detach().cpu().item()))

    sp = spearmanr_np(np.array(preds), np.array(trues))
    return float(sp)


def train_one_fold(ds, train_idx, test_idx, fold_seed):
    tr_idx, val_idx = split_train_val(train_idx, fold_seed)

    train_subset = torch.utils.data.Subset(ds, tr_idx.tolist())
    val_subset = torch.utils.data.Subset(ds, val_idx.tolist())
    test_subset = torch.utils.data.Subset(ds, test_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    in_dim = ds.X.shape[1]
    model = HGNN(in_dim=in_dim).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()

    best_val_sp = -1e9
    best_state = None
    bad = 0

    for _ in range(EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            pred = model(batch).view(-1)
            loss = loss_fn(pred, batch.y.view(-1))
            loss.backward()
            opt.step()

        val_sp = evaluate(model, val_loader)

        if val_sp > best_val_sp:
            best_val_sp = val_sp
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    return evaluate(model, test_loader)


def main():
    print("Loading dataset (kNN edges + entropy + DCA)...")
    ds = TP53StructureDataset(
        CSV_PATH,
        PDB_PATH,
        chain_id=CHAIN_ID,
        graph_type="knn",
        msa_path=MSA_PATH,
        use_entropy=True,
        use_dca=True,
        dca_scale=DCA_SCALE,
        msa_cache_path="cache/tp53_msa_features.npz",
    )

    scores = []
    for fold, (train_idx, test_idx) in enumerate(kfold_indices(len(ds), 5, SEED)):
        print(f"\n===== Fold {fold + 1} =====")
        sp = train_one_fold(ds, train_idx, test_idx, fold_seed=SEED + fold + 1)
        print(f"Fold {fold + 1} Spearman: {sp:.4f}")
        scores.append(sp)

    scores = np.array(scores)
    mean_sp = float(scores.mean())
    std_sp = float(scores.std())

    print("\n==============================")
    print("GNN + Entropy + DCA 5-Fold CV Result")
    print(f"Mean Spearman: {mean_sp:.4f}")
    print(f"Std  Spearman: {std_sp:.4f}")
    print("==============================")

    os.makedirs("checkpoints", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(
            {
                "mean_spearman": mean_sp,
                "std_spearman": std_sp,
                "dca_scale": DCA_SCALE,
                "msa": MSA_PATH,
            },
            f,
            indent=2,
        )

    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
