import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from Bio.PDB import PDBParser

from src.dataset import TP53StructureDataset, parse_hgvs_protein, HyperData, AA20, AA2I
from src.structure import load_ca_coordinates, AA3
from src.hgnn import HGNN


class CompatHGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, aa_dim=64, dropout=0.2):
        super().__init__()
        self.h1 = GCNConv(in_dim, hidden_dim)
        self.h2 = GCNConv(hidden_dim, hidden_dim)
        self.aa_emb = nn.Embedding(20, aa_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + aa_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def encode(self, data, edge_weight=None):
        x = data.x
        edge_index = data.edge_index
        x = self.h1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.h2(x, edge_index, edge_weight)
        x = F.relu(x)
        return x

    def forward(self, data):
        edge_weight = getattr(data, "edge_weight", None)
        x = self.encode(data, edge_weight=edge_weight)

        m = data.mut_mask.unsqueeze(1)
        z = (x * m).sum(dim=0, keepdim=True)

        wt = self.aa_emb(data.wt_idx.view(-1))
        mut = self.aa_emb(data.mut_idx.view(-1))
        delta = mut - wt
        feat = torch.cat([z, delta], dim=1)

        out = self.head(feat).squeeze(1)
        return out.squeeze(0)


CSV_PATH = "urn_mavedb_00001234-a-1_scores.csv"
PDB_PATH = "data/structures/TP53_RCSB.pdb"
CHAIN_ID = "A"


def pca3(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:3].T


def build_adj(edge_index: np.ndarray, num_nodes: int):
    adj = [[] for _ in range(num_nodes)]
    for u, v in edge_index.T:
        adj[int(u)].append(int(v))
    return adj


def compute_hops(adj, num_nodes: int, start: int) -> np.ndarray:
    dist = np.full(num_nodes, fill_value=-1, dtype=np.int32)
    dist[start] = 0
    q = [start]
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def sample_edges(edge_index: np.ndarray, max_edges: int = 5000) -> np.ndarray:
    e = edge_index.shape[1]
    if e <= max_edges:
        return edge_index
    idx = np.random.choice(e, size=max_edges, replace=False)
    return edge_index[:, idx]


def load_model(in_dim: int, device: torch.device, ckpt_path: str):
    model = HGNN(in_dim=in_dim).to(device)

    loaded = False
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            loaded = True
        except Exception:
            loaded = False
    model.eval()
    return model, loaded


def load_prediction_model(in_dim: int, device: torch.device):
    gnn_path = "checkpoints/gnn.pt"
    if os.path.exists(gnn_path):
        model, loaded = load_model(in_dim, device, gnn_path)
        if loaded:
            return model, gnn_path

    compat_path = "checkpoints/hgnn_knn.pt"
    if os.path.exists(compat_path):
        model = CompatHGNN(in_dim=in_dim).to(device)
        try:
            model.load_state_dict(torch.load(compat_path, map_location=device), strict=False)
            model.eval()
            return model, compat_path
        except Exception:
            pass

    return None, None


def load_annotations(path, residues):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    ann_list = data.get("annotations", data if isinstance(data, list) else [])
    if not ann_list:
        return []

    resseq2idx = {r["resseq"]: i for i, r in enumerate(residues)}
    palette = [
        "#ff4d4d", "#4dd2ff", "#ffd24d", "#8aff4d", "#b84dff",
        "#ff8ad6", "#4dffb8", "#ff9c4d",
    ]

    out = []
    for i, ann in enumerate(ann_list):
        name = ann.get("name", f"ann_{i}")
        color = ann.get("color", palette[i % len(palette)])
        indices = set()
        for idx in ann.get("indices", []):
            if isinstance(idx, int) and idx >= 0:
                indices.add(idx)
        for r in ann.get("ranges", []):
            if not isinstance(r, (list, tuple)) or len(r) != 2:
                continue
            start, end = int(r[0]), int(r[1])
            if end < start:
                start, end = end, start
            for pos in range(start, end + 1):
                if pos in resseq2idx:
                    indices.add(resseq2idx[pos])
        indices = sorted([i for i in indices if i >= 0])
        if indices:
            out.append({"name": name, "color": color, "indices": indices})
    return out


def load_residue_atoms(pdb_path, chain_id, residues):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_path)
    model = next(structure.get_models())
    chain = model[chain_id]

    atoms_by_resseq = {}
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        resname = res.get_resname().upper()
        if AA3.get(resname) is None:
            continue
        resseq = int(res.get_id()[1])
        atoms = []
        for atom in res.get_atoms():
            elem = (atom.element or "").strip()
            if not elem:
                name = atom.get_name().strip()
                elem = name[0] if name else "C"
            if elem.upper() == "H":
                continue
            x, y, z = atom.get_coord().astype(np.float32).tolist()
            atoms.append({
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "e": elem.upper(),
                "name": atom.get_name().strip(),
            })
        atoms_by_resseq[resseq] = atoms

    atoms_by_idx = {}
    for i, r in enumerate(residues):
        atoms_by_idx[str(i)] = atoms_by_resseq.get(r["resseq"], [])
    return atoms_by_idx


def load_mutations(csv_path, residues):
    df = pd.read_csv(csv_path)

    mut_col = None
    for c in df.columns:
        if df[c].astype(str).str.contains("p.", regex=False).any():
            mut_col = c
            break
    if mut_col is None:
        raise ValueError("Mutation column not found.")

    score_col = None
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            score_col = c
    if score_col is None:
        raise ValueError("Score column not found.")

    resseq2idx = {r["resseq"]: i for i, r in enumerate(residues)}
    mutations = []
    scores = []
    for _, row in df.iterrows():
        wt, pos, mut = parse_hgvs_protein(row[mut_col])
        if wt is None or mut is None:
            continue
        if pos not in resseq2idx:
            continue
        idx = resseq2idx[pos]
        score = float(row[score_col])
        mutations.append({
            "idx": int(idx),
            "resseq": int(pos),
            "wt": wt,
            "mut": mut,
            "score": float(score),
        })
        scores.append(score)

    scores = np.array(scores, dtype=np.float32) if scores else np.array([0.0], dtype=np.float32)
    stats = {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }
    return mutations, stats


def build_prediction_map(ds, residues, model, device):
    if model is None:
        return {}, None

    model.eval()
    x = ds.X.to(device)
    edge_index = ds.edge_index.to(device)
    edge_weight = ds.edge_weight.to(device) if ds.edge_weight is not None else None
    v_idx = ds.v_idx.to(device)
    e_idx = ds.e_idx.to(device)
    num_edges = int(ds.num_edges)

    preds = {}
    with torch.no_grad():
        for idx, r in enumerate(residues):
            wt = AA2I[r["aa"]]
            mut_mask = torch.zeros(ds.N, dtype=torch.float32, device=device)
            mut_mask[idx] = 1.0

            data = HyperData(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=torch.tensor([0.0], dtype=torch.float32, device=device),
            )
            data.v_idx = v_idx
            data.e_idx = e_idx
            data.num_edges = num_edges
            data.pos_idx = torch.tensor(idx, dtype=torch.long, device=device)
            data.mut_pos = torch.tensor(idx, dtype=torch.long, device=device)
            data.wt_idx = torch.tensor(wt, dtype=torch.long, device=device)
            data.mut_mask = mut_mask

            for aa in AA20:
                mut = AA2I[aa]
                data.mut_idx = torch.tensor(mut, dtype=torch.long, device=device)
                pred = float(model(data).detach().cpu().item())
                raw = pred * ds.y_std + ds.y_mean
                preds[f"{idx}:{aa}"] = float(raw)

    return preds, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gnn"], default="gnn")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max_edges", type=int, default=5000)
    parser.add_argument("--out", type=str, default="webgl/data.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--ann", type=str, default=None, help="path to annotations json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = TP53StructureDataset(CSV_PATH, PDB_PATH, chain_id=CHAIN_ID)
    idx = int(np.clip(args.index, 0, len(ds) - 1))
    data = ds[idx]
    mut_pos = int(data.mut_pos.item()) if hasattr(data, "mut_pos") else int(data.pos_idx.item())

    coords, residues = load_ca_coordinates(PDB_PATH, chain_id=CHAIN_ID)
    coords3d = coords.astype(np.float32)
    residue_list = [
        {"idx": i, "resseq": int(r["resseq"]), "aa": r["aa"]}
        for i, r in enumerate(residues)
    ]

    edge_index = ds.edge_index.detach().cpu().numpy()
    edge_draw = sample_edges(edge_index, max_edges=args.max_edges)

    adj = build_adj(edge_index, ds.N)
    hops = compute_hops(adj, ds.N, mut_pos)
    hops = np.where(hops >= 0, hops, hops.max() + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = "checkpoints/gnn.pt"
    model, loaded = load_model(in_dim=ds.X.shape[1], device=device, ckpt_path=ckpt)
    pred_model, pred_ckpt = load_prediction_model(in_dim=ds.X.shape[1], device=device)

    with torch.no_grad():
        z = model.encode(data.to(device)).detach().cpu().numpy()

    emb3d = pca3(z).astype(np.float32)

    annotations = load_annotations(args.ann, residues)
    mutations, score_stats = load_mutations(CSV_PATH, residues)
    residue_atoms = load_residue_atoms(PDB_PATH, CHAIN_ID, residues)
    predictions, pred_ok = build_prediction_map(ds, residues, pred_model, device)

    label = args.label
    if label is None:
        label = "GNN (baseline)"

    out = {
        "struct": coords3d.tolist(),
        "embed": emb3d.tolist(),
        "edges": edge_draw.T.tolist(),
        "hop": hops.astype(int).tolist(),
        "mut": int(mut_pos),
        "annotations": annotations,
        "residues": residue_list,
        "mutations": mutations,
        "score_stats": score_stats,
        "predictions": predictions,
        "y_stats": {"mean": float(ds.y_mean), "std": float(ds.y_std)},
        "residue_atoms": residue_atoms,
        "meta": {
            "label": label,
            "checkpoint": bool(loaded),
            "nodes": int(ds.N),
            "edges": int(edge_draw.shape[1]),
            "pdb_path": PDB_PATH,
            "chain_id": CHAIN_ID,
            "pred_checkpoint": pred_ckpt or None,
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
