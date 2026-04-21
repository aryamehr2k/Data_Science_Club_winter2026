import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.structure import load_ca_coordinates, build_knn_graph, build_chain_graph
from src.esm_embed import get_esm2_residue_embeddings
from src.msa_features import load_or_compute_msa_features


AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA2I = {a: i for i, a in enumerate(AA20)}

AA3 = {
    "Ala":"A","Cys":"C","Asp":"D","Glu":"E","Phe":"F","Gly":"G","His":"H","Ile":"I","Lys":"K","Leu":"L",
    "Met":"M","Asn":"N","Pro":"P","Gln":"Q","Arg":"R","Ser":"S","Thr":"T","Val":"V","Trp":"W","Tyr":"Y",
}


def parse_hgvs_protein(s: str):
    s = str(s).strip()
    if s.startswith("p."):
        s = s[2:]

    m = re.match(r"^([A-Za-z]{3})(\d+)([A-Za-z]{3})$", s)
    if m:
        wt3, pos, mut3 = m.group(1), int(m.group(2)), m.group(3)
        wt = AA3.get(wt3.capitalize())
        mut = AA3.get(mut3.capitalize())
        return wt, pos, mut

    m = re.match(r"^([A-Z])(\d+)([A-Z])$", s)
    if m:
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
        return wt, pos, mut

    return None, None, None


def parse_mutation(s: str):
    return parse_hgvs_protein(s)


def apply_mutation(seq: str, pos_1based: int, mut_aa: str):
    i = pos_1based - 1
    if i < 0 or i >= len(seq):
        return None
    return seq[:i] + mut_aa + seq[i + 1:]


def build_hypergraph_indices(num_nodes: int, window: int = 2, add_global: bool = True):
    v_idx = []
    e_idx = []
    e = 0

    for center in range(num_nodes):
        start = max(0, center - window)
        end = min(num_nodes - 1, center + window)
        for n in range(start, end + 1):
            v_idx.append(n)
            e_idx.append(e)
        e += 1

    if add_global:
        for n in range(num_nodes):
            v_idx.append(n)
            e_idx.append(e)
        e += 1

    v_idx = torch.tensor(v_idx, dtype=torch.long)
    e_idx = torch.tensor(e_idx, dtype=torch.long)
    return v_idx, e_idx, e


class HyperData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "v_idx":
            return self.num_nodes
        if key == "e_idx":
            return int(self.num_edges)
        if key == "pos_idx":
            return self.num_nodes
        if key in {"mut_pos", "wt_idx", "mut_idx"}:
            return 0
        if key == "mut_mask":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


class TP53StructureDataset(Dataset):

    def __init__(
        self,
        csv_path,
        pdb_path,
        chain_id="A",
        k=16,
        esm_model="esm2_t6_8M_UR50D",
        graph_type="knn",
        msa_path=None,
        use_entropy=False,
        use_dca=False,
        dca_scale=1.0,
        msa_cache_path="cache/tp53_msa_features.npz",
        nerf_features_path=None,
        use_nerf=False,
    ):
        super().__init__()

        df = pd.read_csv(csv_path)

        # -------- Find mutation column dynamically --------
        mut_col = None
        for c in df.columns:
            if df[c].astype(str).str.contains("p.", regex=False).any():
                mut_col = c
                break
        if mut_col is None:
            raise ValueError("Mutation column not found.")

        # -------- Find score column dynamically --------
        score_col = None
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                score_col = c
        if score_col is None:
            raise ValueError("Score column not found.")

        coords, residues = load_ca_coordinates(pdb_path, chain_id=chain_id)
        self.resseq2idx = {r["resseq"]: i for i, r in enumerate(residues)}
        self.N = len(residues)

        seq = "".join([r["aa"] for r in residues])
        self.X = get_esm2_residue_embeddings(seq, model_name=esm_model)
        self.X = self.X[:self.N]

        self.entropy = None
        self.dca_scores = None
        self.nerf_features = None
        if msa_path and (use_entropy or use_dca):
            ent, dca = load_or_compute_msa_features(
                msa_path=msa_path,
                pdb_path=pdb_path,
                chain_id=chain_id,
                cache_path=msa_cache_path,
            )
            if use_entropy:
                self.entropy = torch.from_numpy(ent).float().view(-1, 1)
                self.X = torch.cat([self.X, self.entropy], dim=1)
            if use_dca:
                self.dca_scores = dca

        if use_nerf:
            if not nerf_features_path or not os.path.exists(nerf_features_path):
                raise FileNotFoundError(
                    "NeRF features not found. Run: python nerf/train_nerf.py "
                    "and python nerf/extract_features.py first."
                )
            nerf = np.load(nerf_features_path)
            if nerf.shape[0] != self.N:
                raise ValueError(
                    f"NeRF features length mismatch: got {nerf.shape[0]} residues, "
                    f"expected {self.N}."
                )
            self.nerf_features = torch.from_numpy(nerf).float()
            self.X = torch.cat([self.X, self.nerf_features], dim=1)

        if graph_type == "chain":
            edge_index_np = build_chain_graph(self.N)
        else:
            edge_index_np = build_knn_graph(coords, k=k)
        self.edge_index = torch.from_numpy(edge_index_np).long()

        self.edge_weight = None
        if self.dca_scores is not None:
            edge_weight = []
            for u, v in edge_index_np.T:
                score = float(self.dca_scores[int(u), int(v)])
                edge_weight.append(1.0 + dca_scale * score)
            self.edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

        self.v_idx, self.e_idx, self.num_edges = build_hypergraph_indices(
            self.N, window=2, add_global=True
        )

        rows = []
        for _, row in df.iterrows():
            wt, pos, mut = parse_hgvs_protein(row[mut_col])
            if wt is None or mut is None:
                continue
            if pos not in self.resseq2idx:
                continue
            if wt not in AA2I or mut not in AA2I:
                continue

            idx = self.resseq2idx[pos]
            y = float(row[score_col])
            rows.append((idx, AA2I[wt], AA2I[mut], y))

        if len(rows) == 0:
            raise ValueError("No usable mutations found — numbering mismatch.")

        self.rows = rows

        y_values = np.array([r[3] for r in rows])
        self.y_mean = float(y_values.mean())
        self.y_std = float(y_values.std() + 1e-8)

        print(f"Loaded {len(rows)} usable mutations.")
        print(f"Residues: {self.N}")
        print(f"ESM dim: {self.X.shape[1]}")
        print(f"y_mean={self.y_mean:.6f} y_std={self.y_std:.6f}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        pos, wt, mut, y = self.rows[idx]

        y = (y - self.y_mean) / self.y_std

        mut_mask = torch.zeros(self.N, dtype=torch.float32)
        if 0 <= pos < self.N:
            mut_mask[pos] = 1.0

        data = HyperData(
            x=self.X,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            y=torch.tensor([y], dtype=torch.float32)
        )

        data.pos_idx = torch.tensor(pos, dtype=torch.long)
        data.mut_pos = torch.tensor(pos, dtype=torch.long)
        data.wt_idx = torch.tensor(wt, dtype=torch.long)
        data.mut_idx = torch.tensor(mut, dtype=torch.long)
        data.mut_mask = mut_mask

        data.v_idx = self.v_idx
        data.e_idx = self.e_idx
        data.num_edges = int(self.num_edges)

        return data
