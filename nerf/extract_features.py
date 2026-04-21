import json
import os
import sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.structure import load_ca_coordinates
from nerf.train_nerf import TinyNeRF, normalize_coords


PDB_PATH = "data/structures/TP53_RCSB.pdb"
CHAIN_ID = "A"
MODEL_PATH = "data/nerf/nerf_model.pt"
META_PATH = "data/nerf/nerf_meta.json"
OUT_PATH = "data/nerf/residue_nerf_features.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    coords, residues = load_ca_coordinates(PDB_PATH, chain_id=CHAIN_ID)
    coords, center, scale = normalize_coords(coords)

    model = TinyNeRF().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    pts = torch.tensor(coords, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        _, _, h = model(pts)

    feats = h.detach().cpu().numpy().astype(np.float32)
    np.save(OUT_PATH, feats)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
