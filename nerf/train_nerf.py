import os
import sys
import math
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.structure import load_ca_coordinates


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PDB_PATH = "data/structures/TP53_RCSB.pdb"
CHAIN_ID = "A"

OUT_DIR = "data/nerf"
MODEL_PATH = os.path.join(OUT_DIR, "nerf_model.pt")
META_PATH = os.path.join(OUT_DIR, "nerf_meta.json")

IMG_W = 256
IMG_H = 256
N_VIEWS = 36
FOCAL = 220.0

N_SAMPLES = 64
NEAR = 0.6
FAR = 1.6

TRAIN_ITERS = 2000
RAYS_PER_STEP = 2048
LR = 5e-4


AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA2I = {a: i for i, a in enumerate(AA20)}
PALETTE = np.array([
    [0.91, 0.33, 0.25],
    [0.28, 0.70, 0.39],
    [0.34, 0.59, 0.94],
    [0.91, 0.74, 0.23],
    [0.79, 0.33, 0.90],
    [0.27, 0.84, 0.91],
    [0.95, 0.46, 0.35],
    [0.51, 0.88, 0.42],
    [0.96, 0.66, 0.28],
    [0.64, 0.56, 0.95],
    [0.92, 0.49, 0.69],
    [0.38, 0.72, 0.78],
    [0.87, 0.62, 0.37],
    [0.47, 0.80, 0.55],
    [0.99, 0.54, 0.30],
    [0.55, 0.44, 0.88],
    [0.85, 0.32, 0.55],
    [0.33, 0.62, 0.84],
    [0.95, 0.50, 0.21],
    [0.57, 0.76, 0.49],
], dtype=np.float32)


def normalize_coords(coords):
    c = coords.copy()
    center = c.mean(axis=0, keepdims=True)
    c = c - center
    scale = np.max(np.linalg.norm(c, axis=1))
    c = c / (scale + 1e-8)
    return c, center.squeeze(), float(scale)


def positional_encoding(x, L=6):
    out = [x]
    for i in range(L):
        f = 2.0 ** i
        out.append(torch.sin(f * x))
        out.append(torch.cos(f * x))
    return torch.cat(out, dim=-1)


class TinyNeRF(nn.Module):
    def __init__(self, hidden=128, pe_levels=6):
        super().__init__()
        self.pe_levels = pe_levels
        in_dim = 3 * (1 + 2 * pe_levels)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.sigma = nn.Linear(hidden, 1)
        self.rgb = nn.Linear(hidden, 3)

    def forward(self, x):
        x = positional_encoding(x, self.pe_levels)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        sigma = F.softplus(self.sigma(h))
        rgb = torch.sigmoid(self.rgb(h))
        return rgb, sigma, h


@torch.no_grad()
def make_target_field(points, colors, query):
    # query: [R, S, 3]
    R, S, _ = query.shape
    q = query.view(-1, 3)
    p = points[None, :, :]  # [1, N, 3]
    q = q[:, None, :]       # [RS, 1, 3]
    d2 = ((q - p) ** 2).sum(-1)
    w = torch.exp(-d2 / 0.01)
    w_sum = w.sum(-1, keepdim=True) + 1e-6
    sigma = w_sum.view(R, S, 1)
    c = (w @ colors) / w_sum
    rgb = c.view(R, S, 3)
    return rgb, sigma


def make_cameras(n_views):
    cams = []
    for i in range(n_views):
        angle = 2 * math.pi * i / n_views
        eye = np.array([math.cos(angle) * 2.0, math.sin(angle) * 2.0, 0.6], dtype=np.float32)
        cams.append(eye)
    return cams


def build_rays(eye, w, h, focal):
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    dirs = np.stack([(i - w * 0.5) / focal, -(j - h * 0.5) / focal, -np.ones_like(i)], axis=-1)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    # rotate to look at origin
    forward = -eye / np.linalg.norm(eye)
    up = np.array([0, 0, 1], dtype=np.float32)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    R = np.stack([right, up, forward], axis=1)  # [3,3]
    dirs = dirs @ R.T
    origins = np.broadcast_to(eye, dirs.shape)
    return origins.astype(np.float32), dirs.astype(np.float32)


def sample_rays(origins, dirs, n_samples, near, far):
    # Flatten to [R, 3] so output is [R, S, 3]
    origins = origins.reshape(-1, 3)
    dirs = dirs.reshape(-1, 3)
    t_vals = torch.linspace(near, far, n_samples, device=DEVICE)
    pts = origins[:, None, :] + dirs[:, None, :] * t_vals[None, :, None]
    return pts, t_vals


def volume_render(rgb, sigma, t_vals):
    # rgb: [R,S,3], sigma: [R,S,1]
    delta = t_vals[1:] - t_vals[:-1]
    delta = torch.cat([delta, delta[-1:]], dim=0)
    delta = delta.view(1, -1, 1)
    alpha = 1.0 - torch.exp(-sigma * delta)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=1), dim=1)[:, :-1]
    weights = alpha * T
    comp_rgb = (weights * rgb).sum(dim=1)
    return comp_rgb


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    coords, residues = load_ca_coordinates(PDB_PATH, chain_id=CHAIN_ID)
    coords, center, scale = normalize_coords(coords)
    colors = np.array([PALETTE[AA2I.get(r["aa"], 0)] for r in residues], dtype=np.float32)

    points = torch.tensor(coords, dtype=torch.float32, device=DEVICE)
    colors_t = torch.tensor(colors, dtype=torch.float32, device=DEVICE)

    cams = make_cameras(N_VIEWS)
    all_origins = []
    all_dirs = []
    for eye in cams:
        o, d = build_rays(eye, IMG_W, IMG_H, FOCAL)
        all_origins.append(o)
        all_dirs.append(d)

    all_origins = torch.tensor(np.stack(all_origins), device=DEVICE)  # [V,H,W,3]
    all_dirs = torch.tensor(np.stack(all_dirs), device=DEVICE)        # [V,H,W,3]

    model = TinyNeRF().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print("Device:", DEVICE)
    print("Training NeRF-like field...")

    for step in range(1, TRAIN_ITERS + 1):
        v = random.randrange(N_VIEWS)
        o = all_origins[v]
        d = all_dirs[v]

        # sample random rays
        ys = torch.randint(0, IMG_H, (RAYS_PER_STEP,), device=DEVICE)
        xs = torch.randint(0, IMG_W, (RAYS_PER_STEP,), device=DEVICE)
        o_r = o[ys, xs]
        d_r = d[ys, xs]

        pts, t_vals = sample_rays(o_r, d_r, N_SAMPLES, NEAR, FAR)

        with torch.no_grad():
            gt_rgb, gt_sigma = make_target_field(points, colors_t, pts)
            gt = volume_render(gt_rgb, gt_sigma, t_vals)

        rgb, sigma, _ = model(pts.view(-1, 3))
        rgb = rgb.view(RAYS_PER_STEP, N_SAMPLES, 3)
        sigma = sigma.view(RAYS_PER_STEP, N_SAMPLES, 1)
        pred = volume_render(rgb, sigma, t_vals)

        loss = F.mse_loss(pred, gt)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == 1:
            print(f"step {step:04d} | loss {loss.item():.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump({"center": center.tolist(), "scale": scale}, f, indent=2)
    print("Saved:", MODEL_PATH)
    print("Saved:", META_PATH)


if __name__ == "__main__":
    main()
