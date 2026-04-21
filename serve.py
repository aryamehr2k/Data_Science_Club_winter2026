"""
Local demo server for the TP53 mutation predictor.

- Serves the WebGL UI under /
- Serves precomputed mutation scores under /api/*
- Binds 0.0.0.0 so phones on the same Wi-Fi can hit it

Run:
    python serve.py            # default port 8000
    PORT=9000 python serve.py  # custom port

It prints the LAN URL so presentation-day observers can open it on a phone.
"""
import json
import math
import os
import socket
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


ROOT = Path(__file__).parent
WEBGL_DIR = ROOT / "webgl"
DATA_PATH = WEBGL_DIR / "data.json"

app = FastAPI(title="TP53 Mutation Predictor (demo)")


_data_cache = None


def get_data():
    global _data_cache
    if _data_cache is None:
        if not DATA_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"{DATA_PATH} not found. Run export_webgl.py first.",
            )
        with open(DATA_PATH) as f:
            _data_cache = json.load(f)
        mut_map = {}
        for m in _data_cache.get("mutations", []):
            mut_map[f"{m['idx']}:{m['mut']}"] = m
        _data_cache["_mut_map"] = mut_map
        _data_cache["_residue_by_idx"] = {
            r["idx"]: r for r in _data_cache.get("residues", [])
        }
        stats = _data_cache.get("score_stats", {})
        lo = stats.get("min", 0.0)
        hi = stats.get("max", 1.0)
        span = max(1.0, hi - lo)
        _data_cache["_pred_bounds"] = (lo - span, hi + span)
    return _data_cache


def _pred_is_plausible(score: float, data) -> bool:
    lo, hi = data["_pred_bounds"]
    return lo <= score <= hi


class MutationItem(BaseModel):
    residue_idx: int
    mut_aa: str


class PredictRequest(BaseModel):
    mutations: List[MutationItem]


class ScoredMutation(BaseModel):
    residue_idx: int
    resseq: int
    wt: str
    mut: str
    score: float
    z: float
    source: str  # "measured" | "predicted"


class PredictResponse(BaseModel):
    items: List[ScoredMutation]
    missing: List[str]
    n_valid: int
    aggregate_z_mean: float
    aggregate_z_sum: float
    combined_damage_pct: float
    risk_label: str
    risk_color: str
    note: str


def _label_from_z(z: float):
    if z < -1.0:
        return "High risk", "#ff5c5c"
    if z < -0.5:
        return "Elevated risk", "#ffb347"
    if z > 0.7:
        return "Potential gain", "#5fb1ff"
    return "Low/Neutral risk", "#7dd97d"


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/meta")
def meta():
    data = get_data()
    return {
        "label": data["meta"]["label"],
        "nodes": data["meta"]["nodes"],
        "measured_mutations": len(data.get("mutations", [])),
        "predicted_entries": len(data.get("predictions", {})),
        "score_stats": data.get("score_stats", {}),
        "pred_checkpoint": data["meta"].get("pred_checkpoint"),
    }


@app.get("/api/residues")
def residues():
    data = get_data()
    return data.get("residues", [])


@app.get("/api/score/{residue_idx}/{mut_aa}")
def score_one(residue_idx: int, mut_aa: str):
    mut_aa = mut_aa.upper()
    if mut_aa not in "ACDEFGHIKLMNPQRSTVWY":
        raise HTTPException(400, f"invalid AA: {mut_aa}")
    data = get_data()
    key = f"{residue_idx}:{mut_aa}"
    meas = data["_mut_map"].get(key)
    pred = data.get("predictions", {}).get(key)
    if meas is None and (pred is None or not _pred_is_plausible(float(pred), data)):
        raise HTTPException(404, f"no trustworthy score for {key}")
    score = float(meas["score"]) if meas else float(pred)
    source = "measured" if meas else "predicted"
    stats = data.get("score_stats", {})
    std = stats.get("std") or 1.0
    z = (score - stats.get("mean", 0.0)) / std
    return {"residue_idx": residue_idx, "mut_aa": mut_aa,
            "score": score, "z": z, "source": source}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Score a *set* of point mutations.
    Each mutation is looked up individually (measured if available, else predicted).
    The combined "damage" is an additive approximation from the per-mutation
    z-scores — the underlying GNN was trained on single mutations, so the
    combined number is for illustration, not a calibrated multi-mutant score.
    """
    if not req.mutations:
        raise HTTPException(400, "no mutations provided")

    data = get_data()
    stats = data.get("score_stats", {})
    mean = float(stats.get("mean", 0.0))
    std = float(stats.get("std") or 1.0)

    items: List[ScoredMutation] = []
    missing: List[str] = []
    z_sum = 0.0

    for m in req.mutations:
        mut_aa = m.mut_aa.upper()
        r = data["_residue_by_idx"].get(m.residue_idx)
        if r is None:
            missing.append(f"idx={m.residue_idx}")
            continue
        key = f"{m.residue_idx}:{mut_aa}"
        meas = data["_mut_map"].get(key)
        pred = data.get("predictions", {}).get(key)
        if meas is None and (pred is None or not _pred_is_plausible(float(pred), data)):
            missing.append(f"{r['aa']}{r['resseq']}{mut_aa}")
            continue

        score = float(meas["score"]) if meas else float(pred)
        source = "measured" if meas else "predicted"
        z = (score - mean) / std
        z_sum += z
        items.append(ScoredMutation(
            residue_idx=m.residue_idx,
            resseq=int(r["resseq"]),
            wt=r["aa"],
            mut=mut_aa,
            score=score,
            z=z,
            source=source,
        ))

    n = max(1, len(items))
    z_mean = z_sum / n
    damage = 1.0 / (1.0 + math.exp(z_mean))
    label, color = _label_from_z(z_mean)

    note = (
        "Single mutations use the trained model directly. "
        "Combined damage is an additive approximation over z-scores."
    )

    return PredictResponse(
        items=items,
        missing=missing,
        n_valid=len(items),
        aggregate_z_mean=z_mean,
        aggregate_z_sum=z_sum,
        combined_damage_pct=float(100.0 * damage),
        risk_label=label,
        risk_color=color,
        note=note,
    )


@app.get("/api/residue/{residue_idx}/atoms")
def residue_atoms(residue_idx: int):
    data = get_data()
    atoms = data.get("residue_atoms", {}).get(str(residue_idx), [])
    return {"residue_idx": residue_idx, "atoms": atoms}


app.mount("/", StaticFiles(directory=str(WEBGL_DIR), html=True), name="webgl")


def _lan_ip() -> Optional[str]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def main():
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    ip = _lan_ip()

    print("=" * 66)
    print("  TP53 Mutation Predictor — local demo server")
    print("=" * 66)
    print(f"  On this computer:      http://localhost:{port}")
    if ip and host == "0.0.0.0":
        print(f"  On the same Wi-Fi:     http://{ip}:{port}")
        print("  (phones / laptops on the same network can open that URL)")
    print("=" * 66)
    print()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
