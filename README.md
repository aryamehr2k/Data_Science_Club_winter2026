**Protein Mutation Prediction (Sequence + GNN Baseline)**

This project showcases how a **structure‑aware GNN** learns mutation effects in TP53, and compares it to **sequence‑only baselines**. It also includes a **WebGL 3D visualization** of the structure graph and learned embeddings for presentations.

**What This Showcases**
- Protein structure → graph (kNN edges from a PDB file).
- GNN learns structure‑aware representations for mutation prediction.
- Sequence‑only baselines (ΔESM + MLP, ΔESM + ridge) underperform vs the GNN.
- A 3D WebGL viewer that visually contrasts structure vs embedding space.
- **Our Model** score is shown from saved results only (architecture withheld).

**Repository Layout**
- `show_scores.py` – print benchmark numbers from saved JSONs
- `predict.py` – legacy alias for `show_scores.py`
- `train_seq_cv.py` – sequence baseline (ΔESM + MLP)
- `train_seq_linear_cv.py` – linear baseline (ΔESM + ridge)
- `train_gnn_cv.py` – GNN baseline (5‑fold CV)
- `export_webgl.py` – export structure + embedding to WebGL
- `webgl/plotly.html` – browser 3D viewer
- `src/` – dataset, feature builders, and models

**Setup**
```bash
cd /home/kay/Documents/github/DSClub-Project/protein_hgnn
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Quick Check (No Training)**
Print whatever results are already saved in `checkpoints/`.
```bash
python show_scores.py
```

**Full Training (All Baselines + GNN)**
Run these in order. The first run builds a ΔESM cache and can take time.
```bash
python train_seq_cv.py
python train_seq_linear_cv.py
python train_gnn_cv.py
```

**Show Scores After Training**
```bash
python show_scores.py
```

**3D Visualization (WebGL)**
1) Export the structure + embedding:
```bash
python export_webgl.py --model gnn --index 0 --label "GNN (baseline)"
```

2) Start the local server:
```bash
cd webgl
python -m http.server 8000
```

3) Open in browser:
```
http://localhost:8000/plotly.html
```

**Presentation Tip**
In the viewer, click **Presentation Mode** to hide controls and auto‑rotate. Use **F11** for full‑screen.

**Notes**
- If any line shows `MISSING`, the corresponding JSON in `checkpoints/` has not been generated yet.
- **Our Model** score is reported from `checkpoints/hypergnn_cv_result.json` without releasing model code.

test
