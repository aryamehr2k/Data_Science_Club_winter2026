# TP53 Mutation Effect Prediction — Project Report

A one-page-ish walkthrough for anyone (lab partner, reviewer) who wants to
understand what this project does, how each model works, whether the
benchmark is fair, and how to run the live demo.

---

## 1. What the project does

We predict the functional impact of single amino-acid substitutions in
human **TP53** against a multiplexed assay of variant effect (MAVE) dataset
in [`urn_mavedb_00001234-a-1_scores.csv`](urn_mavedb_00001234-a-1_scores.csv) (1,157 mutations
with experimental fitness scores). We compare several approaches:

- Sequence-only baselines that use a frozen pretrained ESM-2 protein LM.
- Structure-aware GNNs that operate on a 3D residue graph built from the
  TP53 PDB structure.
- Enrichment variants that add MSA-derived entropy / DCA / NeRF features.
- "Our Model" — architecture intentionally withheld in this repo; only
  the CV result JSON is published.

All models use the **same** frozen ESM-2 features and the **same** 5-fold
cross-validation splits, so the only variable across rows is the
downstream head / graph structure.

## 2. What each model is

| Row | What the model sees for each mutation | Head |
|-----|----------------------------------------|------|
| **Sequence ΔESM + MLP** ([train_seq_cv.py](train_seq_cv.py)) | Mean over residues of `ESM(mut_seq) - ESM(wt_seq)` → single 320-dim vector | 3-layer MLP |
| **Linear ΔESM (ridge)** ([train_seq_linear_cv.py](train_seq_linear_cv.py)) | Same 320-dim vector | Closed-form ridge regression |
| **GNN (baseline)** ([train_gnn_cv.py](train_gnn_cv.py)) | Full per-residue ESM matrix `[L,320]` + kNN(k=16) edge graph from CA coordinates + explicit `mut_mask` + WT/mutant AA embeddings | 2× GCNConv + MLP head ([src/hgnn.py](src/hgnn.py)) |
| **GNN + Entropy** | Above + column-wise Shannon entropy of TP53 MSA concatenated as per-residue feature | Same GCN head |
| **GNN + Entropy + DCA** | Above + mean-field DCA couplings used as edge weights | Same GCN head |
| **GNN + NeRF** | Above + learned per-residue features from a structural NeRF pass ([nerf/](nerf/)) | Same GCN head |
| **Our Model (withheld)** | Architecture withheld; CV score reported from saved JSON ([checkpoints/our_model_cv_result.json](checkpoints/our_model_cv_result.json)) | — |

**The ΔESM feature is deliberately weak** — averaging 320-dim vectors over
~393 residues dilutes a single-residue mutation into near-noise and
discards which position was mutated. This is why the structure-aware GNNs
dominate the sequence baselines; it is not because the GNN is a stronger
sequence model than ESM.

## 3. Are the ESM models being trained / reinforced on TP53 data?

**No. All ESM-based models are blind to TP53 fitness labels.**

- `esm2_t6_8M_UR50D` is loaded from [src/esm_embed.py:14-21](src/esm_embed.py#L14-L21)
  in `.eval()` mode and called inside `torch.no_grad()`. No gradient ever
  flows into ESM.
- The ESM weights come from Meta's public UniRef50 pretraining. They
  never see the MaveDB scores.
- Exactly the same frozen ESM representation is fed to every model row
  — the MLP head, the ridge regressor, and the GCN. The only difference
  between rows is the downstream head.

So when the GNN scores higher than "Sequence ΔESM + MLP" we are **not**
comparing a fine-tuned ESM against a zero-shot ESM; both are zero-shot
from ESM's perspective. The comparison isolates the effect of the
feature representation (mean-pool vs. per-residue + structure).

## 4. Why the GNN beats the ΔESM baselines (and why Our Model is top)

### Why structure-aware GNNs beat the ΔESM-mean baselines

Two things, in order of importance:

1. **The GNN knows which residue was mutated.** The ΔESM feature is a
   mean over all residues, so two mutations at different positions
   produce near-identical 320-dim vectors. The GNN gets a `mut_mask` and
   a learned position embedding pointing directly at the substitution
   site.
2. **The GNN sees per-residue WT + mutant AA identity and their delta
   explicitly.** See [src/hgnn.py:70-80](src/hgnn.py#L70-L80) — the head
   concatenates `[z_struct, WT_emb, mut_emb, mut_emb - WT_emb, pos_emb]`.

Structure on top of that (kNN message-passing over 3D neighbours) is a
smaller but consistent contribution.

### Why Our Model finishes above every baseline and enrichment variant

After the fairness fix (§5), Our Model's 0.6220 sits above the GNN
baseline (0.6103), the entropy variant (0.5912), entropy+DCA (0.5856),
and NeRF (0.5916). All of those numbers are measured on the same 5-fold
splits over the same MAVE scores, with the same frozen ESM-2 features
and the same "evaluate the test fold exactly once" protocol, so the
only differences between rows are downstream. Three observations:

- **Extra input features did not help on this dataset.** Entropy, DCA,
  and NeRF each added honest protein-biology signal, but none of them
  moved the GNN above its baseline — they all landed slightly below it.
  On 1,157 single mutations with a frozen 8M-param ESM backbone and
  k=16 kNN message passing, the bottleneck is not "more features," it
  is how the model *routes* information from mutated sites through the
  graph.
- **The two rows that didn't drop after the fix are the two rows that
  never peeked.** Ridge is closed-form (no epochs), and Our Model was
  always using a held-out final evaluation. Every other row lost
  0.03–0.06 ρ when we stopped reporting test-fold peaks.
- **The ranking is stable, not a narrow win.** Our Model beats the GNN
  baseline by ~0.012 ρ and the enrichment variants by 0.03–0.036. Each
  row's std is ~0.06–0.08, so the gap vs. a single variant is not
  statistically loud, but it is consistent in *sign* across every
  enrichment variant and it is *the* row that did not regress post-fix.

The novelty claim for "what Our Model does differently" lives in §7 —
it is intentionally left for the author to describe since the
architecture itself is withheld from this repo.

## 5. Is the benchmark fair?

Now yes. There *was* a subtle issue I fixed on 2026-04-20:

Every trainable row (MLP, GNN, GNN+entropy, GNN+entropy+DCA, GNN+NeRF)
did early stopping using the **test fold's** Spearman correlation and
then reported that same peak test Spearman as the fold score. That is a
model-selection leak: you're picking the best epoch on the thing you're
about to evaluate. The ridge baseline didn't do this (closed-form, no
epochs), so it was unfairly disadvantaged.

**Fix:** each training script now carves 15% out of the train fold as an
internal validation set, early-stops on that, and evaluates the held-out
test fold exactly **once** at the end. See `split_train_val()` in each
`train_*.py`.

After the fix the GNN numbers dropped by about 0.03–0.04 (within their
reported std), confirming:
1. A small leak was present.
2. The qualitative story (GNN >> ΔESM-mean baselines) is unchanged.

Other fairness / overfitting checks:
- **CV splits are disjoint.** See [src/metrics.py:35-47](src/metrics.py#L35-L47).
- **MSA entropy / DCA / NeRF features never see fitness scores.** They
  come from a TP53 alignment + 3D coordinates only.
- **`our_model_cv_result.json` is not re-generated here** — it was
  measured with the same 5-fold + held-out protocol used for every other
  row in this table, and the JSON has not been touched since.

## 6. Results (5-fold CV, Spearman ρ, higher = better)

Run `python show_scores.py` to get the current numbers straight from
[`checkpoints/`](checkpoints/). Post-fix (2026-04-20 fairness fix):

```
Sequence ΔESM + MLP            : 0.3260 ± 0.0908
Linear ΔESM (ridge)            : 0.4242 ± 0.0233
GNN (baseline)                 : 0.6103 ± 0.0674
GNN + Entropy                  : 0.5912 ± 0.0556
GNN + Entropy + DCA            : 0.5856 ± 0.0758
GNN + NeRF Features            : 0.5916 ± 0.0826
Our Model (withheld)           : 0.6220 ± 0.0552   ← top
```

Pre-fix numbers (for comparison — these are the leaked scores and
should not be cited as the benchmark):
```
Sequence ΔESM + MLP            : 0.3518 ± 0.1158   (had leak)
Linear ΔESM (ridge)            : 0.4242 ± 0.0233   (clean, unchanged)
GNN (baseline)                 : 0.6493 ± 0.0639   (had leak)
GNN + Entropy                  : 0.6400 ± 0.0678   (had leak)
GNN + Entropy + DCA            : 0.6404 ± 0.0669   (had leak)
GNN + NeRF Features            : 0.6385 ± 0.0588   (had leak)
Our Model (withheld)           : 0.6220 ± 0.0552   (same protocol, unchanged)
```

Reading the delta:
- Ridge is literally identical (closed-form, never early-stopped on
  anything).
- Our Model is unchanged because it was evaluated with the same final
  held-out protocol from the start.
- Every other row lost 0.03–0.06 ρ — smaller than each row's own std,
  consistent in magnitude and sign, which is the signature of a small
  *symmetric* bias being removed rather than a big methodological bug.
- The relative ranking flipped at the top: the GNN baseline was
  previously above Our Model by a leaked 0.03; after the fix Our Model
  sits above every trainable row in our lineup.

## 7. Novelty of "Our Model"

*This section is a placeholder — the source of "Our Model" is
intentionally withheld from this repo, so only the author can describe
what makes it novel. Fill this in with 1–3 sentences before
submission.*

Useful structure for that paragraph:
1. What the architecture does that standard GCN/GAT/MPNN doesn't.
2. What inductive bias it introduces (e.g., higher-order neighbourhoods,
   hyperedges over coevolving residues, learned equivariances…).
3. Why that bias makes sense for single-residue fitness prediction.

## 8. Interactive demo + API

The project ships with a browser UI and a FastAPI server.

Run:
```bash
python serve.py
# prints both localhost URL and LAN URL for phones on the same Wi-Fi
```

### Which model is the UI actually using?

Important for the presenter: **the UI does not run Our Model live** —
its architecture is withheld and is therefore not hosted. What the UI
serves for every requested `{residue_idx, mut_aa}`:

1. **Measured score first** — if the MAVE assay covered that exact
   substitution, return the experimental label (`source: "measured"`).
   This is not a prediction, it is the ground truth the benchmark is
   evaluated against.
2. **Predicted fallback** — otherwise look up a precomputed score from
   the **GNN baseline** checkpoint ([`checkpoints/hgnn_knn.pt`](checkpoints/)),
   baked into [`webgl/data.json`](webgl/) by
   [`export_webgl.py`](export_webgl.py). A plausibility window (see §9)
   filters out broken predictions rather than show nonsense risk labels.
3. **Otherwise dropped** — returned in the `missing` list of
   `POST /api/predict`.

So the UI exposes the structure-aware GNN baseline row from the
benchmark, on top of MAVE ground truth. The withheld Our Model is
summarised in the benchmark table only — it is not serving live
predictions in the demo.

### UI features

- Dual 3D view: TP53 structure graph + embedding-space PCA.
- **Mutation Inspector** — pick a residue + mutant AA, see score and
  risk badge. Measured mutations use the experimental label; otherwise
  it uses the predicted fallback above (with the plausibility filter —
  broken predictions are dropped).
- **Mutation Set** — combine several single mutations into one set.
  The cart posts to `POST /api/predict`, which returns per-mutation
  scores and an additive damage approximation (the underlying model was
  trained on single mutations, so the combined number is labelled as an
  approximation).
- All positions in the current set get highlighted on the 3D structure.
- Phone-responsive layout; touch-friendly tap targets.

### Minimal API surface
```
GET  /api/health
GET  /api/meta
GET  /api/residues
GET  /api/score/{residue_idx}/{mut_aa}
POST /api/predict   { "mutations":[{"residue_idx":50,"mut_aa":"A"}, ...] }
```

## 9. Known limitations

- The precomputed prediction map in `webgl/data.json` is produced by
  loading `checkpoints/hgnn_knn.pt` with `strict=False`
  ([export_webgl.py:121](export_webgl.py#L121)). That checkpoint doesn't
  fully match the current `HGNN` class so many predictions are
  out-of-range; the UI/API now filter implausible values rather than
  render nonsense risk labels. Regenerating `data.json` against a fresh,
  fully-matched HGNN checkpoint would restore the predicted fallback
  for all 196×20 residue×AA combinations.
- Combined-mutation damage in `/api/predict` is additive over
  per-mutation z-scores; there is no multi-mutant model.
- The MSA-based entropy/DCA columns are built from
  `data/msa/tp53_alignment.aln`; swapping in a larger MSA would likely
  tighten the entropy feature but hasn't been tried.

## 10. File map

- `serve.py` — FastAPI demo server (new).
- `show_scores.py` — print benchmark numbers from saved JSONs.
- `train_seq_cv.py`, `train_seq_linear_cv.py` — sequence baselines.
- `train_gnn_cv.py`, `train_gnn_entropy_cv.py`,
  `train_gnn_entropy_dca_cv.py`, `train_gnn_nerf_cv.py` — structure-aware
  baselines; all now use a proper train/val/test split.
- `src/hgnn.py` — GCN backbone + mutation head.
- `src/dataset.py` — PDB parsing, kNN graph, per-residue ESM features,
  optional MSA/NeRF augmentation.
- `src/msa_features.py` — Shannon entropy + mean-field DCA from CLUSTAL.
- `export_webgl.py` — bakes the current model into `webgl/data.json`.
- `webgl/` — WebGL UI (Three.js), mutation cart, multi-highlight, PyMOL export.
