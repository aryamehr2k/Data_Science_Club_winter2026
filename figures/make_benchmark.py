"""
Regenerate figures/benchmark.png from the same bar values shown in the
original chart, with the internal model labelled "Enigma model
(details withheld)".

Run:
    python figures/make_benchmark.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "benchmark.png")


LABELS = [
    "Sequence ΔESM + MLP",
    "GNN (pairwise)",
    "GCN (chain)",
    "GCN (kNN structure)",
    "Enigma model (details withheld)",
]
MEANS = [0.35, 0.65, 0.56, 0.59, 0.62]
STDS = [0.12, 0.07, 0.07, 0.04, 0.06]


def main():
    x = np.arange(len(LABELS))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        x,
        MEANS,
        yerr=STDS,
        color="#c8881e",
        edgecolor="#7a4f00",
        capsize=6,
        width=0.7,
    )

    ax.set_title("Model Benchmark (5-Fold CV)", fontsize=14)
    ax.set_ylabel("Spearman (mean ± std)")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
