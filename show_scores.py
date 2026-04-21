import argparse
import json
import os


ROOT = os.path.dirname(__file__)
DEFAULT_RESULTS = {
    "Sequence ΔESM + MLP": os.path.join(ROOT, "checkpoints", "seq_cv_result.json"),
    "Linear ΔESM (ridge)": os.path.join(ROOT, "checkpoints", "seq_linear_cv_result.json"),
    "GNN (baseline)": os.path.join(ROOT, "checkpoints", "gnn_cv_result.json"),
    "GNN + Entropy": os.path.join(ROOT, "checkpoints", "gnn_entropy_cv_result.json"),
    "GNN + Entropy + DCA": os.path.join(ROOT, "checkpoints", "gnn_entropy_dca_cv_result.json"),
    "GNN + NeRF Features": os.path.join(ROOT, "checkpoints", "gnn_nerf_cv_result.json"),
    "Our Model (details withheld)": os.path.join(ROOT, "checkpoints", "our_model_cv_result.json"),
}


def load_result(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    mean = data.get("mean_spearman")
    std = data.get("std_spearman")
    if mean is None or std is None:
        return None
    return float(mean), float(std)


def format_line(label: str, result):
    if result is None:
        return f"{label:<30} : MISSING"
    mean, std = result
    return f"{label:<30} : {mean:.4f} ± {std:.4f}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Showcase benchmark numbers for baselines and our internal model. "
            "No model code is executed here."
        )
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of labels to print (exact match).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("MODEL BENCHMARK (5-Fold CV)")
    print("=" * 60)
    print()

    labels = list(DEFAULT_RESULTS.keys())
    if args.only:
        labels = [l for l in labels if l in set(args.only)]

    for label in labels:
        result = load_result(DEFAULT_RESULTS[label])
        print(format_line(label, result))

    print("=" * 60)


if __name__ == "__main__":
    main()
