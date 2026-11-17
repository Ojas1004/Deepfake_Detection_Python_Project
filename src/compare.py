"""
compare.py - Side-by-side comparison of all evaluated runs (for the report).

    python -m src.compare

Reads results/<run>/metrics.json (from evaluate.py) and, if present,
results/<run>/robustness.json (from robustness.py) for every trained run, and
writes:
  - results/comparison.md   tables: overall, per source/generator, unseen
                            generators, robustness
  - results/comparison.png  share of each source classified correctly, per run
"""

import json

from src import config
from src.models import list_runs, run_label
from src.plots import plot_comparison


def load(run: str, name: str) -> dict | None:
    path = config.RESULTS_DIR / run / name
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    runs = [r for r in list_runs(config.CHECKPOINT_DIR) if load(r, "metrics.json")]
    if not runs:
        raise SystemExit("No evaluated runs found - run `python -m src.evaluate --name <run>` first.")
    metrics = {r: load(r, "metrics.json") for r in runs}
    labels = {r: run_label(r, metrics[r]["model"]) for r in runs}
    header = "| | " + " | ".join(labels[r] for r in runs) + " |\n|---|" + "---|" * len(runs)
    out = ["# Model comparison", "",
           "All numbers are on the same test set (data/processed, split `test`) and the same "
           "unseen-generator sets.", ""]

    out += ["## Overall (test set)", "", header]
    for key, name in [("accuracy", "Accuracy"), ("f1", "F1 (fake)"), ("roc_auc", "ROC-AUC"),
                      ("precision", "Precision (fake)"), ("recall", "Recall (fake)"),
                      ("specificity", "Specificity (real)")]:
        out.append(f"| {name} | " + " | ".join(f"{metrics[r][key]:.3f}" for r in runs) + " |")
    out.append("| Calibration error (ECE, after) | "
               + " | ".join(f"{metrics[r]['calibration']['test_ece_after']:.3f}" for r in runs) + " |")

    present = {g for r in runs for g in metrics[r].get("per_group", {})}
    order = config.REAL_GROUPS + config.FAKE_GROUPS  # reals first, then fakes, in a fixed order
    groups = [g for g in order if g in present] + sorted(present - set(order))
    out += ["", "## By source / generator (share classified correctly)", "", header]
    rows = {}
    for g in groups:
        vals = [metrics[r].get("per_group", {}).get(g, {}).get("correct_rate") for r in runs]
        rows[config.GROUP_NAMES.get(g, g)] = vals
        out.append(f"| {config.GROUP_NAMES.get(g, g)} | "
                   + " | ".join("-" if v is None else f"{v:.1%}" for v in vals) + " |")

    unseen = list(dict.fromkeys(g for r in runs for g in metrics[r].get("unseen", {})))
    if unseen:
        out += ["", "## Generators never seen in training", "",
                "Detection rate (share of fakes caught); ROC-AUC against the test-set reals.", "", header]
        for g in unseen:
            cells = []
            for r in runs:
                u = metrics[r].get("unseen", {}).get(g)
                cells.append("-" if u is None else f"{u['detection_rate']:.1%} (AUC {u['roc_auc_vs_test_reals']:.3f})")
                if u is not None:
                    rows.setdefault(config.GROUP_NAMES.get(g, g), [None] * len(runs))[runs.index(r)] = u["detection_rate"]
            out.append(f"| {config.GROUP_NAMES.get(g, g)} | " + " | ".join(cells) + " |")

    robust = {r: load(r, "robustness.json") for r in runs}
    if any(robust.values()):
        conds = list(next(v for v in robust.values() if v)["conditions"])
        out += ["", "## Robustness (test-set accuracy under real-world degradations)", "", header]
        for c in conds:
            out.append(f"| {c} | " + " | ".join(
                "-" if not robust[r] else f"{robust[r]['conditions'][c]['accuracy']:.1%}" for r in runs) + " |")

    path = config.RESULTS_DIR / "comparison.md"
    path.write_text("\n".join(out) + "\n")
    plot_comparison(rows, [labels[r] for r in runs], config.RESULTS_DIR / "comparison.png")
    print("\n".join(out))
    print(f"\nSaved {path} and {config.RESULTS_DIR / 'comparison.png'}")


if __name__ == "__main__":
    main()
