"""
plots.py - Figures saved by evaluate.py.

  - plot_history          -> loss_curve.png, accuracy_curve.png (train vs val per epoch)
  - plot_confusion_matrix -> confusion_matrix.png
  - plot_roc              -> roc_curve.png

Kept separate from evaluate.py so that file stays focused on the numbers.
The colours are a colour-blind-safe pair (blue/orange) and the same style is
used in every figure.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to files; no window/display needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
MUTED = "#8a8984"
TRAIN_COLOR = "#2a78d6"  # blue
VAL_COLOR = "#eb6834"    # orange
# Single-hue blue ramp (light -> dark) for the confusion matrix counts.
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "legend.frameon": False,
    "legend.labelcolor": TEXT_SECONDARY,
    "font.size": 10,
})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def _plot_curve(history: dict, key: str, ylabel: str, title: str, path: Path,
                as_percent: bool) -> None:
    """Train vs val curve for one metric ("loss" or "acc") over the epochs."""
    epochs = [e["epoch"] for e in history["epochs"]]
    fig, ax = plt.subplots(figsize=(7, 4.2))

    for split, color in (("train", TRAIN_COLOR), ("val", VAL_COLOR)):
        values = [e[f"{split}_{key}"] for e in history["epochs"]]
        label = "Train" if split == "train" else "Validation"
        ax.plot(epochs, values, color=color, linewidth=2, marker="o", markersize=4, label=label)
        # Direct label at the end of each line, so it is readable without the legend.
        ax.annotate(label, (epochs[-1], values[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=TEXT_SECONDARY, fontsize=9)

    # Mark the epoch whose weights were saved as the best checkpoint.
    # (Both markers are explained in the legend, so no text can overlap the lines.)
    best = history.get("best_epoch")
    if best is not None:
        best_val = history["epochs"][best - 1][f"val_{key}"]
        ax.plot(best, best_val, linestyle="none", marker="o", markersize=10, color=VAL_COLOR,
                markeredgecolor=TEXT, markeredgewidth=1.5, zorder=5,
                label=f"Best checkpoint (epoch {best})")

    # EfficientNet: show where the backbone was unfrozen.
    unfreeze = history.get("unfreeze_epoch")
    if unfreeze is not None and unfreeze <= epochs[-1]:
        ax.axvline(unfreeze - 0.5, color=MUTED, linestyle="--", linewidth=1,
                   label=f"Backbone unfrozen (from epoch {unfreeze})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(epochs if len(epochs) <= 20 else epochs[::2])
    ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 1.5)  # leave room for the end labels
    if as_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="best")
    _save(fig, path)


def plot_history(history: dict, results_dir: Path, model_label: str) -> list[Path]:
    """Save loss_curve.png and accuracy_curve.png; return their paths."""
    loss_path = results_dir / "loss_curve.png"
    acc_path = results_dir / "accuracy_curve.png"
    _plot_curve(history, "loss", "Binary cross-entropy loss",
                f"Loss per epoch - {model_label}", loss_path, as_percent=False)
    _plot_curve(history, "acc", "Accuracy",
                f"Accuracy per epoch - {model_label}", acc_path, as_percent=True)
    return [loss_path, acc_path]


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path,
                          model_label: str) -> None:
    """2x2 grid: rows = true class, columns = predicted class.

    Each cell shows the count and its share of that true class (row %), so
    the diagonal cells read directly as per-class recall.
    """
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # pcolormesh with surface-coloured edges gives a small gap between cells.
    ax.pcolormesh(cm, cmap=BLUES, vmin=0, vmax=cm.max(), edgecolors=SURFACE, linewidth=3)
    row_totals = cm.sum(axis=1, keepdims=True)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            dark_cell = cm[i, j] > cm.max() * 0.5
            ax.text(j + 0.5, i + 0.5, f"{cm[i, j]:,}\n{cm[i, j] / row_totals[i, 0]:.1%}",
                    ha="center", va="center", fontsize=13,
                    color="white" if dark_cell else TEXT)

    names = [c.capitalize() for c in class_names]
    ax.set_xticks(np.arange(len(names)) + 0.5, names)
    ax.set_yticks(np.arange(len(names)) + 0.5, names)
    ax.tick_params(length=0)
    ax.invert_yaxis()  # first class on top, like a table
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix\n{model_label}")
    ax.set_aspect("equal")
    _save(fig, path)


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------
def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float,
             operating_point: tuple[float, float], threshold: float,
             path: Path, model_label: str) -> None:
    """ROC curve: true-positive rate vs false-positive rate over all thresholds.

    The dashed diagonal is a random guesser (AUC = 0.5); the closer the curve
    hugs the top-left corner, the better. The dot marks the threshold that is
    actually used for the Real/Fake decision.
    """
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1,
            label="Random guess (AUC = 0.500)")
    ax.plot(fpr, tpr, color=TRAIN_COLOR, linewidth=2, label=f"{model_label} (AUC = {auc:.3f})")

    x, y = operating_point
    ax.plot(x, y, marker="o", markersize=9, color=TRAIN_COLOR,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.annotate(f"threshold {threshold:g}\nFPR {x:.3f}, TPR {y:.3f}", (x, y),
                xytext=(14, -28), textcoords="offset points", color=TEXT, fontsize=9)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate (real images flagged as fake)")
    ax.set_ylabel("True positive rate (fakes caught)")
    ax.set_title(f"ROC curve\n{model_label}")
    ax.legend(loc="lower right")
    _save(fig, path)


# ---------------------------------------------------------------------------
# Reliability diagram (confidence calibration)
# ---------------------------------------------------------------------------
def plot_reliability(bins_before: list[dict], bins_after: list[dict], calibration: dict,
                     path: Path, model_label: str) -> None:
    """Confidence vs actual accuracy, before and after temperature scaling.

    Each point is one confidence bin (e.g. all predictions made with 90-95%
    confidence). A perfectly calibrated model lies on the dashed diagonal;
    points BELOW it mean the model is over-confident.
    """
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot([0.5, 1], [0.5, 1], color=MUTED, linestyle="--", linewidth=1,
            label="Perfect calibration")
    series = [
        (bins_before, VAL_COLOR, f"Before (ECE = {calibration['test_ece_before']:.3f})"),
        (bins_after, TRAIN_COLOR,
         f"After, T = {calibration['temperature']:.2f} (ECE = {calibration['test_ece_after']:.3f})"),
    ]
    for bins, color, label in series:
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], color=color,
                linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)
    # Confidence is always >= 50%, but accuracy within a bin can be lower, so
    # the y-axis extends downwards if needed (never cut off a point).
    lowest = min([b["accuracy"] for b in bins_before + bins_after] + [0.5])
    ax.set_xlim(0.49, 1.01)
    ax.set_ylim(max(0.0, lowest - 0.03), 1.01)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Confidence (probability of the predicted class)")
    ax.set_ylabel("Actual accuracy")
    ax.set_title(f"Reliability diagram (test set)\n{model_label}")
    ax.legend(loc="best")
    _save(fig, path)


# ---------------------------------------------------------------------------
# Results per source / generator
# ---------------------------------------------------------------------------
def plot_per_group(per_group: dict, unseen: dict, path: Path, model_label: str) -> None:
    """Horizontal bars: share of each source/generator classified correctly.

    Blue = sources seen during training (test images are still new images);
    orange = generators NEVER seen in training (the generalisation test).
    """
    rows = [(g["name"], g["correct_rate"], False) for g in per_group.values()]
    rows += [(g["name"], g["detection_rate"], True) for g in unseen.values()]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.4))
    ax.grid(axis="y", visible=False)
    for i, (name, value, is_unseen) in enumerate(rows):
        ax.barh(i, value, height=0.6, color=VAL_COLOR if is_unseen else TRAIN_COLOR)
        ax.text(min(value, 1.0) + 0.01, i, f"{value:.1%}", va="center", color=TEXT, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()  # first row on top
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Classified correctly (reals: kept as real, fakes: detected)")
    ax.set_title(f"Test results by source\n{model_label}")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TRAIN_COLOR, label="Seen in training (new test images)"),
                       Patch(color=VAL_COLOR, label="Generator never seen in training")],
              loc="lower right", fontsize=8)
    _save(fig, path)


# ---------------------------------------------------------------------------
# Comparison of runs (compare.py)
# ---------------------------------------------------------------------------
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order: blue, orange, aqua, yellow


def plot_comparison(rows: dict[str, list], run_labels: list[str], path: Path) -> None:
    """Grouped horizontal bars: for each source/generator, share correct per run."""
    runs = run_labels[:len(RUN_COLORS)]
    n_runs, names = len(runs), list(rows)
    bar_h = 0.8 / n_runs
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) * max(1, n_runs / 2) + 1.6))
    ax.grid(axis="y", visible=False)
    for j, run in enumerate(runs):
        ys = [i - 0.4 + bar_h * (j + 0.5) for i in range(len(names))]
        vals = [rows[n][j] if rows[n][j] is not None else 0 for n in names]
        ax.barh(ys, vals, height=bar_h * 0.9, color=RUN_COLORS[j], label=run)
        for y, v, raw in zip(ys, vals, (rows[n][j] for n in names)):
            if raw is not None:
                ax.text(v + 0.01, y, f"{v:.0%}", va="center", color=TEXT_SECONDARY, fontsize=7)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Classified correctly (reals: kept as real, fakes: detected)")
    ax.set_title("Model comparison by source / generator")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)
