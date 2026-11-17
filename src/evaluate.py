"""
evaluate.py - Measure a trained model on the held-out TEST set.

    python -m src.evaluate --model cnn
    python -m src.evaluate --model effnet
    python -m src.evaluate --name effnet_v1      # any trained run

Loads checkpoints/<name>_best.pt, predicts every test image and saves to
results/<name>/:
  - metrics.json          accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
                          per-generator results and UNSEEN-generator results
  - per_generator.png     share of each source/generator classified correctly
  - confusion_matrix.png
  - roc_curve.png
  - loss_curve.png, accuracy_curve.png   (from history.json written by train.py)
  - calibration.json, reliability_diagram.png   (confidence calibration, see calibration.py)

"Positive" class = FAKE (label 1), because the task is detecting fakes:
  - precision = of the images we flagged as fake, how many really were fake
  - recall    = of all the fake images, how many we caught
  - F1        = harmonic mean of precision and recall
  - ROC-AUC   = probability that a random fake gets a higher "fake" score than
                a random real image; it is threshold-independent (0.5 = random,
                1.0 = perfect).

Unseen generators (DALL-E 3, StyleGAN3) were never used for training or
validation. Their splits contain only fakes, so we report the DETECTION RATE
(share of fakes caught) and the ROC-AUC of those fakes against the test reals.
This is the most honest number for "will it catch a new kind of fake?".
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from tqdm import tqdm

from src import config
from src.calibration import (binary_nll, expected_calibration_error, fit_temperature,
                             reliability_bins, sigmoid)
from src.data import get_dataloader
from src.models import MODEL_NAMES, load_checkpoint, run_label
from src.plots import (plot_confusion_matrix, plot_history, plot_per_group, plot_reliability,
                       plot_roc)
from src.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    p.add_argument("--model", choices=MODEL_NAMES, default=None,
                   help="shortcut for --name <model> (the default run of that model)")
    p.add_argument("--name", default=None, help="run name: evaluates checkpoints/<name>_best.pt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="predict 'fake' when P(fake) >= threshold")
    args = p.parse_args()
    args.name = args.name or args.model
    if args.name is None:
        p.error("give --model or --name")
    return args


def per_group_results(groups: list[str], labels: np.ndarray, probs: np.ndarray,
                      threshold: float) -> dict:
    """For each source/generator: how many images, and the share classified correctly.

    For a real group that share is its specificity; for a fake group it is
    the detection rate (recall) for that generator.
    """
    preds = (probs >= threshold).astype(int)
    out = {}
    for g in dict.fromkeys(groups):  # keeps first-seen order
        idx = np.array([i for i, x in enumerate(groups) if x == g])
        out[g] = {"name": config.GROUP_NAMES.get(g, g), "n": int(len(idx)),
                  "class": config.CLASSES[int(labels[idx[0]])],
                  "correct_rate": float((preds[idx] == labels[idx]).mean())}
    return out


@torch.inference_mode()  # no gradients needed: faster and uses less memory
def predict(model, loader, device, desc: str = "test") -> tuple[np.ndarray, np.ndarray]:
    """Return (true labels, raw logits) for every image in `loader`.

    Logits rather than probabilities are returned so that calibration can
    rescale them (logit / T) before applying the sigmoid.
    """
    all_labels, all_logits = [], []
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits = model(images).squeeze(1)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels)
    return torch.cat(all_labels).numpy(), torch.cat(all_logits).numpy().astype(np.float64)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """All test-set metrics, with FAKE (1) as the positive class."""
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),  # uses the scores, not the thresholded preds
        # Specificity = recall of the REAL class: real images correctly let through.
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_class": classification_report(labels, preds, labels=[0, 1],
                                           target_names=config.CLASSES,
                                           output_dict=True, zero_division=0),
    }


def to_builtin(obj):
    """Convert numpy numbers inside nested dicts to plain floats/ints for JSON."""
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main() -> None:
    args = parse_args()
    set_seed(config.SEED)
    device = get_device()

    ckpt_path = config.CHECKPOINT_DIR / f"{args.name}_best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}\n"
                         f"Train first: python -m src.train --model <cnn|effnet> --name {args.name}")
    results_dir = config.RESULTS_DIR / args.name
    results_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_checkpoint(ckpt_path, device)
    label = run_label(args.name, checkpoint["model_name"])
    print(f"Loaded {ckpt_path} (epoch {checkpoint['epoch']}, "
          f"val acc {checkpoint['val_acc']:.4f})")

    # ---- Predict on the test set -----------------------------------------
    loader = get_dataloader("test", args.batch_size, args.num_workers)
    labels, logits = predict(model, loader, device)
    probs = sigmoid(logits)  # uncalibrated P(fake); calibration below never changes predictions

    # ---- Metrics ------------------------------------------------------------
    metrics = compute_metrics(labels, probs, args.threshold)
    metrics["per_group"] = per_group_results(loader.dataset.groups, labels, probs, args.threshold)

    # ---- Unseen generators: fakes the model never saw in training ------------
    real_probs = probs[labels == 0]
    metrics["unseen"] = {}
    for split in config.UNSEEN_SPLITS:
        try:
            u_loader = get_dataloader(split, args.batch_size, args.num_workers)
        except (ValueError, FileNotFoundError):
            continue  # split not present in this dataset
        u_labels, u_logits = predict(model, u_loader, device, desc=split)
        u_probs = sigmoid(u_logits)
        both = np.concatenate([real_probs, u_probs])
        both_labels = np.concatenate([np.zeros(len(real_probs)), u_labels])
        group = u_loader.dataset.groups[0]
        metrics["unseen"][group] = {
            "name": config.GROUP_NAMES.get(group, group), "n": int(len(u_labels)),
            "detection_rate": float((u_probs >= args.threshold).mean()),
            "roc_auc_vs_test_reals": float(roc_auc_score(both_labels, both)),
        }

    report = {
        "model": checkpoint["model_name"],
        "run": args.name,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "num_test_images": int(len(labels)),
        "threshold": args.threshold,
        "positive_class": "fake",
        **metrics,
    }

    # ---- Confidence calibration (temperature fitted on VALIDATION data) ------
    val_labels, val_logits = predict(model, get_dataloader("val", args.batch_size, args.num_workers),
                                     device, desc="val")
    temperature = fit_temperature(val_logits, val_labels)
    calibrated_probs = sigmoid(logits / temperature)
    calibration = {
        "temperature": temperature,
        "fitted_on": "validation set",
        "val_nll_before": binary_nll(val_logits, val_labels),
        "val_nll_after": binary_nll(val_logits / temperature, val_labels),
        "test_ece_before": expected_calibration_error(probs, labels),
        "test_ece_after": expected_calibration_error(calibrated_probs, labels),
    }
    with open(results_dir / "calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    report["calibration"] = calibration
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(to_builtin(report), f, indent=2)

    # ---- Figures -------------------------------------------------------------
    cm = confusion_matrix(labels, (probs >= args.threshold).astype(int), labels=[0, 1])
    plot_confusion_matrix(cm, config.CLASSES, results_dir / "confusion_matrix.png", label)

    fpr, tpr, _ = roc_curve(labels, probs)
    c = metrics["confusion_matrix"]
    operating_point = (c["fp"] / (c["fp"] + c["tn"]), c["tp"] / (c["tp"] + c["fn"]))
    plot_roc(fpr, tpr, metrics["roc_auc"], operating_point, args.threshold,
             results_dir / "roc_curve.png", label)

    plot_reliability(reliability_bins(probs, labels), reliability_bins(calibrated_probs, labels),
                     calibration, results_dir / "reliability_diagram.png", label)

    plot_per_group(metrics["per_group"], metrics["unseen"], results_dir / "per_generator.png", label)

    history_path = results_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            plot_history(json.load(f), results_dir, label)
    else:
        print(f"(no {history_path} - skipping loss/accuracy curves)")

    # ---- Summary -------------------------------------------------------------
    print(f"\nTest results - {label} ({len(labels)} images, threshold {args.threshold}):")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "specificity"):
        print(f"  {key:12s} {metrics[key]:.4f}")
    print(f"  confusion    TN={c['tn']}  FP={c['fp']}  FN={c['fn']}  TP={c['tp']}")
    print(f"  calibration  T={temperature:.3f}, test ECE {calibration['test_ece_before']:.4f} "
          f"-> {calibration['test_ece_after']:.4f}")
    print("\n  By source / generator (share classified correctly):")
    for g in metrics["per_group"].values():
        print(f"    {g['name']:32s} {g['correct_rate']:7.1%}   (n={g['n']}, {g['class']})")
    for g in metrics["unseen"].values():
        print(f"    {g['name']:32s} {g['detection_rate']:7.1%}   (n={g['n']}, fake, NEVER SEEN; "
              f"AUC vs reals {g['roc_auc_vs_test_reals']:.3f})")
    print(f"\nSaved metrics and plots to {results_dir}/")


if __name__ == "__main__":
    main()
