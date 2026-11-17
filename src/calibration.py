"""
calibration.py - Making the confidence score trustworthy.

Neural networks are often OVER-confident: a model may say "99% fake" on images
it only gets right 90% of the time. A confidence score is only useful if it
is *calibrated*: among all predictions made with ~90% confidence, about 90%
should be correct.

Two tools:
  - Expected Calibration Error (ECE) measures how far confidence and actual
    accuracy are apart (0 = perfectly calibrated).
  - Temperature scaling fixes it. Logits are divided by one number T before
    the sigmoid:  P(fake) = sigmoid(logit / T).
      T > 1 softens over-confident predictions, T < 1 sharpens under-confident ones.
    T is chosen on the VALIDATION set (never the test set). Dividing by a
    positive T never changes which side of 0 a logit is on, so predictions,
    accuracy and ROC-AUC stay exactly the same; only the confidence changes.

evaluate.py fits T and saves it to results/<model>/calibration.json; the demo
app and explain.py read it from there.
"""

import json

import numpy as np

from src import config


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def binary_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    """Average binary cross-entropy (negative log-likelihood), computed stably."""
    # log(1 + e^z) - y*z  ==  BCE(sigmoid(z), y), without overflow for large |z|
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Return the temperature T that minimises validation NLL.

    A plain grid search over 2,000 values between 0.05 and 20 (log-spaced):
    with only one parameter this is fast, exact enough and easy to verify.
    """
    temperatures = np.logspace(np.log10(0.05), np.log10(20), 2000)
    losses = [binary_nll(logits / t, labels) for t in temperatures]
    return float(temperatures[int(np.argmin(losses))])


def reliability_bins(probs_fake: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Group predictions by confidence and compare confidence with real accuracy.

    For a binary classifier, confidence = probability of the PREDICTED class,
    so it lies between 0.5 and 1.0; we split that range into equal bins.
    """
    confidence = np.maximum(probs_fake, 1 - probs_fake)
    correct = (probs_fake >= 0.5).astype(int) == labels
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    bins = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (confidence >= lo if i == 0 else confidence > lo) & (confidence <= hi)
        if in_bin.any():
            bins.append({"lo": float(lo), "hi": float(hi), "count": int(in_bin.sum()),
                         "confidence": float(confidence[in_bin].mean()),
                         "accuracy": float(correct[in_bin].mean())})
    return bins


def expected_calibration_error(probs_fake: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE = weighted average of |accuracy - confidence| over the confidence bins."""
    n = len(labels)
    return float(sum(b["count"] / n * abs(b["accuracy"] - b["confidence"])
                     for b in reliability_bins(probs_fake, labels, n_bins)))


def load_temperature(model_name: str) -> float | None:
    """Temperature saved by evaluate.py, or None if the model hasn't been evaluated yet."""
    path = config.RESULTS_DIR / model_name / "calibration.json"
    if not path.exists():
        return None
    with open(path) as f:
        return float(json.load(f)["temperature"])
