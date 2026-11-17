"""
robustness.py - How well do models hold up on real-world-quality images?

    python -m src.robustness --names effnet effnet_v1 cnn cnn_v1

Test-set accuracy measures performance on clean images that look exactly like
the training data. Real images are usually re-compressed, resized, blurred or
framed differently. This script applies those changes to every TEST image and
reports, for each condition: accuracy, share of reals kept as real, and share
of fakes detected.

Saves results/<name>/robustness.json per run and results/robustness.md, a
side-by-side table of all runs for the report.
"""

import argparse
import io
import json
from functools import partial

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

from src import config
from src.data import FaceDataset, get_transforms
from src.models import load_checkpoint, run_label
from src.utils import get_device, set_seed

S = config.IMG_SIZE


# ---- Image degradations (module-level functions so worker processes can use them)
def identity(im):
    return im


def jpeg(im, quality):
    buffer = io.BytesIO()
    im.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def rescale(im, factor):
    small = im.resize((int(S * factor), int(S * factor)), Image.Resampling.BILINEAR)
    return small.resize((S, S), Image.Resampling.BILINEAR)


def blur(im, radius):
    return im.filter(ImageFilter.GaussianBlur(radius))


def whatsapp(im):
    return jpeg(rescale(im, 0.7), 60)


def zoom(im, factor):
    """factor > 1: zoom in (centre crop); factor < 1: zoom out (shrink + mirror-pad)."""
    if factor > 1:
        c = int(S / factor)
        o = (S - c) // 2
        return im.crop((o, o, o + c, o + c)).resize((S, S), Image.Resampling.LANCZOS)
    small = np.asarray(im.resize((int(S * factor), int(S * factor)), Image.Resampling.LANCZOS))
    p = (S - small.shape[0]) // 2
    q = S - small.shape[0] - p
    return Image.fromarray(np.pad(small, ((p, q), (p, q), (0, 0)), mode="reflect"))


CONDITIONS = {
    "clean": identity,
    "JPEG q90": partial(jpeg, quality=90),
    "JPEG q70": partial(jpeg, quality=70),
    "JPEG q50": partial(jpeg, quality=50),
    "low-res (50%)": partial(rescale, factor=0.5),
    "blur (r=1)": partial(blur, radius=1),
    "WhatsApp-like": whatsapp,
    "zoom in 1.3x": partial(zoom, factor=1.3),
    "zoom out 0.75x": partial(zoom, factor=0.75),
}


class DegradedTestSet(Dataset):
    """The test split with one degradation applied to every image."""

    def __init__(self, degrade):
        self.base = FaceDataset("test")
        self.degrade = degrade
        self.transform = get_transforms(train=False)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        rel, label = self.base.samples[i]
        image = Image.open(self.base.root / rel).convert("RGB")
        return self.transform(self.degrade(image)), label


@torch.inference_mode()
def evaluate_runs(names: list[str], batch_size: int, num_workers: int) -> dict:
    device = get_device()
    models = {n: load_checkpoint(config.CHECKPOINT_DIR / f"{n}_best.pt", device) for n in names}
    results = {n: {} for n in names}
    for cond, fn in CONDITIONS.items():
        loader = DataLoader(DegradedTestSet(fn), batch_size=batch_size, num_workers=num_workers)
        preds = {n: [] for n in names}
        labels = []
        for x, y in loader:  # every model sees exactly the same degraded images
            x = x.to(device, non_blocking=True)
            for n, (model, _) in models.items():
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    preds[n].append((model(x).squeeze(1) > 0).long().cpu())
            labels.append(y)
        y = torch.cat(labels)
        for n in names:
            p = torch.cat(preds[n])
            results[n][cond] = {"accuracy": float((p == y).float().mean()),
                                "reals_correct": float((p[y == 0] == 0).float().mean()),
                                "fakes_detected": float((p[y == 1] == 1).float().mean())}
        print(f"  {cond:15s} " + "  ".join(f"{n}: {results[n][cond]['accuracy']:.1%}" for n in names),
              flush=True)
    return {n: {"label": run_label(n, models[n][1]["model_name"]), "conditions": results[n]}
            for n in names}


def main() -> None:
    p = argparse.ArgumentParser(description="Robustness of trained models to image degradations.")
    p.add_argument("--names", nargs="+", required=True, help="run names, e.g. effnet effnet_v1")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    args = p.parse_args()
    set_seed(config.SEED)

    print("Accuracy on the degraded test set:")
    results = evaluate_runs(args.names, args.batch_size, args.num_workers)
    for n, r in results.items():
        (config.RESULTS_DIR / n).mkdir(parents=True, exist_ok=True)
        with open(config.RESULTS_DIR / n / "robustness.json", "w") as f:
            json.dump(r, f, indent=2)

    # Markdown table for the report: accuracy (reals kept / fakes caught) per run.
    lines = ["| Condition | " + " | ".join(r["label"] for r in results.values()) + " |",
             "|---|" + "---|" * len(results)]
    for cond in CONDITIONS:
        cells = [f"{r['conditions'][cond]['accuracy']:.1%} "
                 f"({r['conditions'][cond]['reals_correct']:.0%} / {r['conditions'][cond]['fakes_detected']:.0%})"
                 for r in results.values()]
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    out = config.RESULTS_DIR / "robustness.md"
    out.write_text("Accuracy on the test set under each condition "
                   "(in brackets: reals kept as real / fakes detected).\n\n" + table + "\n")
    print(f"\n{table}\n\nSaved {out} and results/<name>/robustness.json")


if __name__ == "__main__":
    main()
