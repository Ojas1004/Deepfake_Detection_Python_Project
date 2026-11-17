"""
wild_eval.py - "In the wild" test: modern, convincing AI images vs real photos.

    python -m src.wild_eval                      # build the image set (once) + evaluate all detectors
    python -m src.wild_eval --extra photo.png    # also print every detector's verdict on your own image(s)
    python -m src.wild_eval --folder data/raw/tpdne/images --label fake --name tpdne --sample 2000
                                                 # detection rate of every detector on a folder of known fakes (or reals)

Our normal test set contains many older fakes (2019 StyleGAN, SD 1.5) that are
easy to spot. This script measures detectors on recent, realistic AI images:

  OpenFakeTiny (ComplexDataLab, CC BY-NC 4.0), two test sets:
    - core:   images from current generators (GPT-image-1.5, Midjourney 7,
              FLUX.2, Z-Image, Seedream 5, Nano Banana Pro, Veo 3 / Wan video
              frames) and real photos,
    - reddit: AI images and real photos as actually posted on Reddit.
  Only images with a clearly visible face (>= 96 px) are kept; anime-style
  images (Illustrious) are excluded.

Detectors compared (the external ones are NOT trained in this project):
  - our trained runs (checkpoints/*_best.pt): given the aligned face crop,
  - CommunityForensics ViT (CVPR 2025) and other public Hugging Face detectors,
    pinned to exact versions: given the whole image, as they were designed.

Note: only ~65 real photos with faces exist in these sets, so the "reals kept
as real" numbers have a wide margin of error (roughly +/-10 percentage points).
"""

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src import config

WILD_DIR = config.RAW_DIR / "openfake_tiny"
INDEX = WILD_DIR / "faces_index.csv"
OPENFAKE_TINY = "https://huggingface.co/datasets/ComplexDataLab/OpenFakeTiny/resolve/main/{part}/test-00000-of-00001.parquet"
MIN_FACE = 96
EXCLUDED_MODELS = {"illustrious"}  # anime-style generator, not relevant to photo deepfakes

# External detectors: (Hugging Face id, pinned commit, short name)
HF_DETECTORS = [
    ("Ateeqq/ai-vs-human-image-detector", "60e82406916921b823616bee33397baab38af3f0", "SigLIP AI-vs-human (Ateeqq)"),
    ("haywoodsloan/ai-image-detector-deploy", "4ae1822603cb1ec980c217cfde1ef1b8e0477a16", "SwinV2 AI detector (haywoodsloan)"),
    ("prithivMLmods/Deep-Fake-Detector-v2-Model", "3a99ae26f52c7ac7c3a53103b6cf3a8b617f7093", "ViT deepfake detector (prithivMLmods)"),
]


# ---------------------------------------------------------------------------
# 1. Build the image set (once)
# ---------------------------------------------------------------------------
def build_index() -> None:
    import urllib.request

    import pyarrow.parquet as pq

    from src.faces import FaceAligner
    from src.remote_zip import USER_AGENT, _hf_token

    aligner, rows = FaceAligner(), []
    for subset in ("core", "reddit"):
        path = WILD_DIR / f"{subset}.parquet"
        if not path.exists():
            WILD_DIR.mkdir(parents=True, exist_ok=True)
            headers = {"User-Agent": USER_AGENT}
            if _hf_token():
                headers["Authorization"] = f"Bearer {_hf_token()}"
            print(f"downloading OpenFakeTiny {subset} test set (~0.8 GB) ...")
            req = urllib.request.Request(OPENFAKE_TINY.format(part=subset), headers=headers)
            with urllib.request.urlopen(req) as resp, open(path, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
        n = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=50, columns=["image", "label", "model"]):
            for r in batch.to_pylist():
                n += 1
                try:
                    img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                except OSError:
                    continue
                found = aligner.detect(np.asarray(img))
                crop = 0.0
                if found:
                    quad = aligner.ffhq_quad(found[0])
                    crop = float(np.hypot(*(quad[3] - quad[0])))
                keep = crop >= MIN_FACE and r["model"] not in EXCLUDED_MODELS
                out = WILD_DIR / "images" / subset / r["label"] / f"{subset}_{n:04d}_{r['model']}.png"
                if keep:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    img.save(out)
                rows.append({"subset": subset, "label": r["label"], "model": r["model"],
                             "crop": round(crop), "path": out.relative_to(config.PROJECT_ROOT).as_posix() if keep else ""})
    with open(INDEX, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_items() -> list[dict]:
    with open(INDEX, newline="") as f:
        return [r for r in csv.DictReader(f) if r["path"] and r["model"] not in EXCLUDED_MODELS]


# ---------------------------------------------------------------------------
# 2. Detectors: each returns P(fake) for a list of PIL images
# ---------------------------------------------------------------------------
class HFDetectorBatch:
    """Wrapper so every detector here exposes p_fake(list of images) -> array."""

    def __init__(self, model_id: str, revision: str, device: torch.device):
        from src.pretrained import HFDetector

        self.det = HFDetector(model_id, revision, device)

    def p_fake(self, images):
        return self.det.p_fake_batch(images)


class CommunityForensics:
    """Our wrapper (src/pretrained.py), given the whole image."""

    def __init__(self, device):
        from src.pretrained import PretrainedDetector

        self.det = PretrainedDetector(device)

    def p_fake(self, images):
        return 1 / (1 + np.exp(-self.det.logits(images)))


class ProjectRun:
    """One of OUR trained models, given the aligned face (like its training data)."""

    def __init__(self, run: str, device: torch.device):
        from src.explain import EVAL_TRANSFORM, prepare_input
        from src.faces import FaceAligner
        from src.models import load_checkpoint

        self.model, _ = load_checkpoint(config.CHECKPOINT_DIR / f"{run}_best.pt", device)
        self.aligner, self.prepare, self.transform, self.device = FaceAligner(), prepare_input, EVAL_TRANSFORM, device

    @torch.inference_mode()
    def p_fake(self, images):
        x = torch.stack([self.transform(self.prepare(im, self.aligner)[0]) for im in images]).to(self.device)
        return torch.sigmoid(self.model(x).squeeze(1).float()).cpu().numpy()


def detectors(device):
    """(name, factory) for every detector, loaded one at a time to save GPU memory."""
    from src.models import list_runs, load_checkpoint, run_label
    from src.pretrained import LABEL as CF_LABEL

    out = []
    for run in list_runs(config.CHECKPOINT_DIR):
        arch = load_checkpoint(config.CHECKPOINT_DIR / f"{run}_best.pt", torch.device("cpu"))[1]["model_name"]
        out.append((f"[ours] {run_label(run, arch)}", lambda run=run: ProjectRun(run, device)))
    out.append((f"[external] {CF_LABEL.split(':', 1)[1].split(' - ')[0].strip()}", lambda: CommunityForensics(device)))
    for model_id, rev, name in HF_DETECTORS:
        out.append((f"[external] {name}", lambda m=model_id, r=rev: HFDetectorBatch(m, r, device)))
    return out


# ---------------------------------------------------------------------------
# 3. Evaluate
# ---------------------------------------------------------------------------
def folder_eval(folder: Path, label: str, name: str, sample: int, batch_size: int, device) -> None:
    """Share of images each detector classifies as `label`, for a folder of images
    that are ALL real or ALL fake (e.g. a dataset of generated faces)."""
    import random

    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if sample and len(files) > sample:
        files = sorted(random.Random(config.SEED).sample(files, sample))
    sizes = {Image.open(files[0]).size}
    print(f"{len(files)} images from {folder} (all {label}); first image size {sizes.pop()}")
    rows = []
    for det_name, make in detectors(device):
        det = make()
        probs = np.concatenate([det.p_fake([ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                                            for f in files[i:i + batch_size]])
                                for i in tqdm(range(0, len(files), batch_size), desc=det_name[:40], leave=False)])
        correct = (probs >= 0.5) if label == "fake" else (probs < 0.5)
        rows.append((det_name, float(correct.mean()), float(np.median(probs))))
        print(f"  {det_name:70s} {correct.mean():6.1%}", flush=True)
        del det
        torch.cuda.empty_cache()
    what = "Fakes detected" if label == "fake" else "Reals kept as real"
    table = (f"| Detector | {what} | Median P(fake) |\n|---|---|---|\n"
             + "\n".join(f"| {n} | {c:.1%} | {m:.3f} |" for n, c, m in sorted(rows, key=lambda r: -r[1])))
    report = (f"# Folder test: {name}\n\n{len(files)} images from `{folder.relative_to(config.PROJECT_ROOT) if folder.is_relative_to(config.PROJECT_ROOT) else folder}`, "
              f"all {label}.\n\n{table}\n")
    out = config.RESULTS_DIR / f"folder_eval_{name}.md"
    out.write_text(report)
    print("\n" + report + f"\nSaved {out}")


def main() -> None:
    from src.utils import get_device, set_seed

    p = argparse.ArgumentParser(description="Evaluate detectors on in-the-wild modern AI images.")
    p.add_argument("--extra", nargs="*", default=[], help="your own image(s): print each detector's P(fake)")
    p.add_argument("--folder", type=Path, default=None, help="evaluate a folder of images that are all real or all fake")
    p.add_argument("--label", choices=["real", "fake"], default="fake", help="what every image in --folder is")
    p.add_argument("--name", default="folder", help="name for the --folder report file")
    p.add_argument("--sample", type=int, default=0, help="--folder: random sample size (0 = all)")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    set_seed(config.SEED)
    device = get_device()
    if args.folder is not None:
        folder_eval(args.folder.resolve(), args.label, args.name, args.sample, args.batch_size, device)
        return

    if not INDEX.exists():
        build_index()
    items = load_items()
    labels = np.array([1 if r["label"] == "fake" else 0 for r in items])
    subsets = np.array([r["subset"] for r in items])
    print(f"{len(items)} images with faces: {int((labels == 0).sum())} real, "
          f"{int(((labels == 1) & (subsets == 'core')).sum())} modern-generator fakes (core), "
          f"{int(((labels == 1) & (subsets == 'reddit')).sum())} Reddit fakes")

    results, extra_rows = {}, []
    for name, make in detectors(device):
        det = make()
        probs = []
        for i in tqdm(range(0, len(items), args.batch_size), desc=name[:40], leave=False):
            batch = [ImageOps.exif_transpose(Image.open(config.PROJECT_ROOT / r["path"])).convert("RGB")
                     for r in items[i:i + args.batch_size]]
            probs.append(det.p_fake(batch))
        probs = np.concatenate(probs)
        pred = probs >= 0.5
        fake_core = pred[(labels == 1) & (subsets == "core")].mean()
        fake_reddit = pred[(labels == 1) & (subsets == "reddit")].mean()
        real_ok = (~pred[labels == 0]).mean()
        results[name] = {"fakes_caught_core": float(fake_core), "fakes_caught_reddit": float(fake_reddit),
                         "reals_correct": float(real_ok),
                         "balanced_accuracy": float((pred[labels == 1].mean() + real_ok) / 2),
                         "roc_auc": float(roc_auc_score(labels, probs))}
        if args.extra:
            extra_rows.append((name, det.p_fake([ImageOps.exif_transpose(Image.open(x)).convert("RGB") for x in args.extra])))
        print(f"  {name:70s} done", flush=True)
        del det
        torch.cuda.empty_cache()

    lines = ["| Detector | Modern-generator fakes caught | Reddit fakes caught | Reals kept as real | Balanced accuracy | ROC-AUC |",
             "|---|---|---|---|---|---|"]
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["roc_auc"]):
        lines.append(f"| {name} | {r['fakes_caught_core']:.1%} | {r['fakes_caught_reddit']:.1%} | "
                     f"{r['reals_correct']:.1%} | {r['balanced_accuracy']:.1%} | {r['roc_auc']:.3f} |")
    table = "\n".join(lines)
    n_real = int((labels == 0).sum())
    report = (f"# In-the-wild evaluation (OpenFakeTiny, faces only)\n\n{len(items)} images: {n_real} real, "
              f"{int((labels == 1).sum())} fake. Balanced accuracy = average of fake and real accuracy "
              f"(50% = guessing). Only {n_real} real photos, so 'reals kept as real' is approximate.\n\n{table}\n")
    if extra_rows:
        report += "\n## Your own images: P(fake) per detector\n\n| Detector | " + " | ".join(Path(x).name for x in args.extra) + " |\n|---|" + "---|" * len(args.extra) + "\n"
        for name, ps in extra_rows:
            report += f"| {name} | " + " | ".join(f"{p:.3f}" for p in ps) + " |\n"
    (config.RESULTS_DIR / "wild_eval.md").write_text(report)
    (config.RESULTS_DIR / "wild_eval.json").write_text(json.dumps(results, indent=2))
    print("\n" + report + f"\nSaved {config.RESULTS_DIR / 'wild_eval.md'}")


if __name__ == "__main__":
    main()
