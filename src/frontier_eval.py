"""
frontier_eval.py - Test detectors on the newest generators, incl. social-media copies.

    python -m src.frontier_eval

Dataset: WasItAIGenerated Frontier Image Benchmark, public split (October 2026)
https://huggingface.co/datasets/wasitaigeneratedcom/frontier-image-video-bench-2026-10
  - 261 AI images from 19 current generators (GPT-Image-2, Nano Banana Pro /
    2.1, FLUX.3, FLUX.2, Seedream 5 Pro, Qwen-Image 3, Ideogram 4.5, ...)
  - 486 real photos (CC0 / public domain / CC BY / CC BY-SA)
  - every image ALSO as a WhatsApp, Instagram and re-shared copy
  - metadata (EXIF, C2PA) stripped, so detectors must use the pixels.
  Caveat: built by a company that sells a detector; not face-specific.

Reported per detector and per variant (original / whatsapp / instagram / reshared):
  - ROC-AUC (threshold-free separation of fake vs real),
  - "detected @ 1% FP": share of fakes caught when the threshold is set so
    that only 1% of real photos are flagged (the benchmark's headline metric),
  - fakes caught / reals kept at the normal 0.5 threshold.
The external detectors see every image; our own models only images with a
detectable face (they were trained on faces only).
"""

import csv
import json
import socket
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src import config

REPO = "wasitaigeneratedcom/frontier-image-video-bench-2026-10"
ROOT = config.RAW_DIR / "frontier_bench"
OUT_DIR = config.RESULTS_DIR / "frontier"
VARIANTS = ["original", "whatsapp", "instagram", "reshared"]
SKIP_DETECTORS = ("prithivMLmods",)  # flags every image as fake (see wild_eval.md)


def download() -> None:
    """Images + metadata only (no videos). IPv4 is preferred because the Hugging
    Face CDN hangs over IPv6 on some networks."""
    meta = ROOT / "metadata.csv"
    if meta.exists():  # skip only if EVERY listed image file is already here
        with open(meta, newline="", encoding="utf-8") as f:
            wanted = [r["file"] for r in csv.DictReader(f) if r["media"] == "image"]
        if all((ROOT / p).exists() for p in wanted):
            return
    from huggingface_hub import snapshot_download

    original = socket.getaddrinfo
    socket.getaddrinfo = lambda *a, **k: ([r for r in original(*a, **k) if r[0] == socket.AF_INET]
                                          or original(*a, **k))
    try:
        snapshot_download(REPO, repo_type="dataset", local_dir=ROOT,
                          allow_patterns=["metadata.csv", "LICENSE*", "README.md",
                                          "images/*", "images_social/*/*"])
    finally:
        socket.getaddrinfo = original


def load_image(path: Path, max_side: int = 2048) -> Image.Image:
    """Open an image, shrinking very large ones (some real photos are ~200 megapixels).

    The detectors resize everything to 224-440 px anyway, so capping at 2048 px
    changes nothing they use, but avoids decoding 600 MB of pixels per file.
    JPEG 'draft' mode decodes directly at a reduced size, which is fast.
    """
    Image.MAX_IMAGE_PIXELS = 250_000_000  # allow the largest real photos, still refuse absurd files
    img = Image.open(path)
    if img.format == "JPEG":
        img.draft("RGB", (max_side, max_side))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def load_items() -> list[dict]:
    with open(ROOT / "metadata.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["media"] == "image" and (ROOT / r["file"]).exists()]
    return [{"path": ROOT / r["file"], "id": r["id"], "label": int(r["label"] == "ai"),
             "variant": r["variant"], "generator": r["generator"] or "real"} for r in rows]


def face_ids(items: list[dict]) -> set[str]:
    """Ids of images that contain a clear face (>= 96 px), checked on the originals."""
    cache = OUT_DIR / "face_ids.json"
    if cache.exists():
        return set(json.loads(cache.read_text()))
    from src.faces import FaceAligner

    aligner, ids = FaceAligner(), []
    for it in tqdm([i for i in items if i["variant"] == "original"], desc="finding faces", leave=False):
        img = load_image(it["path"])
        found = aligner.detect(np.asarray(img))
        if found is not None:
            quad = aligner.ffhq_quad(found[0])
            if np.hypot(*(quad[3] - quad[0])) >= 96:
                ids.append(it["id"])
    cache.write_text(json.dumps(ids))
    return set(ids)


def metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    real, fake = probs[labels == 0], probs[labels == 1]
    if len(real) == 0 or len(fake) == 0:
        return {}
    threshold_1fp = np.quantile(real, 0.99)  # only 1% of real photos score above this
    return {"n_fake": int(len(fake)), "n_real": int(len(real)),
            "roc_auc": float(roc_auc_score(labels, probs)),
            "detected_at_1pct_fp": float((fake > threshold_1fp).mean()),
            "fakes_caught": float((fake >= 0.5).mean()),
            "reals_kept": float((real < 0.5).mean())}


def main() -> None:
    from src.utils import get_device, set_seed
    from src.wild_eval import detectors

    set_seed(config.SEED)
    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    download()
    items = load_items()
    faces = face_ids(items)
    print(f"{len(items)} image files ({sum(i['variant'] == 'original' for i in items)} originals); "
          f"{len(faces)} originals contain a clear face")

    results, per_generator = {}, {}
    for name, make in detectors(device):
        if any(s in name for s in SKIP_DETECTORS):
            continue
        ours = name.startswith("[ours]")
        subset = [i for i in items if i["id"] in faces] if ours else items
        scores_file = OUT_DIR / f"scores_{''.join(c if c.isalnum() else '_' for c in name)[:60]}.csv"
        if scores_file.exists():  # resume: reuse scores computed earlier
            with open(scores_file) as f:
                cached = {r["file"]: float(r["p_fake"]) for r in csv.DictReader(f)}
            probs = np.array([cached[str(i["path"])] for i in subset])
        else:
            det = make()
            probs = []
            for it in tqdm(subset, desc=name[:40], leave=False):
                probs.append(float(det.p_fake([load_image(it["path"])])[0]))
            probs = np.array(probs)
            with open(scores_file, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["file", "p_fake"])
                w.writerows((str(i["path"]), p) for i, p in zip(subset, probs))
            del det
            torch.cuda.empty_cache()
        labels = np.array([i["label"] for i in subset])
        variants = np.array([i["variant"] for i in subset])
        has_face = np.array([i["id"] in faces for i in subset])
        results[name] = {}
        for v in VARIANTS:
            m = variants == v
            results[name][f"{v} | {'faces' if ours else 'all'}"] = metrics(labels[m], probs[m])
            if not ours:
                results[name][f"{v} | faces"] = metrics(labels[m & has_face], probs[m & has_face])
        if not ours:  # fakes caught per generator (originals, threshold 0.5)
            gens = defaultdict(list)
            for it, p in zip(subset, probs):
                if it["variant"] == "original" and it["label"] == 1:
                    gens[it["generator"]].append(p >= 0.5)
            per_generator[name] = {g: float(np.mean(v)) for g, v in gens.items()}
        o = results[name][f"original | {'faces' if ours else 'all'}"]
        print(f"  {name:65s} originals: AUC {o['roc_auc']:.3f}, caught {o['fakes_caught']:.0%}, "
              f"reals kept {o['reals_kept']:.0%}", flush=True)

    # ---- Report --------------------------------------------------------------
    lines = ["# Frontier benchmark (19 current generators, public split)", "",
             "Fakes from GPT-Image-2, Nano Banana Pro/2.1, FLUX.3, FLUX.2, Seedream 5, Qwen-Image 3, "
             "Ideogram 4.5, Grok Imagine 2 and more, vs real photos; each also as WhatsApp / Instagram / "
             "re-shared copies. 'Detected @ 1% FP' = share of fakes caught when only 1% of real photos "
             "may be flagged. Our models are scored on face images only.", ""]
    for scope in ("all", "faces"):
        lines += [f"## {'All images' if scope == 'all' else 'Only images with a face'}", "",
                  "| Detector | Variant | ROC-AUC | Detected @ 1% FP | Fakes caught (0.5) | Reals kept (0.5) | n fake / real |",
                  "|---|---|---|---|---|---|---|"]
        for name, res in results.items():
            for v in VARIANTS:
                m = res.get(f"{v} | {scope}")
                if m:
                    lines.append(f"| {name} | {v} | {m['roc_auc']:.3f} | {m['detected_at_1pct_fp']:.1%} | "
                                 f"{m['fakes_caught']:.1%} | {m['reals_kept']:.1%} | {m['n_fake']} / {m['n_real']} |")
        lines.append("")
    gens = sorted({g for v in per_generator.values() for g in v})
    lines += ["## Fakes caught per generator (originals, threshold 0.5)", "",
              "| Generator | " + " | ".join(per_generator) + " |", "|---|" + "---|" * len(per_generator)]
    for g in gens:
        lines.append(f"| {g} | " + " | ".join(f"{per_generator[n].get(g, float('nan')):.0%}" for n in per_generator) + " |")
    report = "\n".join(lines) + "\n"
    (config.RESULTS_DIR / "frontier_eval.md").write_text(report)
    (OUT_DIR / "results.json").write_text(json.dumps({"results": results, "per_generator": per_generator}, indent=2))
    print("\n" + report + f"\nSaved {config.RESULTS_DIR / 'frontier_eval.md'}")


if __name__ == "__main__":
    main()
