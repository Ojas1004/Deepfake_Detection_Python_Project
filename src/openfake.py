"""
openfake.py - OpenFake as a data source: modern generators + varied real photos.

    python -m src.openfake        # download missing shards, extract aligned faces, print counts

OpenFake (ComplexDataLab, CC BY-NC 4.0) stores its images inside large parquet
files ("shards", ~4,000 images / ~5.6 GB each), half real photos (LAION,
Pexels) and half fakes from ~70 generators released 2022-2025. Most images
show no usable face, so this module works in two steps:

  1. EXTRACT (slow, done once, cached): read every image of every shard,
     standardise it exactly like all other sources (faces.standardize_face:
     detect face -> align -> 256 px -> JPEG q95 -> 224 px, faces under
     MIN_FACE_CROP skipped) and save the result to data/raw/openfake/aligned/.
     One index CSV per shard records label, generator and content hash.
  2. PLAN (fast): choose which of those faces go into which split.
       - Split = a hash of the image content, so an image (and any exact
         duplicate) can only ever be in one split.
       - Real photos: a seeded random sample.
       - Fakes: taken ROUND-ROBIN across generators. A few generators (SD 3.5,
         SD 2.1, FLUX.1 dev) make up a third of the fakes; round-robin uses
         every face of the rare, newer ones before taking more of the common ones.

prepare_data.py calls plan() and copies the chosen faces into data/processed.
"""

import csv
import hashlib
import io
import random
import re
import urllib.request
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from src import config

ALIGNED_DIR = config.OPENFAKE_DIR / "aligned"
INDEX_FIELDS = ["row_id", "sha256", "label", "model", "crop_size", "file"]
# Extraction decodes full-size photos (some are 20+ megapixels): measured peak
# 1.3 GB of RAM per process, so 3 processes need about 4 GB.
MAX_EXTRACT_WORKERS = 3


def shard_path(shard: int) -> Path:
    return config.OPENFAKE_DIR / Path(config.OPENFAKE_URL.format(shard=shard)).name


def download() -> None:
    """Fetch the shards that are not on disk yet (resumable with curl, see README)."""
    from src.remote_zip import USER_AGENT

    config.OPENFAKE_DIR.mkdir(parents=True, exist_ok=True)
    for shard in range(config.OPENFAKE_SHARDS):
        path = shard_path(shard)
        if path.exists():
            continue
        print(f"[OpenFake] downloading shard {shard} (~5.6 GB) ...", flush=True)
        tmp = path.with_suffix(".part")
        req = urllib.request.Request(config.OPENFAKE_URL.format(shard=shard), headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)


def _extract_shard(shard: int) -> Counter:
    """Standardise every face of one shard; write aligned PNGs + the shard's index CSV."""
    import cv2
    import pyarrow.parquet as pq

    from src.faces import FaceAligner, standardize_face

    index_path = ALIGNED_DIR / f"index_{shard:05d}.csv"
    if index_path.exists():  # this shard was already extracted
        return Counter()
    cv2.setNumThreads(1)
    Image.MAX_IMAGE_PIXELS = 250_000_000
    aligner, rows, stats = FaceAligner(), [], Counter()
    out_dir = ALIGNED_DIR / f"{shard:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["image", "label", "model", "row_id", "image_sha256"]
    # One row group (~70 images, ~100 MB) at a time, each with a fresh reader.
    # ParquetFile.iter_batches() must NOT be used here: its reader keeps every
    # row group it has read, so memory grows to the size of the whole shard
    # (5.6 GB per process). This way it stays flat at about 0.7-1.3 GB.
    n_groups = pq.ParquetFile(shard_path(shard)).metadata.num_row_groups
    for group in range(n_groups):
        reader = pq.ParquetFile(shard_path(shard), pre_buffer=False)
        batch = reader.read_row_group(group, columns=columns, use_threads=False).to_pylist()
        del reader
        for r in batch:
            try:
                img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                face, aligned, status = standardize_face(img, aligner, min_crop=config.MIN_FACE_CROP)
            except (OSError, ValueError, Image.DecompressionBombError):
                face, aligned, status = None, None, "unreadable"
            stats[(r["label"], status)] += 1
            if face is None:
                continue
            name = f"{shard:05d}/{r['row_id']}.png"
            face.save(ALIGNED_DIR / name, format="PNG")
            rows.append({"row_id": r["row_id"], "sha256": r["image_sha256"], "label": r["label"],
                         "model": r["model"], "crop_size": round(aligned.crop_size, 1), "file": name})
    tmp = index_path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.rename(index_path)  # written last: an interrupted shard is simply redone
    return stats


def extract(workers: int = config.NUM_WORKERS) -> None:
    """Step 1: aligned faces of all shards (skips shards already done)."""
    todo = [s for s in range(config.OPENFAKE_SHARDS) if not (ALIGNED_DIR / f"index_{s:05d}.csv").exists()]
    if not todo:
        return
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    workers = min(workers, MAX_EXTRACT_WORKERS)
    print(f"[OpenFake] extracting aligned faces from {len(todo)} shard(s) with "
          f"{min(workers, len(todo))} processes (several minutes per shard) ...", flush=True)
    total = Counter()
    with Pool(min(workers, len(todo))) as pool:
        for i, stats in enumerate(pool.imap_unordered(_extract_shard, todo), 1):
            total += stats
            print(f"[OpenFake]   {i}/{len(todo)} shards done", flush=True)
    for label in ("real", "fake"):
        print(f"[OpenFake]   {label}: " + ", ".join(f"{status} {n}" for (lab, status), n in sorted(total.items())
                                                    if lab == label))


def split_of(sha256: str) -> str:
    """Deterministic split from the image's content hash (uniform in [0, 1))."""
    u = int(hashlib.sha256(sha256.encode()).hexdigest()[:8], 16) / 0x100000000
    edge = 0.0
    for split, fraction in config.OPENFAKE_SPLIT_FRACTIONS.items():
        edge += fraction
        if u < edge:
            return split
    return split


def load_faces() -> list[dict]:
    """All extracted faces: photo-like generators only, exact duplicates removed."""
    exclude, seen, faces = re.compile(config.OPENFAKE_EXCLUDE), set(), []
    for shard in range(config.OPENFAKE_SHARDS):
        with open(ALIGNED_DIR / f"index_{shard:05d}.csv", newline="") as f:
            for r in csv.DictReader(f):
                if r["sha256"] in seen or (r["label"] == "fake" and exclude.search(r["model"])):
                    continue
                seen.add(r["sha256"])
                faces.append(r | {"split": split_of(r["sha256"]),
                                  "group": "of_real" if r["label"] == "real" else "of_fake"})
    return sorted(faces, key=lambda r: r["row_id"])  # fixed order -> same selection every run


def select(faces: list[dict], seed: int = config.SEED) -> list[dict]:
    """Step 2: choose config.SPLIT_PLAN[split][group] faces per split and group."""
    chosen = []
    for split in config.OPENFAKE_SPLIT_FRACTIONS:
        for group in ("of_real", "of_fake"):
            want = config.SPLIT_PLAN[split].get(group, 0)
            pool = [r for r in faces if r["split"] == split and r["group"] == group]
            rng = random.Random(f"{seed}-{split}-{group}")
            rng.shuffle(pool)
            if group == "of_fake":  # round-robin across generators
                by_model = defaultdict(list)
                for r in pool:
                    by_model[r["model"]].append(r)
                pool = []
                while by_model:
                    for model in sorted(by_model):
                        pool.append(by_model[model].pop())
                    by_model = {m: v for m, v in by_model.items() if v}
            chosen += pool[:want]
    return chosen


def plan(workers: int = config.NUM_WORKERS) -> list[dict]:
    """Download + extract if needed, then return the chosen faces (dicts with
    split, group, file, model, row_id, crop_size)."""
    download()
    extract(workers)
    return select(load_faces())


if __name__ == "__main__":
    download()
    extract()
    faces = load_faces()
    print(f"\n{len(faces)} usable faces (>= {config.MIN_FACE_CROP} px, photo-like generators, no duplicates)")
    for split in config.OPENFAKE_SPLIT_FRACTIONS:
        n = Counter(r["group"] for r in faces if r["split"] == split)
        print(f"  {split:6s} available: {n['of_real']:5d} real, {n['of_fake']:5d} fake   "
              f"planned: {config.SPLIT_PLAN[split].get('of_real', 0)} real, {config.SPLIT_PLAN[split].get('of_fake', 0)} fake")
    chosen = select(faces)
    models = Counter(r["model"] for r in chosen if r["group"] == "of_fake" and r["split"] == "train")
    print(f"\ntrain fakes by generator ({len(models)} generators): "
          + ", ".join(f"{m} {n}" for m, n in models.most_common()))
