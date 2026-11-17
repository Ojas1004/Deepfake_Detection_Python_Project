"""
prepare_data.py - Build the multi-source, face-aligned dataset (run once).

    python -m src.prepare_data            # ~4-5 GB of downloads, then processing
    python -m src.prepare_data --force    # rebuild data/processed from cached downloads
    python -m src.prepare_data --add-openfake   # add source 5 to an existing data/processed

Sources (see config.py and the README for why each was chosen):
  1. 140k Real and Fake Faces (Kaggle): real FFHQ photos + StyleGAN (2019) fakes
  2. SFHQ-T2I (Hugging Face):  FLUX.1 schnell/dev/pro, SDXL, DALL-E 3 fakes
  3. DeepFakeFace (Hugging Face): real IMDB-WIKI photos + Stable Diffusion 1.5,
     SD inpainting and InsightFace face-swap fakes made from them
  4. StyleGAN3 fakes (Kaggle): test only
  5. OpenFake (Hugging Face): fakes from ~70 generators released 2022-2025 and
     real LAION/Pexels photos (src/openfake.py; ~39 GB of parquet shards)

Steps:
  1. PLAN: decide which images go into which split, with a fixed seed, before
     touching any pixels. Extra candidates are planned because some images
     will be rejected (no face found / face too small).
  2. DOWNLOAD only the planned images (cached in data/raw/, re-runs skip them).
     The two Hugging Face datasets are huge zips; src/remote_zip.py fetches
     single files out of them without downloading the whole archive.
  3. STANDARDISE every image identically (src/faces.py):
       detect face -> FFHQ-style alignment -> 256x256 -> JPEG q95 -> 224x224 PNG
     so real and fake images share one processing history and the model can't
     tell them apart by size, framing or compression.
  4. KEEP the first N successful images per (split, group) and write
     data/processed/manifest.csv.

Data-leakage rules:
  - 140k: our splits come from the dataset's own train/valid/test folders.
  - SFHQ-T2I: each image is used in at most one split.
  - DeepFakeFace: every fake is made FROM one real photo, so photos (=people;
    the dataset has one photo per person) are split first, and a photo's real
    image and its fakes always land in the same split.
  - DALL-E 3 and StyleGAN3 are never used for training or validation.
  - OpenFake: the split comes from a hash of the image content, so an image is
    in exactly one split; only its "train" shards are used, never the test
    splits that src/wild_eval.py evaluates on.

Two DeepFakeFace quirks are corrected (found by inspecting the data):
  - Inpainting and InsightFace fakes were stretched from the original photo's
    shape to 512x512 (faces squashed/stretched by ~1.3x). We resize them back
    to the original photo's size first, otherwise "distorted face" = fake.
  - Many real photos are wide shots with small faces. Faces under
    MIN_FACE_CROP pixels are skipped in ALL sources, so blurry upscaled faces
    can't become a "real" giveaway.
"""

import argparse
import csv
import math
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from src import config

# Group -> class name. Real groups are label 0, fake groups label 1.
GROUP_CLASS = {g: "real" for g in config.REAL_GROUPS} | {
    g: "fake" for g in config.FAKE_GROUPS + ["dalle3", "stylegan3"]}

# Plan extra candidates per group to make up for rejected images
# (rates measured on samples: e.g. ~40% of IMDB-WIKI faces are too small).
OVERSAMPLE = {"ffhq": 1.05, "stylegan1": 1.05, "flux_schnell": 1.2, "sdxl": 1.2,
              "flux_dev": 1.2, "flux_pro": 1.2, "dalle3": 1.0, "stylegan3": 1.1,
              "dff_real": 1.9, "sd15_t2i": 1.9, "sd_inpaint": 1.9, "insightface": 1.9,
              "of_real": 1.0, "of_fake": 1.0}  # OpenFake faces are checked before planning

SFHQ_PREFIX_TO_GROUP = {"FLUX1_schnell": "flux_schnell", "FLUX1_dev": "flux_dev",
                        "FLUX1_pro": "flux_pro", "SDXL": "sdxl", "DALLE3": "dalle3"}
DFF_FAKE_PARTS = {"sd15_t2i": "text2img", "sd_inpaint": "inpainting", "insightface": "insight"}
DFF_SPLIT_FRACTIONS = {"train": 0.75, "val": 0.125, "test": 0.125}
SPLIT_ORDER = ["train", "val", "test", "unseen_dalle3", "unseen_stylegan3"]


@dataclass
class Candidate:
    split: str
    group: str
    src: Path                       # downloaded original image
    original: str                   # where it came from (for the manifest)
    unstretch_to: Path | None = None  # DFF edits: resize to this image's size first
    # OpenFake: src is ALREADY the standardised 224 px face (made by src/openfake.py
    # with the same standardize_face); it is copied as it is. Holds its crop size.
    standardized_crop: float | None = None


def n_candidates(split: str, group: str) -> int:
    return math.ceil(config.SPLIT_PLAN[split].get(group, 0) * OVERSAMPLE[group])


# ---------------------------------------------------------------------------
# Source 1: 140k Real and Fake Faces (Kaggle)
# ---------------------------------------------------------------------------
def find_dataset_root(raw_dir: Path) -> Path | None:
    """Locate the folder that contains train/, valid/ and test/ of the 140k dataset."""
    for candidate in raw_dir.rglob("train"):
        root = candidate.parent
        if all((root / s / c).is_dir() for s in ("train", "valid", "test") for c in ("real", "fake")):
            return root
    return None


def download_140k(raw_dir: Path) -> Path:
    """Download and unzip the 140k dataset with the Kaggle API, unless already present."""
    root = find_dataset_root(raw_dir) if raw_dir.exists() else None
    if root is not None:
        print(f"[140k] already downloaded at {root}")
        return root
    from kaggle.api.kaggle_api_extended import KaggleApi  # needs ~/.kaggle credentials

    raw_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"[140k] downloading {config.KAGGLE_DATASET} (~4 GB) ...")
    api.dataset_download_files(config.KAGGLE_DATASET, path=str(raw_dir), unzip=True, quiet=False)
    root = find_dataset_root(raw_dir)
    if root is None:
        raise RuntimeError(f"No train/valid/test folders found in {raw_dir}")
    return root


def plan_140k(rng: random.Random) -> list[Candidate]:
    root = download_140k(config.RAW_DIR)
    kaggle_split = {"train": "train", "val": "valid", "test": "test"}
    out = []
    for split, folder in kaggle_split.items():
        for group, cls in (("ffhq", "real"), ("stylegan1", "fake")):
            # Sort first: directory order differs between filesystems.
            files = sorted((root / folder / cls).glob("*.jpg"))
            for f in rng.sample(files, n_candidates(split, group)):
                out.append(Candidate(split, group, f, f"140k/{f.relative_to(root).as_posix()}"))
    return out


# ---------------------------------------------------------------------------
# Source 2: SFHQ-T2I (Hugging Face, remote zip)
# ---------------------------------------------------------------------------
def plan_sfhq(rng: random.Random, download_workers: int) -> list[Candidate]:
    from src.remote_zip import RemoteZip

    print("[SFHQ-T2I] reading the zip's table of contents ...")
    archive = RemoteZip(config.SFHQ_T2I_ZIP)
    by_group = defaultdict(list)
    for name in sorted(archive.members):
        prefix = re.sub(r"_image_\d+\.\w+$", "", Path(name).name)
        by_group[SFHQ_PREFIX_TO_GROUP[prefix]].append(name)

    out, wanted = [], []
    for group, names in by_group.items():
        rng.shuffle(names)
        start = 0
        for split in SPLIT_ORDER:  # consecutive slices -> each image in at most one split
            n = min(n_candidates(split, group), len(names) - start)
            for name in names[start:start + n]:
                out.append(Candidate(split, group, config.RAW_DIR / "sfhq_t2i" / name,
                                     f"sfhq_t2i/{name}"))
                wanted.append(name)
            start += n
    print(f"[SFHQ-T2I] downloading {len(wanted)} of {len(archive.members)} images ...")
    archive.download(wanted, config.RAW_DIR / "sfhq_t2i", workers=download_workers)
    return out


# ---------------------------------------------------------------------------
# Source 3: DeepFakeFace (Hugging Face, remote zips)
# ---------------------------------------------------------------------------
def plan_dff(rng: random.Random, download_workers: int) -> list[Candidate]:
    from src.remote_zip import RemoteZip

    archives = {part: RemoteZip(config.DFF_ZIP.format(part=part))
                for part in ["wiki", *DFF_FAKE_PARTS.values()]}
    photos = sorted(Path(n).relative_to("wiki").as_posix() for n in archives["wiki"].members)
    rng.shuffle(photos)

    # Split PHOTOS first (one photo per person), so a real photo and the
    # fakes made from it are always in the same split.
    out, start = [], 0
    fake_groups = list(DFF_FAKE_PARTS)
    downloads = defaultdict(list)
    for split, fraction in DFF_SPLIT_FRACTIONS.items():
        pool = photos[start:start + round(fraction * len(photos))]
        start += len(pool)
        n_fake_each = max(n_candidates(split, g) for g in fake_groups)
        k = min(len(pool), max(n_candidates(split, "dff_real"), 3 * n_fake_each))
        for j, rel in enumerate(pool[:k]):
            real_path = config.RAW_DIR / "dff" / "wiki" / rel
            out.append(Candidate(split, "dff_real", real_path, f"dff/wiki/{rel}"))
            downloads["wiki"].append(f"wiki/{rel}")
            # Each photo also contributes ONE fake, rotating through the 3 types.
            group = fake_groups[j % 3]
            part = DFF_FAKE_PARTS[group]
            out.append(Candidate(split, group, config.RAW_DIR / "dff" / part / rel, f"dff/{part}/{rel}",
                                 unstretch_to=real_path if group != "sd15_t2i" else None))
            downloads[part].append(f"{part}/{rel}")

    for part, names in downloads.items():
        print(f"[DeepFakeFace] downloading {len(names)} images from {part}.zip ...")
        archives[part].download(names, config.RAW_DIR / "dff", workers=download_workers)
    return out


# ---------------------------------------------------------------------------
# Source 4: StyleGAN3 fakes (Kaggle, test only)
# ---------------------------------------------------------------------------
def plan_stylegan3(rng: random.Random) -> list[Candidate]:
    folder = config.RAW_DIR / "stylegan3"
    fakes_dir = folder / "Fake faces"
    if not fakes_dir.is_dir():
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print(f"[StyleGAN3] downloading {config.STYLEGAN3_KAGGLE} (~2 GB zip) ...")
        api.dataset_download_files(config.STYLEGAN3_KAGGLE, path=str(folder), unzip=False, quiet=False)
        archive_path = next(folder.glob("*.zip"))
        with zipfile.ZipFile(archive_path) as z:  # we only need the fake half
            members = [m for m in z.namelist() if m.startswith("Fake faces/") and m.endswith(".png")]
            z.extractall(folder, members=members)
        archive_path.unlink()
    files = sorted(fakes_dir.glob("*.png"))
    return [Candidate("unseen_stylegan3", "stylegan3", f, f"stylegan3/{f.name}")
            for f in rng.sample(files, n_candidates("unseen_stylegan3", "stylegan3"))]


# ---------------------------------------------------------------------------
# Source 5: OpenFake (Hugging Face, parquet shards)
# ---------------------------------------------------------------------------
def plan_openfake(workers: int) -> list[Candidate]:
    """Modern-generator fakes + LAION/Pexels real photos (see src/openfake.py).

    Uses its own seeded selection, not the shared rng, so adding this source
    leaves the images chosen from sources 1-4 unchanged.
    """
    from src import openfake

    chosen = openfake.plan(workers)
    print(f"[OpenFake] {len(chosen)} faces selected")
    return [Candidate(r["split"], r["group"], openfake.ALIGNED_DIR / r["file"],
                      f"openfake/{r['model']}/{r['row_id']}", standardized_crop=float(r["crop_size"]))
            for r in chosen]


# ---------------------------------------------------------------------------
# Processing (runs in worker processes)
# ---------------------------------------------------------------------------
_ALIGNER = None


def _init_worker():
    import cv2

    from src.faces import FaceAligner

    global _ALIGNER
    cv2.setNumThreads(1)  # we already run one process per core
    _ALIGNER = FaceAligner()


def process(job: tuple[Candidate, Path]) -> tuple[str, float | None]:
    """Standardise one image and save it; return (status, face crop size)."""
    from src.faces import standardize_face

    cand, dst = job
    if cand.standardized_crop is not None:  # already standardised: copy unchanged
        shutil.copyfile(cand.src, dst)
        return "ok", cand.standardized_crop
    try:
        with Image.open(cand.src) as img:
            img = img.convert("RGB")
            if cand.unstretch_to is not None:  # undo DeepFakeFace's 512x512 stretch
                with Image.open(cand.unstretch_to) as ref:
                    img = img.resize(ref.size, Image.Resampling.LANCZOS)
            face, aligned, status = standardize_face(img, _ALIGNER, min_crop=config.MIN_FACE_CROP)
    except OSError:
        return "unreadable", None
    if face is not None:
        face.save(dst, format="PNG")
    return status, (round(aligned.crop_size, 1) if aligned else None)


def build(candidates: list[Candidate], workers: int, first_index: int = 0) -> list[dict]:
    """Standardise the candidates, keep the first N successes per (split, group)
    and return their manifest rows. Prints a planned/kept table."""
    # ---- 3. Standardise everything in parallel
    jobs = []
    for i, c in enumerate(candidates, first_index):
        cls = GROUP_CLASS[c.group]
        out_dir = config.PROCESSED_DIR / c.split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(c.original).stem)
        jobs.append((c, out_dir / f"{c.group}_{stem}_{i}.png"))
    print(f"[process] aligning and standardising {len(jobs)} candidate images "
          f"with {workers} processes ...")
    with Pool(workers, initializer=_init_worker) as pool:
        results = []
        for i, r in enumerate(pool.imap(process, jobs, chunksize=32), 1):  # imap keeps order
            results.append(r)
            if i % 5000 == 0 or i == len(jobs):
                print(f"[process]   {i}/{len(jobs)}", flush=True)

    # ---- 4. Keep the first N successes per (split, group), in planned order
    kept, stats, rows = Counter(), defaultdict(Counter), []
    for (cand, dst), (status, crop) in zip(jobs, results):
        key = (cand.split, cand.group)
        stats[key][status] += 1
        if status != "ok":
            continue
        if kept[key] >= config.SPLIT_PLAN[cand.split][cand.group]:
            dst.unlink()  # surplus candidate: not needed
            continue
        kept[key] += 1
        cls = GROUP_CLASS[cand.group]
        rows.append({"split": cand.split, "path": dst.relative_to(config.PROCESSED_DIR).as_posix(),
                     "label": config.CLASS_TO_IDX[cls], "class": cls, "group": cand.group,
                     "original": cand.original, "crop_size": crop})

    planned_groups = {c.group for c in candidates}
    print(f"\n{'split':17s} {'group':13s} {'planned':>8s} {'kept':>6s}  rejected")
    for split in SPLIT_ORDER:
        for group, wanted in config.SPLIT_PLAN[split].items():
            if group not in planned_groups:
                continue
            s = stats[(split, group)]
            rejected = ", ".join(f"{k} {v}" for k, v in s.items() if k != "ok") or "-"
            flag = "" if kept[(split, group)] == wanted else f"   <-- SHORT by {wanted - kept[(split, group)]}"
            print(f"{split:17s} {group:13s} {wanted:8d} {kept[(split, group)]:6d}  {rejected}{flag}")
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[done] {path}: {len(rows)} images")


def add_openfake(manifest_path: Path, workers: int) -> None:
    """Add source 5 to an existing dataset without touching the images already in it.

    Gives the same result as a full --force rebuild (OpenFake's selection does
    not depend on the other sources), but needs no re-download of sources 1-4.
    """
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found - run `python -m src.prepare_data` first.")
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if any(r["group"] in ("of_real", "of_fake") for r in rows):
        print("OpenFake is already in the dataset - nothing to do "
              "(use --force to rebuild everything).")
        return
    backup = manifest_path.with_name("manifest_before_openfake.csv")
    shutil.copyfile(manifest_path, backup)
    print(f"[add] existing manifest ({len(rows)} images) backed up to {backup}")
    new_rows = build(plan_openfake(workers), workers, first_index=len(rows))
    write_manifest(manifest_path, rows + new_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the aligned multi-source dataset.")
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS, help="processing processes")
    parser.add_argument("--download-workers", type=int, default=32, help="parallel downloads")
    parser.add_argument("--force", action="store_true", help="rebuild data/processed")
    parser.add_argument("--add-openfake", action="store_true",
                        help="add the OpenFake source to an existing data/processed")
    args = parser.parse_args()

    manifest_path = config.PROCESSED_DIR / "manifest.csv"
    if args.add_openfake:
        add_openfake(manifest_path, args.workers)
        return
    if manifest_path.exists() and not args.force:
        print(f"{manifest_path} already exists - nothing to do (use --force to rebuild).")
        return
    if config.PROCESSED_DIR.exists():
        shutil.rmtree(config.PROCESSED_DIR)

    # ---- 1+2. Plan and download (one RNG, fixed order -> same dataset every run)
    rng = random.Random(config.SEED)
    candidates = (plan_140k(rng) + plan_sfhq(rng, args.download_workers)
                  + plan_dff(rng, args.download_workers) + plan_stylegan3(rng)
                  + plan_openfake(args.workers))
    write_manifest(manifest_path, build(candidates, args.workers))


if __name__ == "__main__":
    main()
