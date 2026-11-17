"""
data.py - Dataset, transforms and DataLoaders.

Reads the aligned, pre-resized 224x224 images created by src/prepare_data.py,
using data/processed/manifest.csv as the single source of truth for which
images belong to which split, their labels and which source/generator
("group") they came from.

Labels: 0 = real, 1 = fake (see config.CLASS_TO_IDX).

Run directly for a sanity check of the pipeline:
    python -m src.data
"""

import csv
import io
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src import config
from src.utils import seed_worker


def to_model_size(image: Image.Image) -> Image.Image:
    """FALLBACK preprocessing when no face can be detected (see faces.py for
    the normal path): EXIF rotation, RGB, centre square crop, LANCZOS resize
    to IMG_SIZE x IMG_SIZE.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.LANCZOS)


class FaceDataset(Dataset):
    """Real-vs-fake face images for one split ("train", "val" or "test")."""

    def __init__(self, split: str, transform=None, root: Path | None = None):
        self.root = Path(root or config.PROCESSED_DIR)
        self.transform = transform

        manifest = self.root / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"{manifest} not found - run `python -m src.prepare_data` first.")

        # Keep only this split's rows: (relative_path, label) pairs, plus each
        # image's group (source/generator) for per-generator evaluation.
        with open(manifest, newline="") as f:
            rows = [row for row in csv.DictReader(f) if row["split"] == split]
        self.samples = [(row["path"], int(row["label"])) for row in rows]
        self.groups = [row.get("group", "") for row in rows]
        if not self.samples:
            raise ValueError(f"No images found for split '{split}' in {manifest}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path, label = self.samples[idx]
        # The images are already 224x224, so no resizing happens here.
        image = Image.open(self.root / rel_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class RandomJPEG:
    """Re-compress the image as JPEG with a random quality.

    Images shared online (WhatsApp, Instagram, websites) are almost always
    re-compressed. Training on random JPEG qualities stops the model from
    treating "JPEG artifacts" as a sign of fakeness (the v1 model did).
    """

    def __init__(self, p: float = 0.5, quality: tuple[int, int] = (30, 95)):
        self.p, self.quality = p, quality

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(*self.quality))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomDownscale:
    """Shrink the image and enlarge it back: simulates low-resolution photos.

    The v1 model relied on pixel-level detail and missed 76% of fakes after a
    50% downscale. This forces it to also use cues that survive resizing.
    """

    def __init__(self, p: float = 0.25, scale: tuple[float, float] = (0.35, 0.9)):
        self.p, self.scale = p, scale

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        w, h = image.size
        f = random.uniform(*self.scale)
        small = image.resize((max(1, round(w * f)), max(1, round(h * f))), Image.Resampling.BILINEAR)
        return small.resize((w, h), Image.Resampling.BILINEAR)


def get_transforms(train: bool) -> transforms.Compose:
    """Return the preprocessing pipeline for training or evaluation.

    Training uses ROBUSTNESS augmentation (Wang et al., CVPR 2020, "CNN-
    generated images are surprisingly easy to spot... for now"): every epoch
    each image randomly gets the kinds of damage real-world images suffer,
    so the model can't depend on fragile pixel-perfect detail:
      - RandomResizedCrop (80-100% area): slight zoom and shift, so small
        alignment differences don't matter,
      - RandomHorizontalFlip: a mirrored face is still real/fake,
      - ColorJitter: lighting / white-balance differences,
      - RandomDownscale (25%): low-resolution or resized images,
      - GaussianBlur (20%): soft focus, motion, smoothing filters,
      - RandomJPEG (50%, quality 30-95): re-compression when shared online.
    JPEG comes last because that's the order it happens in real life
    (capture -> edit/resize -> compress when uploading).

    Evaluation applies no randomness, so val/test results are deterministic.

    Both then convert to a tensor in [0, 1] and normalise with ImageNet
    mean/std, as the pretrained EfficientNet expects.
    """
    common = [
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ]
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            RandomDownscale(p=0.25),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
            RandomJPEG(p=0.5),
            *common,
        ])
    return transforms.Compose(common)


def get_dataloader(split: str, batch_size: int = 64,
                   num_workers: int = config.NUM_WORKERS,
                   seed: int = config.SEED) -> DataLoader:
    """Build the DataLoader for one split ("train", "val" or "test").

    - Only the training set is shuffled and augmented; val/test are read in a
      fixed order with no randomness, so their results are deterministic.
    - pin_memory=True: page-locked host memory speeds up CPU->GPU copies.
    - persistent_workers=True: keep worker processes alive between epochs
      instead of re-spawning them each time.
    - generator + worker_init_fn: make the shuffle order and per-worker
      augmentation reproducible for a given seed.
    """
    is_train = split == "train"
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        FaceDataset(split, transform=get_transforms(train=is_train)),
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def get_dataloaders(batch_size: int = 64, num_workers: int = config.NUM_WORKERS,
                    seed: int = config.SEED) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders (see get_dataloader for the details)."""
    return {split: get_dataloader(split, batch_size, num_workers, seed)
            for split in ("train", "val", "test")}


# ---------------------------------------------------------------------------
# Sanity check: `python -m src.data`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    from torchvision.utils import save_image

    from src.utils import set_seed

    set_seed(config.SEED)
    loaders = get_dataloaders(batch_size=64)

    # 1) Split sizes, class balance and composition by source/generator.
    from collections import Counter

    for split in ("train", "val", "test", *config.UNSEEN_SPLITS):
        dataset = FaceDataset(split)
        labels = [label for _, label in dataset.samples]
        n_fake = sum(labels)
        groups = ", ".join(f"{g} {n}" for g, n in Counter(dataset.groups).items())
        print(f"{split:16s}: {len(labels):6d} images ({len(labels) - n_fake} real / {n_fake} fake)  [{groups}]")

    # 2) One batch: shape, dtype, value range after normalisation.
    images, labels = next(iter(loaders["train"]))
    print(f"\nbatch images: {tuple(images.shape)} {images.dtype}, "
          f"range [{images.min():.2f}, {images.max():.2f}]")
    print(f"batch labels: {tuple(labels.shape)}, fakes in batch = {labels.sum().item()}/{len(labels)}")

    # 3) Save a grid of 32 un-normalised (augmented) training images for a visual check.
    mean = torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = config.RESULTS_DIR / "sample_batch.png"
    save_image((images[:32] * std + mean).clamp(0, 1), grid_path, nrow=8)
    print(f"saved preview grid -> {grid_path}")
    print("labels of preview (row by row, 0=real 1=fake):")
    for r in range(4):
        print("  ", labels[r * 8:(r + 1) * 8].tolist())

    # 4) Loader throughput: how many images/sec the workers can supply.
    start, n = time.perf_counter(), 0
    for i, (images, _) in enumerate(loaders["train"]):
        n += images.size(0)
        if i == 50:
            break
    elapsed = time.perf_counter() - start
    print(f"\nloader throughput: {n / elapsed:.0f} images/sec "
          f"({len(loaders['train'])} batches/epoch)")
