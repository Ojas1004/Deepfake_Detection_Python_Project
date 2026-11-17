"""
train_frozen.py - Train a small head on top of a FROZEN CLIP / DINOv2 backbone.

Examples:
    python -m src.train_frozen --model clip
    python -m src.train_frozen --model dinov2 --name dinov2_linear --hidden 0

How this differs from train.py:
  train.py changes every weight of the network (EfficientNet) or trains one
  from scratch (SimpleCNN). Here the big pretrained network is never changed,
  so its output for a given image is always the same. That allows a shortcut:

  1. FEATURES (slow, done once): run every training and validation image
     through the backbone and save the feature vectors to data/features/.
     Training images are passed through several times: once clean and
     --views times with the random robustness augmentation from data.py
     (JPEG, blur, downscale...), so the head still sees damaged images.
  2. HEAD (seconds): train the small head on those saved vectors.

  The saved features are reused by later runs, so trying another head design
  takes seconds instead of another pass over 44,000 images.

The result is a normal checkpoint (checkpoints/<name>_best.pt) that evaluate.py,
the benchmarks and the app load like any other model. It stores only the head;
the backbone is loaded from Hugging Face.

The test set is NOT touched here (see evaluate.py).
"""

import argparse
import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.data import FaceDataset, get_transforms
from src.models import BACKBONES, DISPLAY_NAMES, FrozenViT
from src.utils import get_device, seed_worker, set_seed

FEATURE_DIR = config.DATA_DIR / "features"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a head on a frozen CLIP / DINOv2 backbone.")
    p.add_argument("--model", choices=list(BACKBONES), required=True)
    p.add_argument("--name", default=None,
                   help="run name for checkpoint/results files (default: the model name)")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="backbone layers the head reads (default: all layers listed in models.BACKBONES)")
    p.add_argument("--patch-mean", action="store_true",
                   help="also use the average patch token of each layer, not just its CLS token")
    p.add_argument("--hidden", type=int, default=256,
                   help="size of one hidden layer in the head (0 = a single linear layer)")
    p.add_argument("--dropout", type=float, default=0.5, help="dropout on the features, before the head")
    p.add_argument("--views", type=int, default=2,
                   help="augmented copies of each training image, on top of the clean one")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=512, help="batch size for training the head")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--extract-batch-size", type=int, default=64, help="batch size for the backbone")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


# ---------------------------------------------------------------------------
# 1. Features
# ---------------------------------------------------------------------------
@torch.inference_mode()
def extract(model: FrozenViT, layers: tuple[int, ...], split: str, augment: bool, args, seed: int) -> np.ndarray:
    """Tokens of every image in `split` (images x layers x 2 x D), as float16 to halve the disk space."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(FaceDataset(split, transform=get_transforms(train=augment)),
                        batch_size=args.extract_batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        worker_init_fn=seed_worker, generator=generator)
    device = next(model.parameters()).device
    out = [model.tokens(images.to(device, non_blocking=True), layers).cpu().numpy().astype(np.float16)
           for images, _ in tqdm(loader, desc=f"features: {split}{' (augmented)' if augment else ''}", leave=False)]
    return np.concatenate(out)


def cached_features(model: FrozenViT, args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(train features, train labels, val features, val labels), computed once and cached.

    The cache always holds ALL candidate layers with CLS + patch-mean tokens,
    whatever --layers / --patch-mean say, so every head design can reuse it.
    Shape on disk: images x layers x 2 (CLS, patch mean) x hidden size.
    """
    folder = FEATURE_DIR / args.model
    folder.mkdir(parents=True, exist_ok=True)
    all_layers = tuple(BACKBONES[args.model]["layers"])

    def get(name: str, split: str, augment: bool, seed: int) -> np.ndarray:
        path = folder / f"{name}.npy"
        if not path.exists():
            np.save(path, extract(model, all_layers, split, augment, args, seed))
        return np.load(path, mmap_mode="r")  # memory-mapped: only the parts we use are read

    def select(feats: np.ndarray) -> np.ndarray:
        cols = []
        for layer in model.layer_idx:
            i = all_layers.index(layer)
            cols.append(np.asarray(feats[:, i, 0]))
            if model.patch_mean:
                cols.append(np.asarray(feats[:, i, 1]))
        return np.concatenate(cols, axis=1)

    labels = lambda split: np.array([label for _, label in FaceDataset(split).samples])
    train = [select(get("train_v0", "train", False, args.seed))]
    for v in range(1, args.views + 1):
        train.append(select(get(f"train_v{v}", "train", True, args.seed + v)))
    y_train = np.concatenate([labels("train")] * len(train))
    return np.concatenate(train), y_train, select(get("val", "val", False, args.seed)), labels("val")


# ---------------------------------------------------------------------------
# 2. Head
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.name = args.name or args.model
    args.layers = args.layers or list(BACKBONES[args.model]["layers"])
    set_seed(args.seed)
    device = get_device()

    ckpt_path = config.CHECKPOINT_DIR / f"{args.name}_best.pt"
    results_dir = config.RESULTS_DIR / args.name
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    vit_args = {"layer_idx": args.layers, "patch_mean": args.patch_mean,
                "hidden": args.hidden, "dropout": args.dropout}
    model = FrozenViT(args.model, **vit_args).to(device)
    start_time = time.time()
    x_train, y_train, x_val, y_val = cached_features(model, args)
    print(f"Model: {DISPLAY_NAMES[args.model]} | features per image: {x_train.shape[1]:,} | "
          f"head parameters: {sum(p.numel() for p in model.head.parameters()):,} | device: {device}")
    print(f"Train: {len(x_train)} feature vectors ({1 + args.views} views per image) | Val: {len(x_val)}")

    # Everything fits on the GPU at once (float16), so no DataLoader is needed.
    x_train = torch.from_numpy(x_train).to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    x_val = torch.from_numpy(x_val).to(device)
    y_val = torch.from_numpy(y_val).float().to(device)

    # Standardise each feature with statistics of the TRAINING set (stored in the model).
    # Computed in chunks: converting all features to float32 at once would not fit on the GPU.
    total = torch.zeros(x_train.shape[1], device=device, dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    for i in range(0, len(x_train), 8192):
        chunk = x_train[i:i + 8192].double()
        total += chunk.sum(dim=0)
        total_sq += (chunk * chunk).sum(dim=0)
    mean = total / len(x_train)
    model.feat_mean.copy_(mean.float())
    model.feat_std.copy_(((total_sq / len(x_train) - mean * mean).clamp_min(0).sqrt() + 1e-6).float())

    criterion = nn.BCEWithLogitsLoss()
    # AdamW = Adam with proper weight decay, which keeps the head's weights small
    # (it has thousands of inputs, so it could otherwise memorise the training set).
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    history = {"model": args.model, "run": args.name, "args": vars(args),
               "unfreeze_epoch": None, "best_epoch": None, "epochs": []}
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        epoch_start, lr = time.time(), optimizer.param_groups[0]["lr"]
        model.train()  # dropout on (the backbone is not used here at all)
        order = torch.randperm(len(x_train), device=device)
        total_loss, correct = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            logits = model.classify(x_train[idx]).squeeze(1)
            loss = criterion(logits, y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += ((logits > 0) == (y_train[idx] > 0.5)).sum().item()
        scheduler.step()
        train_loss, train_acc = total_loss / len(order), correct / len(order)

        model.eval()
        with torch.no_grad():
            logits = model.classify(x_val).squeeze(1)
            val_loss = criterion(logits, y_val).item()
            val_acc = ((logits > 0) == (y_val > 0.5)).float().mean().item()

        history["epochs"].append({"epoch": epoch, "phase": "head", "lr": lr,
                                  "train_loss": train_loss, "train_acc": train_acc,
                                  "val_loss": val_loss, "val_acc": val_acc,
                                  "time_sec": round(time.time() - epoch_start, 1)})
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            torch.save({"model_name": args.model, "run_name": args.name,
                        "model_state_dict": model.head_state_dict(), "vit_args": vit_args,
                        "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc,
                        "train_args": vars(args), "classes": config.CLASSES}, ckpt_path)
        with open(results_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"Epoch {epoch:2d}/{args.epochs} lr={lr:.1e} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}" + ("  * saved best" if improved else ""))

    best = history["epochs"][history["best_epoch"] - 1]
    print(f"\nDone in {(time.time() - start_time) / 60:.1f} min. Best epoch {best['epoch']}: "
          f"val loss {best['val_loss']:.4f}, val acc {best['val_acc']:.4f}")
    print(f"Checkpoint: {ckpt_path}\nNext: python -m src.evaluate --name {args.name} --num-workers 2")


if __name__ == "__main__":
    main()
