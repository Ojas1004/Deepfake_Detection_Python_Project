"""
train.py - Train one of the two models.

Examples:
    python -m src.train --model cnn    --epochs 30 --batch-size 64
    python -m src.train --model effnet --epochs 15 --batch-size 64
    python -m src.train --model effnet --name effnet_try2   # keep several runs side by side

What it does each epoch:
  1. trains on the training set (with AMP mixed precision),
  2. measures loss/accuracy on the validation set,
  3. lets ReduceLROnPlateau lower the learning rate if val loss stalls,
  4. saves the model to checkpoints/<name>_best.pt if val loss is the best so far,
  5. appends the epoch's numbers to results/<name>/history.json,
  6. stops early if val loss hasn't improved for --patience epochs.

The test set is NOT touched here. It is only used once, in evaluate.py, so the
reported test results are an honest estimate on unseen data.

Because history and the best checkpoint are written every epoch, stopping
training with Ctrl+C at any point still leaves usable results.
"""

import argparse
import json
import math
import time

import torch
from torch import nn
from tqdm import tqdm

from src import config
from src.data import get_dataloader
from src.models import (DISPLAY_NAMES, MODEL_NAMES, build_model,
                        count_parameters, set_backbone_trainable)
from src.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a real-vs-fake face classifier.")
    p.add_argument("--model", choices=MODEL_NAMES, required=True,
                   help="cnn = SimpleCNN from scratch, effnet = EfficientNet-B0 fine-tuned")
    p.add_argument("--name", default=None,
                   help="run name for checkpoint/results files (default: the model name)")
    p.add_argument("--epochs", type=int, default=20, help="maximum number of epochs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="learning rate (for effnet: used while the backbone is frozen)")
    p.add_argument("--finetune-lr", type=float, default=1e-4,
                   help="effnet only: lower learning rate once the backbone is unfrozen")
    p.add_argument("--freeze-epochs", type=int, default=2,
                   help="effnet only: epochs to train just the new head before unfreezing")
    p.add_argument("--patience", type=int, default=5,
                   help="early stopping: epochs without val-loss improvement before stopping")
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def make_optimizer(model: nn.Module, lr: float):
    """Adam over the currently trainable parameters, plus an LR scheduler.

    - Adam adapts the step size for each parameter individually, which makes
      it a robust default that needs little tuning.
    - ReduceLROnPlateau halves the learning rate when val loss hasn't improved
      for 2 epochs: big steps early on, finer steps once progress stalls.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)
    return optimizer, scheduler


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              desc: str = "") -> tuple[float, float]:
    """One pass over `loader`. Trains if an optimizer is given, otherwise only evaluates.

    Returns (average loss, accuracy). The caller sets model.train()/model.eval().

    Mixed precision (AMP):
      - torch.autocast runs most operations in float16, which is roughly twice
        as fast on the GPU's tensor cores and uses less memory; PyTorch keeps
        precision-sensitive operations (like the loss) in float32 automatically.
      - float16 can't represent very small numbers, so tiny gradients could
        round to zero ("underflow"). GradScaler multiplies the loss by a large
        factor before backward() and divides the gradients back before the
        optimizer step, which avoids this.
    """
    training = optimizer is not None
    use_amp = device.type == "cuda"
    total_loss, correct, seen = 0.0, 0, 0

    # Gradients are only needed when training; disabling them for evaluation
    # saves memory and time.
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc=desc, leave=False):
            # non_blocking=True overlaps the copy with computation (needs pin_memory).
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images).squeeze(1)  # shape (N,): one score per image
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)  # clear old gradients
                scaler.scale(loss).backward()          # backprop on the scaled loss
                scaler.step(optimizer)                 # unscale grads, then update weights
                scaler.update()                        # adapt the scale factor

            # logit > 0  <=>  sigmoid(logit) > 0.5  <=>  predicted "fake"
            preds = (logits > 0).long()
            batch = images.size(0)
            total_loss += loss.item() * batch
            correct += (preds == labels.long()).sum().item()
            seen += batch

    return total_loss / seen, correct / seen


def main() -> None:
    args = parse_args()
    args.name = args.name or args.model
    set_seed(args.seed)  # before building the model, so weight init is reproducible too
    device = get_device()

    ckpt_path = config.CHECKPOINT_DIR / f"{args.name}_best.pt"
    results_dir = config.RESULTS_DIR / args.name
    history_path = results_dir / "history.json"
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data -------------------------------------------------------------
    train_loader = get_dataloader("train", args.batch_size, args.num_workers, args.seed)
    val_loader = get_dataloader("val", args.batch_size, args.num_workers, args.seed)

    # ---- Model, loss, AMP scaler ------------------------------------------
    model = build_model(args.model).to(device)
    # Sigmoid + binary cross-entropy in one numerically stable function.
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    # ---- EfficientNet: start with the backbone frozen ---------------------
    is_effnet = args.model == "effnet"
    frozen = is_effnet and args.freeze_epochs > 0
    if frozen:
        set_backbone_trainable(model, False)
    optimizer, scheduler = make_optimizer(model, args.lr)

    total, trainable = count_parameters(model)
    print(f"Model: {DISPLAY_NAMES[args.model]} | params: {total:,} "
          f"(trainable now: {trainable:,}) | device: {device}")
    print(f"Train: {len(train_loader.dataset)} images | Val: {len(val_loader.dataset)} images | "
          f"batch size {args.batch_size}")

    history = {
        "model": args.model,
        "run": args.name,
        "args": vars(args),
        "unfreeze_epoch": args.freeze_epochs + 1 if frozen else None,
        "best_epoch": None,
        "epochs": [],
    }
    best_val_loss = math.inf
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- EfficientNet: unfreeze after --freeze-epochs ------------------
        if frozen and epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, True)
            frozen = False
            # A fresh optimizer is needed: the old one only knew the head's
            # parameters. A lower learning rate makes small, careful changes
            # so the pretrained ImageNet features aren't destroyed.
            optimizer, scheduler = make_optimizer(model, args.finetune_lr)
            _, trainable = count_parameters(model)
            print(f"\n>> Epoch {epoch}: backbone unfrozen, fine-tuning all {trainable:,} "
                  f"parameters at lr={args.finetune_lr}")

        phase = "frozen" if frozen else ("finetune" if is_effnet else "train")
        lr = optimizer.param_groups[0]["lr"]  # learning rate used during this epoch
        epoch_start = time.time()

        # ---- Train ---------------------------------------------------------
        model.train()  # enables dropout and BatchNorm batch statistics
        if frozen:
            # While frozen, also keep the backbone's BatchNorm layers in eval
            # mode, so their ImageNet running statistics aren't overwritten.
            model.features.eval()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device,
                                          optimizer, scaler, desc=f"epoch {epoch} train")

        # ---- Validate ------------------------------------------------------
        model.eval()  # disables dropout, uses BatchNorm's running statistics
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device,
                                      desc=f"epoch {epoch} val")

        scheduler.step(val_loss)

        # ---- Log -----------------------------------------------------------
        epoch_time = time.time() - epoch_start
        history["epochs"].append({
            "epoch": epoch, "phase": phase, "lr": lr,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "time_sec": round(epoch_time, 1),
        })

        # ---- Checkpoint + early stopping (both based on val loss) ----------
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            history["best_epoch"] = epoch
            torch.save({
                "model_name": args.model,
                "run_name": args.name,
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_args": vars(args),
                "classes": config.CLASSES,
            }, ckpt_path)
        else:
            epochs_without_improvement += 1

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"Epoch {epoch:2d}/{args.epochs} [{phase:8s}] lr={lr:.1e} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | {epoch_time:.0f}s"
              + ("  * saved best" if improved else ""))

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping: val loss hasn't improved for {args.patience} epochs.")
            break

    minutes = (time.time() - start_time) / 60
    if history["best_epoch"] is None:  # only possible if val loss was NaN every epoch
        print(f"\nDone in {minutes:.1f} min, but val loss never improved - no checkpoint saved.")
        return
    best = history["epochs"][history["best_epoch"] - 1]
    print(f"\nDone in {minutes:.1f} min. Best epoch {best['epoch']}: "
          f"val loss {best['val_loss']:.4f}, val acc {best['val_acc']:.4f}")
    print(f"Checkpoint: {ckpt_path}\nHistory:    {history_path}")
    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
