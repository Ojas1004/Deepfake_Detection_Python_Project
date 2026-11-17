"""
models.py - The classifiers we compare, plus checkpoint loading.

  1. SimpleCNN      - a small convolutional network trained from scratch.
  2. EfficientNet-B0 - a network pretrained on ImageNet, fine-tuned for our task
                       (transfer learning).
  3. FrozenViT      - a large pretrained Vision Transformer (CLIP or DINOv2)
                       that is never changed, plus a small trained head
                       (trained by src/train_frozen.py).

All output ONE number per image: a "logit" (raw score). sigmoid(logit) is the
predicted probability that the image is FAKE (label 1). One output plus
BCEWithLogitsLoss is the standard setup for binary classification.

Run directly to print parameter counts:
    python -m src.models
"""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src import config

MODEL_NAMES = ("cnn", "effnet")   # trained by src/train.py
DISPLAY_NAMES = {"cnn": "SimpleCNN (from scratch)", "effnet": "EfficientNet-B0 (fine-tuned)"}

# Frozen backbones for FrozenViT (trained by src/train_frozen.py). Each entry:
# Hugging Face id, pinned commit, the pixel mean/std the backbone was trained
# with, and which of its transformer layers the head reads.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
BACKBONES = {
    "clip": {"model_id": "openai/clip-vit-large-patch14", "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41",
             "mean": CLIP_MEAN, "std": CLIP_STD, "layers": (6, 9, 12, 15, 18, 21, 24),
             "display": "CLIP ViT-L/14 (frozen) + trained head"},
    "dinov2": {"model_id": "facebook/dinov2-base", "revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
               "mean": config.IMAGENET_MEAN, "std": config.IMAGENET_STD, "layers": (3, 6, 9, 12),
               "display": "DINOv2 ViT-B/14 (frozen) + trained head"},
}
DISPLAY_NAMES.update({name: b["display"] for name, b in BACKBONES.items()})


# ---------------------------------------------------------------------------
# 1. SimpleCNN
# ---------------------------------------------------------------------------
def conv_block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
    """One convolution block: (Conv -> BatchNorm -> ReLU) x 2 -> MaxPool -> Dropout.

    - Conv 3x3, padding 1: learns local patterns, keeps the spatial size.
      bias=False because BatchNorm right after it has its own bias (beta).
    - BatchNorm: normalises activations per channel, which stabilises and
      speeds up training and allows higher learning rates.
    - ReLU: the non-linearity.
    - MaxPool 2x2: halves height and width, so deeper layers see a larger
      area of the face and computation stays manageable.
    - Dropout2d: randomly zeroes whole feature maps during training to reduce
      overfitting (it is switched off automatically in eval mode).
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(dropout),
    )


class SimpleCNN(nn.Module):
    """A 4-block CNN built from scratch.

    Shape of the data as it flows through (batch size N):
        input          N x   3 x 224 x 224
        block 1        N x  32 x 112 x 112
        block 2        N x  64 x  56 x  56
        block 3        N x 128 x  28 x  28
        block 4        N x 256 x  14 x  14
        global avg pool N x 256           (average of each feature map)
        classifier     N x 1              (the logit)

    Channels double as the image shrinks: early layers find simple patterns
    (edges, colour), later layers combine them into more complex ones.
    Global average pooling instead of flattening keeps the classifier tiny
    (256 inputs instead of 256*14*14 = 50,176), which reduces overfitting.
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, dropout=0.1),
            conv_block(32, 64, dropout=0.1),
            conv_block(64, 128, dropout=0.2),
            conv_block(128, 256, dropout=0.2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


# ---------------------------------------------------------------------------
# 2. EfficientNet-B0 (transfer learning)
# ---------------------------------------------------------------------------
def build_efficientnet(pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 with its 1000-class ImageNet head replaced by 1 output.

    torchvision's EfficientNet has two parts:
      - model.features   : the convolutional "backbone" (feature extractor)
      - model.classifier : Dropout(0.2) -> Linear(1280, 1000)
    We keep the backbone (and its ImageNet knowledge) and swap the final
    Linear layer for a new, randomly initialised Linear(1280, 1).

    pretrained=False is used when loading our own checkpoint (evaluate.py,
    src/api.py): the weights come from the checkpoint, so downloading the
    ImageNet weights would be wasted effort.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze (trainable=False) or unfreeze (trainable=True) EfficientNet's backbone.

    Frozen parameters get no gradients, so the optimizer never changes them.
    We freeze for the first epochs so the new, random head can learn first:
    if everything were trainable from the start, the large random gradients
    from the head would flow back and damage the useful pretrained features.
    """
    for param in model.features.parameters():
        param.requires_grad = trainable


# ---------------------------------------------------------------------------
# 3. FrozenViT: frozen CLIP / DINOv2 backbone + small trained head
# ---------------------------------------------------------------------------
def load_backbone(name: str) -> nn.Module:
    """The pretrained Vision Transformer for BACKBONES[name], frozen, in eval mode.

    Downloaded from Hugging Face on first use (cached in ~/.cache/huggingface).
    For CLIP only the image half is loaded; its text half is not needed.
    """
    from transformers import AutoModel, CLIPVisionModel  # imported here: only FrozenViT needs it

    spec = BACKBONES[name]
    cls = CLIPVisionModel if "clip" in spec["model_id"] else AutoModel
    backbone = cls.from_pretrained(spec["model_id"], revision=spec["revision"]).eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


class FrozenViT(nn.Module):
    """A frozen CLIP or DINOv2 Vision Transformer with a small trained head.

    Why: EfficientNet learned what a "real photo" looks like only from our own
    ~22,000 real training photos, so unfamiliar real photos looked suspicious
    to it. CLIP and DINOv2 were trained on hundreds of millions of images of
    every kind. We never change their weights (so they cannot overfit to our
    data) and only train a small head on the features they produce.

    How an image flows through (batch size N):
        input            N x 3 x 224 x 224   (ImageNet-normalised, like all our models)
        re-normalise                         (to the mean/std the backbone expects)
        backbone         a ViT cuts the image into 16 x 16 patches of 14 px and
                         adds one summary token, "CLS": 257 tokens per layer
        features         for each layer in `layers`: its CLS token, and (if
                         `patch_mean`) the average of its 256 patch tokens,
                         all joined into one long vector
        standardise      (x - mean) / std per feature, measured on the training set
        head             Linear -> 1 logit   (or one hidden layer if hidden > 0)

    Reading several layers, not just the last one, matters for this task: the
    last layer mostly describes WHAT is in the image, while middle layers still
    carry the low-level texture traces that generators leave behind.
    """

    def __init__(self, backbone_name: str, layer_idx: tuple[int, ...], patch_mean: bool = False,
                 hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        spec = BACKBONES[backbone_name]
        self.backbone_name, self.layer_idx, self.patch_mean = backbone_name, tuple(layer_idx), patch_mean
        self.backbone = load_backbone(backbone_name)
        dim = self.backbone.config.hidden_size * len(self.layer_idx) * (2 if patch_mean else 1)

        # Buffers travel with the model (.to(device), state_dict) but are not trained.
        self.register_buffer("in_mean", torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("in_std", torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("bb_mean", torch.tensor(spec["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("bb_std", torch.tensor(spec["std"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("feat_mean", torch.zeros(dim))
        self.register_buffer("feat_std", torch.ones(dim))

        if hidden:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU(),
                                      nn.Dropout(0.3), nn.Linear(hidden, 1))
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 1))

    def train(self, mode: bool = True):
        """Like nn.Module.train(), but the frozen backbone always stays in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def tokens(self, x: torch.Tensor, layers: tuple[int, ...]) -> torch.Tensor:
        """CLS token and average patch token of each requested layer: N x len(layers) x 2 x D."""
        x = (x * self.in_std + self.in_mean - self.bb_mean) / self.bb_std
        # float16 on the GPU: about 3x faster for these large models, same features.
        with torch.no_grad(), torch.autocast(device_type=x.device.type, dtype=torch.float16,
                                             enabled=x.device.type == "cuda"):
            hidden_states = self.backbone(pixel_values=x, output_hidden_states=True).hidden_states
        # hidden_states[layer] is N x 257 x D; token 0 is CLS, the other 256 are patches.
        return torch.stack([torch.stack([hidden_states[l][:, 0], hidden_states[l][:, 1:].mean(dim=1)], dim=1)
                            for l in layers], dim=1).float()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Raw (not yet standardised) feature vector, N x dim, float32."""
        tokens = self.tokens(x, self.layer_idx)
        if not self.patch_mean:
            tokens = tokens[:, :, :1]       # keep only the CLS tokens
        return tokens.flatten(start_dim=1)  # layer by layer: CLS, then patch mean

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Head only: raw features (N x dim) -> logits (N x 1). Used for training on cached features."""
        return self.head((features.float() - self.feat_mean) / self.feat_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.features(x))

    def head_state_dict(self) -> dict:
        """Everything except the frozen backbone (which is re-downloaded, not stored)."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone.")}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def build_model(name: str, pretrained: bool = True, **vit_args) -> nn.Module:
    """Create a model by name: "cnn", "effnet" or a FrozenViT backbone name."""
    if name == "cnn":
        return SimpleCNN()
    if name == "effnet":
        return build_efficientnet(pretrained=pretrained)
    if name in BACKBONES:
        return FrozenViT(name, **vit_args)
    raise ValueError(f"Unknown model '{name}', expected one of {(*MODEL_NAMES, *BACKBONES)}")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def list_runs(checkpoint_dir: Path) -> list[str]:
    """Names of all trained runs, i.e. checkpoints/<run>_best.pt -> "<run>".

    A run is one trained model. Its name defaults to the architecture ("cnn",
    "effnet"); the first, StyleGAN-only models were kept as "cnn_v1" and
    "effnet_v1" for comparison.
    """
    runs = sorted(p.name.removesuffix("_best.pt") for p in checkpoint_dir.glob("*_best.pt"))
    # newest models first: plain names, then versioned ones
    return sorted(runs, key=lambda r: (r not in MODEL_NAMES, r))


def run_label(run: str, model_name: str) -> str:
    """Human-readable name for a run, e.g. 'EfficientNet-B0 (fine-tuned) - v1: StyleGAN only'."""
    base = DISPLAY_NAMES[model_name]
    if run.endswith("_v1"):
        return f"{base} - v1: StyleGAN only"
    if run == model_name and model_name in MODEL_NAMES:
        return f"{base} - v2: multi-source + robust"
    if run == model_name:
        return base
    return f"{base} [{run}]"


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Rebuild a model from a checkpoint saved by train.py, ready for inference.

    Returns the model (in eval mode, on `device`) and the checkpoint dict, which
    also holds metadata such as the epoch and validation accuracy.
    weights_only=True only loads tensors and plain Python values, never
    arbitrary pickled code, so loading a checkpoint is safe.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["model_name"] in BACKBONES:
        # FrozenViT: the checkpoint holds only the head; the backbone is loaded
        # from Hugging Face. strict=False allows exactly those missing keys.
        model = build_model(checkpoint["model_name"], **checkpoint["vit_args"])
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if unexpected or any(not k.startswith("backbone.") for k in missing):
            raise RuntimeError(f"{path} does not match the model: missing {missing}, unexpected {unexpected}")
    else:
        model = build_model(checkpoint["model_name"], pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


if __name__ == "__main__":
    for name in MODEL_NAMES:
        m = build_model(name)
        total, _ = count_parameters(m)
        out = m(torch.randn(2, 3, 224, 224))
        print(f"{DISPLAY_NAMES[name]:30s} params: {total:>10,}   output shape: {tuple(out.shape)}")
