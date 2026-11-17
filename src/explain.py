"""
explain.py - Explainable AI: WHY did the model call this face real or fake?

Three pieces of evidence for one prediction:

1. CONFIDENCE SCORE (calibrated)
   P(fake) = sigmoid(logit / T), with temperature T from calibration.py.

2. SUSPICIOUS AREA HIGHLIGHTING - Grad-CAM
   Grad-CAM (Selvaraju et al., 2017) asks the last convolutional layer which
   of its feature maps mattered for the decision:
     a) forward pass, keep the layer's feature maps A_k (e.g. 1280 maps of 7x7),
     b) backward pass from the score, keep the gradients dScore/dA_k,
     c) weight of map k = average of its gradients (how much the score rises
        when that map gets stronger),
     d) heatmap = ReLU( sum_k weight_k * A_k ), upsampled to 224x224.
   ReLU keeps only regions that push TOWARDS the chosen class. Using
   score = +logit explains "fake", score = -logit explains "real".

3. EXPLANATION - which facial regions mattered, checked two ways
   The dataset faces are aligned (eyes, mouth, etc. are always in roughly the
   same place - see FACE_REGIONS), so each pixel can be given a region name.
     a) Attention: what share of the Grad-CAM heat falls in each region.
     b) Occlusion test: blur ONE region (removing its fine detail, where GAN
        artifacts live), predict again, and measure how much the confidence
        in the verdict drops. This is a cause-and-effect check, unlike
        Grad-CAM which only reads gradients.
   When both methods point at the same region, the explanation is stronger.

Limitations (important when presenting this):
  - These methods show WHERE the evidence is, not WHAT it is. The "what to
    look for" hints are general knowledge about GAN faces, not model output.
  - Region names assume a centred, aligned face like the training images.
  - EfficientNet's Grad-CAM grid is only 7x7, so its heatmaps are coarse.

Command line (saves figures for the report):
    python -m src.explain --name effnet --image path/to/face.jpg
    python -m src.explain --name effnet --test-samples 8
"""

import argparse
import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageFilter
from torch import nn

from src import config
from src.calibration import load_temperature, sigmoid
from src.data import get_transforms, to_model_size
from src.faces import FaceAligner, jpeg_roundtrip, standardize_face
from src.models import BACKBONES

EVAL_TRANSFORM = get_transforms(train=False)

# ---------------------------------------------------------------------------
# 1. Confidence
# ---------------------------------------------------------------------------
def confidence_level(confidence: float) -> str:
    """Turn the confidence (probability of the predicted class) into words."""
    if confidence >= 0.90:
        return "high"
    if confidence >= 0.70:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# 2. Grad-CAM
# ---------------------------------------------------------------------------
def get_target_layer(model_name: str, model: nn.Module) -> nn.Module:
    """The convolutional layer Grad-CAM reads.

    The LAST conv layer is used because it has the most high-level features
    while still keeping a spatial layout (later layers pool it away).
      - SimpleCNN: final ReLU of block 4, before its max-pool -> 256 maps of 28x28
      - EfficientNet-B0: final conv block (features[-1])      -> 1280 maps of 7x7
    """
    if model_name == "cnn":
        return model.features[3][5]
    return model.features[-1]


def grad_cam(model: nn.Module, target_layer: nn.Module, x: torch.Tensor,
             target_class: int) -> np.ndarray:
    """Grad-CAM heatmap (224x224, values 0..1) for ONE image tensor x (1x3x224x224).

    target_class 1 = evidence for FAKE, 0 = evidence for REAL.
    """
    store = {}

    def forward_hook(module, inputs, output):
        store["activations"] = output
        # Also capture the gradient flowing back into this output.
        output.register_hook(lambda grad: store.__setitem__("gradients", grad))

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        with torch.enable_grad():  # gradients are needed even though we're not training
            logit = model(x).squeeze()
            score = logit if target_class == 1 else -logit
            model.zero_grad(set_to_none=True)
            score.backward()
    finally:
        handle.remove()  # never leave hooks behind on the (cached) model
        model.zero_grad(set_to_none=True)

    activations = store["activations"].detach()[0]  # C x h x w
    gradients = store["gradients"].detach()[0]      # C x h x w
    weights = gradients.mean(dim=(1, 2))            # one importance weight per feature map
    cam = F.relu((weights[:, None, None] * activations).sum(dim=0))
    cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear",
                        align_corners=False)[0, 0]
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.cpu().numpy()


@torch.inference_mode()
def occlusion_cams(model: nn.Module, image: Image.Image, device: torch.device,
                   grid: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Heatmaps (evidence for FAKE, evidence for REAL) for transformer models (FrozenViT).

    Grad-CAM needs a convolutional layer and failed a faithfulness test on
    transformer detectors (see pretrained.occlusion_map), so here the image is
    split into a grid x grid layout, each block is blurred in turn and the
    model is run again. If the fake score (log-odds) drops, that block held
    evidence for FAKE; if it rises, evidence for REAL.
    """
    from src.pretrained import heatmaps_from_occlusion

    original = np.asarray(image)
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=6)))
    h, w = original.shape[:2]
    batch = [EVAL_TRANSFORM(image)]
    for r in range(grid):
        for c in range(grid):
            a = original.copy()
            ys, ye, xs, xe = r * h // grid, (r + 1) * h // grid, c * w // grid, (c + 1) * w // grid
            a[ys:ye, xs:xe] = blurred[ys:ye, xs:xe]
            batch.append(EVAL_TRANSFORM(Image.fromarray(a)))
    logits = model(torch.stack(batch).to(device)).squeeze(1).float().cpu().numpy()
    change = (logits[0] - logits[1:]).reshape(grid, grid)
    return heatmaps_from_occlusion(change, (w, h))


# Transparent -> yellow -> orange -> red: low values vanish, hot spots stand out.
HEAT_CMAP = LinearSegmentedColormap.from_list("heat", ["#fab219", "#eb6834", "#d03b3b"])


def overlay_heatmap(image: Image.Image, cam: np.ndarray, opacity: float = 0.6) -> Image.Image:
    """Blend the heatmap over the image; colour AND transparency follow the heat."""
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    colours = HEAT_CMAP(cam)[..., :3]
    alpha = (opacity * np.clip(cam, 0, 1) ** 1.5)[..., None]  # ^1.5: keep cool areas clear
    blended = base * (1 - alpha) + colours * alpha
    return Image.fromarray((blended * 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# 3. Facial regions
# ---------------------------------------------------------------------------
# (name, x0, y0, x1, y1) as fractions of image width/height, measured from the
# average of 2,000 aligned training faces (eyes at ~47% height, mouth ~73%).
# Painted in order, so later (smaller, more specific) regions override earlier
# ones; any pixel not covered is "Background".
FACE_REGIONS = [
    ("Hair & top of head", 0.15, 0.00, 0.85, 0.27),
    ("Ears & sides of face", 0.15, 0.27, 0.85, 0.80),
    ("Forehead", 0.27, 0.27, 0.73, 0.41),
    ("Eyes & eyebrows", 0.25, 0.41, 0.75, 0.53),
    ("Cheeks", 0.27, 0.53, 0.73, 0.68),
    ("Nose", 0.40, 0.53, 0.60, 0.66),
    ("Mouth & teeth", 0.32, 0.66, 0.68, 0.80),
    ("Chin & jaw", 0.25, 0.80, 0.75, 0.90),
    ("Neck & clothing", 0.15, 0.90, 0.85, 1.00),
]
BACKGROUND = "Background"
REGION_NAMES = [BACKGROUND] + [r[0] for r in FACE_REGIONS]

# General knowledge about where GAN-generated faces tend to go wrong. Shown to
# the user as "what to look for" - this is NOT something the model outputs.
GAN_ARTIFACT_HINTS = {
    BACKGROUND: "warped, melted or nonsensical shapes; blurry objects; garbled text",
    "Hair & top of head": "strands that smear, merge or fade into the background; painted-looking texture",
    "Ears & sides of face": "mismatched earrings, malformed or asymmetric ears",
    "Forehead": "waxy, over-smooth skin; small 'water droplet' blob artifacts (typical of StyleGAN)",
    "Eyes & eyebrows": "pupils of different shapes, mismatched reflections, eyes not looking the same way",
    "Cheeks": "over-smooth or repeating skin texture, odd shading",
    "Nose": "asymmetric nostrils, unnatural shading",
    "Mouth & teeth": "blurred, irregular or merged teeth; asymmetric lips",
    "Chin & jaw": "blurred jawline, face blending into the neck or background",
    "Neck & clothing": "collars or clothing that make no sense, asymmetric accessories",
}


def region_label_map(size: int = config.IMG_SIZE) -> np.ndarray:
    """size x size array where each pixel holds the index of its region in REGION_NAMES."""
    labels = np.zeros((size, size), dtype=np.int64)  # 0 = Background
    for idx, (_, x0, y0, x1, y1) in enumerate(FACE_REGIONS, start=1):
        labels[round(y0 * size):round(y1 * size), round(x0 * size):round(x1 * size)] = idx
    return labels


REGION_LABELS = region_label_map()


def region_attention(cam: np.ndarray) -> dict[str, float]:
    """Share of the total Grad-CAM heat that falls in each region (sums to 1)."""
    total = cam.sum()
    if total == 0:
        return {name: 0.0 for name in REGION_NAMES}
    return {name: float(cam[REGION_LABELS == i].sum() / total)
            for i, name in enumerate(REGION_NAMES)}


@torch.inference_mode()
def occlusion_test(model: nn.Module, image: Image.Image, device: torch.device,
                   temperature: float) -> dict[str, float]:
    """P(fake) after blurring each region in turn (all regions in one batch).

    Blurring removes fine detail - exactly where GAN artifacts tend to hide -
    while keeping the overall layout, so the image still looks like a face.
    """
    original = np.asarray(image)
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=6)))
    batch = []
    for i in range(len(REGION_NAMES)):
        mask = (REGION_LABELS == i)[..., None]
        batch.append(EVAL_TRANSFORM(Image.fromarray(np.where(mask, blurred, original))))
    logits = model(torch.stack(batch).to(device)).squeeze(1).float().cpu().numpy()
    probs = sigmoid(logits / temperature)
    return dict(zip(REGION_NAMES, probs.tolist()))


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------
@dataclass
class RegionEvidence:
    name: str
    attention: float           # share of Grad-CAM heat (0..1)
    verdict_prob_blurred: float  # P(predicted class) with this region blurred
    drop: float                # confidence lost when blurred (positive = region supported the verdict)


@dataclass
class Explanation:
    image: Image.Image          # the 224x224 image the model saw
    p_fake: float               # calibrated P(fake)
    is_fake: bool
    confidence: float           # P(predicted class)
    level: str                  # high / moderate / low
    calibrated: bool
    face_found: bool            # False -> no face detected, a plain centre crop was used
    cam_fake: np.ndarray        # Grad-CAM evidence for FAKE
    cam_real: np.ndarray        # Grad-CAM evidence for REAL
    cam_method: str = "Grad-CAM"  # how cam_fake / cam_real were made ("Grad-CAM" or "Occlusion map")
    regions: list[RegionEvidence] = field(default_factory=list)  # sorted, most influential first
    summary: list[str] = field(default_factory=list)              # plain-language sentences

    @property
    def verdict(self) -> str:
        return "FAKE" if self.is_fake else "REAL"

    @property
    def cam_verdict(self) -> np.ndarray:
        return self.cam_fake if self.is_fake else self.cam_real


def _fmt_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def build_summary(e: Explanation) -> list[str]:
    """Plain-language sentences describing the evidence."""
    verdict = e.verdict.lower()
    lines = [f"The model is **{e.level}ly confident** ({e.confidence:.0%}) that this face is "
             f"**{verdict}**." if e.level != "low" else
             f"The model leans towards **{verdict}**, but with **low confidence** "
             f"({e.confidence:.0%}): treat this result as uncertain."]

    by_attention = sorted(e.regions, key=lambda r: r.attention, reverse=True)
    top = [f"{r.name.lower()} ({r.attention:.0%})" for r in by_attention[:3] if r.attention >= 0.05]
    if top:
        lines.append(f"Its attention ({e.cam_method}) was concentrated on: {_fmt_list(top)}.")
    else:
        lines.append(f"{e.cam_method} found no clearly focused region for this verdict.")

    strongest = e.regions[0]  # regions are sorted by drop
    if strongest.drop >= 0.05:
        flips = strongest.verdict_prob_blurred < 0.5
        lines.append(
            f"Blurring the **{strongest.name.lower()}** lowered the model's confidence in "
            f"'{verdict}' from {e.confidence:.0%} to {strongest.verdict_prob_blurred:.0%}"
            + (" and **flipped the verdict**" if flips else "")
            + ", making it the most influential region.")
        if by_attention[0].name == strongest.name:
            lines.append("Both methods point to the same region, which makes this explanation more reliable.")
    elif e.is_fake and e.level != "low":
        lines.append(
            "Blurring any single region barely changes the result: the evidence is spread "
            "across the whole image, which is typical of the fine texture 'fingerprint' "
            "GANs leave behind.")
    else:
        lines.append("Blurring any single region barely changes the result: no one region "
                     "is decisive on its own.")
    return lines


def prepare_input(image: Image.Image, aligner: FaceAligner | None) -> tuple[Image.Image, bool]:
    """Preprocess an arbitrary image exactly like the training data.

    Normal path: detect + align the face (faces.standardize_face). If no face
    is found, fall back to a centre crop (with the same JPEG step), and report
    that so the app can warn the user.
    """
    if aligner is not None:
        face, _, status = standardize_face(image, aligner)
        if face is not None:
            return face, True
    return jpeg_roundtrip(to_model_size(image)), False


def explain_image(model: nn.Module, model_name: str, image: Image.Image,
                  device: torch.device, temperature: float | None = None,
                  aligner: FaceAligner | None = None, run_name: str | None = None,
                  standardized: bool = False) -> Explanation:
    """Full analysis of one image (any size; it is preprocessed like the training data).

    model_name is the architecture ("cnn"/"effnet", picks the Grad-CAM layer);
    run_name picks the calibration file (defaults to model_name).
    standardized=True: the image comes from data/processed and is already
    aligned and standardised, so it is used exactly as it is.
    """
    if temperature is None:
        temperature = load_temperature(run_name or model_name)
    calibrated = temperature is not None
    temperature = temperature or 1.0

    model.eval()
    if standardized:
        image, face_found = image.convert("RGB"), True
    else:
        image, face_found = prepare_input(image, aligner)
    x = EVAL_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logit = model(x).item()
    p_fake = float(sigmoid(np.array(logit / temperature)))
    is_fake = p_fake >= 0.5
    confidence = p_fake if is_fake else 1 - p_fake

    if model_name in BACKBONES:   # transformer: no conv layer for Grad-CAM, use occlusion
        cam_method = "Occlusion map"
        cam_fake, cam_real = occlusion_cams(model, image, device)
    else:
        cam_method = "Grad-CAM"
        layer = get_target_layer(model_name, model)
        cam_fake = grad_cam(model, layer, x, target_class=1)
        cam_real = grad_cam(model, layer, x, target_class=0)

    e = Explanation(image=image, p_fake=p_fake, is_fake=is_fake, confidence=confidence,
                    level=confidence_level(confidence), calibrated=calibrated,
                    face_found=face_found, cam_fake=cam_fake, cam_real=cam_real,
                    cam_method=cam_method)

    attention = region_attention(e.cam_verdict)
    blurred_p_fake = occlusion_test(model, image, device, temperature)
    for name in REGION_NAMES:
        p_verdict = blurred_p_fake[name] if is_fake else 1 - blurred_p_fake[name]
        e.regions.append(RegionEvidence(name, attention[name], p_verdict, confidence - p_verdict))
    e.regions.sort(key=lambda r: r.drop, reverse=True)
    e.summary = build_summary(e)
    return e


# ---------------------------------------------------------------------------
# Command line: figures for the report
# ---------------------------------------------------------------------------
def _save_single(e: Explanation, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt  # imported here: only the CLI draws figures

    from src.plots import _save  # shared style + save helper

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    panels = [(e.image, "Input (224×224)"),
              (overlay_heatmap(e.image, e.cam_fake), "Evidence for FAKE"),
              (overlay_heatmap(e.image, e.cam_real), "Evidence for REAL")]
    for ax, (img, sub) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(sub, fontsize=10, fontweight="normal")
        ax.axis("off")
    fig.suptitle(f"{title} - predicted {e.verdict}, confidence {e.confidence:.1%}",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, out_path)


def _save_grid(items: list[tuple[Explanation, str]], out_path: Path, model_label: str) -> None:
    import matplotlib.pyplot as plt

    from src.plots import _save

    n = len(items)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5.2))
    for col, (e, true_cls) in enumerate(items):
        correct = e.verdict.lower() == true_cls
        axes[0, col].imshow(e.image)
        axes[0, col].set_title(f"true: {true_cls}", fontsize=9, fontweight="normal")
        axes[1, col].imshow(overlay_heatmap(e.image, e.cam_verdict))
        axes[1, col].set_title(f"pred: {e.verdict.lower()} {e.confidence:.0%}"
                               + ("" if correct else "  ✗"), fontsize=9,
                               fontweight="normal" if correct else "bold")
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.suptitle(f"Grad-CAM on test images - {model_label} (heat = evidence for the predicted class)",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, out_path)


def main() -> None:
    from src.models import load_checkpoint, run_label
    from src.utils import get_device

    p = argparse.ArgumentParser(description="Explain predictions with Grad-CAM + occlusion.")
    p.add_argument("--name", required=True, help="run name, e.g. effnet or effnet_v1")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="explain one image file")
    group.add_argument("--test-samples", type=int, help="grid of N random test images (half real, half fake)")
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args()

    device = get_device()
    model, checkpoint = load_checkpoint(config.CHECKPOINT_DIR / f"{args.name}_best.pt", device)
    arch = checkpoint["model_name"]
    aligner = FaceAligner()
    out_dir = config.RESULTS_DIR / args.name / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = run_label(args.name, arch)

    if args.image:
        e = explain_image(model, arch, Image.open(args.image), device, aligner=aligner, run_name=args.name)
        if not e.face_found:
            print("WARNING: no face detected - used a centre crop instead.")
        out_path = out_dir / f"{args.image.stem}_explained.png"
        _save_single(e, out_path, args.image.name)
        print(f"Prediction: {e.verdict} (P(fake) = {e.p_fake:.3f}, "
              f"{'calibrated' if e.calibrated else 'uncalibrated'})\n")
        for line in e.summary:
            print(" -", line.replace("**", ""))
        print(f"\n{'Region':24s} {'attention':>9s} {'conf. if blurred':>17s} {'drop':>7s}")
        for r in e.regions:
            print(f"{r.name:24s} {r.attention:9.1%} {r.verdict_prob_blurred:17.1%} {r.drop:+7.1%}")
        print(f"\nSaved {out_path}")
        return

    # --test-samples: a reproducible, balanced random pick from the test set
    with open(config.PROCESSED_DIR / "manifest.csv", newline="") as f:
        test_rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    rng = random.Random(args.seed)
    half = args.test_samples // 2
    picks = (rng.sample([r for r in test_rows if r["class"] == "real"], args.test_samples - half)
             + rng.sample([r for r in test_rows if r["class"] == "fake"], half))
    # Test images are already aligned and standardised, so no aligner is needed here.
    items = [(explain_image(model, arch, Image.open(config.PROCESSED_DIR / r["path"]), device,
                            run_name=args.name, standardized=True),
              r["class"]) for r in picks]
    out_path = out_dir / "gradcam_test_samples.png"
    _save_grid(items, out_path, label)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
