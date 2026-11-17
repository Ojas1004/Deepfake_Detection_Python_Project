"""
pretrained.py - A PRETRAINED reference detector downloaded from Hugging Face.

    python -m src.pretrained          # evaluate it on our test + unseen-generator sets

This model was NOT trained in this project. It is included as an external
reference point, to compare our own models against a published detector:

    CommunityForensics DeepfakeDet-ViT
    Paper:  "Community Forensics: Using Thousands of Generators to Train Fake
            Image Detectors", CVPR 2025 (arXiv:2411.04125)
    Model:  https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT
            (MIT licence; a ViT-Small trained on 2.7M images from 4,803 generators)

It is a Vision Transformer, not a CNN, so our Grad-CAM explanations (which
read a convolutional layer) don't apply to it; the app shows its verdict and
confidence only.

Preprocessing follows the model card exactly: shortest edge -> 440 (bicubic),
centre crop 384x384, CLIP normalisation. Output: one logit,
sigmoid(logit) = P(fake).
"""

import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src import config

MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
REVISION = "ac6ee457bea904a373065754107451793b56db00"  # pinned commit -> reproducible results
RUN_NAME = "pretrained_communityforensics"             # results/<RUN_NAME>/ like our own runs
LABEL = "Pretrained reference: CommunityForensics ViT (CVPR 2025) - not trained here"

# Pretrained detectors offered in the DEMO APP (each selectable on its own).
# Numbers in "limits" were measured by src/wild_eval.py on recent, convincing AI
# images (GPT-image, Midjourney 7, FLUX.2, ...) and real photos.
APP_DETECTORS = {
    "pretrained_swinv2": {
        "model_id": "haywoodsloan/ai-image-detector-deploy",
        "revision": "4ae1822603cb1ec980c217cfde1ef1b8e0477a16",
        "label": "SwinV2 AI-image detector - trained model",
        "short": "SwinV2 AI-image detector",
        # Scores between 0.5 and sure_fake are shown as "leans fake - uncertain".
        # Chosen from data: on 1,932 real benchmark photos (src/frontier_eval.py,
        # originals + WhatsApp/Instagram/re-shared copies) only 0.8% scored >= 0.95.
        "sure_fake": 0.95,
        "licence": "Apache-2.0",
        "wild_key": "SwinV2",
        "limits": "Best balance in our test: caught about 61-72% of recent AI fakes and wrongly "
                  "flagged about 12% of real photos.",
    },
    "pretrained_siglip": {
        "model_id": "Ateeqq/ai-vs-human-image-detector",
        "revision": "60e82406916921b823616bee33397baab38af3f0",
        "label": "SigLIP AI-vs-human detector - trained model",
        "short": "SigLIP AI-vs-human detector",
        # SigLIP's scores pile up near 1.0 even for real photos (8.8% of real
        # benchmark photos scored >= 0.999); 1% of them score >= 0.9999.
        "sure_fake": 0.9999,
        "licence": "Apache-2.0",
        "wild_key": "SigLIP",
        "limits": "Catches the most fakes in our test (about 80-92%), but also wrongly flagged "
                  "about half (54%) of real photos as fake, so a FAKE verdict is weak evidence.",
    },
}

FAKE_LABELS = {"ai", "artificial", "deepfake", "fake"}

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class PretrainedDetector:
    def __init__(self, device: torch.device):
        from transformers import ViTForImageClassification  # only needed for this model

        self.device = device
        self.model = _from_pretrained(ViTForImageClassification, MODEL_ID, REVISION)
        self.model.to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize(440, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])

    @torch.inference_mode()
    def logits(self, images: list[Image.Image]) -> np.ndarray:
        x = torch.stack([self.transform(im.convert("RGB")) for im in images]).to(self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            out = self.model(pixel_values=x).logits.squeeze(1)
        return out.float().cpu().numpy().astype(np.float64)

    def p_fake(self, image: Image.Image, temperature: float = 1.0) -> float:
        return float(1 / (1 + np.exp(-self.logits([image])[0] / temperature)))


def _from_pretrained(loader, model_id: str, revision: str):
    """Load from the local cache first (no network), download only if missing.

    On some networks the Hugging Face CDN hangs forever over IPv6; the cache
    avoids that once a model has been downloaded once.
    """
    try:
        return loader.from_pretrained(model_id, revision=revision, local_files_only=True)
    except OSError:
        return loader.from_pretrained(model_id, revision=revision)


class HFDetector:
    """Any 2-class Hugging Face image classifier, given the whole image."""

    def __init__(self, model_id: str, revision: str, device: torch.device):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.processor = _from_pretrained(AutoImageProcessor, model_id, revision)
        self.model = _from_pretrained(AutoModelForImageClassification, model_id, revision).to(device).eval()
        labels = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.fake_idx = next(i for i, name in labels.items() if name in FAKE_LABELS)
        self.device = device

    @torch.inference_mode()
    def p_fake_batch(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits.float()
        return torch.softmax(logits, dim=1)[:, self.fake_idx].cpu().numpy()

    def p_fake(self, image: Image.Image) -> float:
        return float(self.p_fake_batch([image])[0])

    @torch.inference_mode()
    def margin_batch(self, images: list[Image.Image]) -> np.ndarray:
        """logit(fake) - logit(real) = log-odds of 'fake'. Unlike P(fake), it keeps
        changing even when the model is ~100% sure, so it shows small effects."""
        inputs = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits.float()
        return (logits[:, self.fake_idx] - logits[:, 1 - self.fake_idx]).cpu().numpy()

    def occlusion_map(self, image: Image.Image, grid: int = 8, batch_size: int = 16):
        """Where is the evidence? Blur each block of a grid x grid layout in turn.

        Returns (fake log-odds of the image, grid x grid array of changes).
        change = log-odds(fake) - log-odds(fake with that block blurred):
          > 0  blurring the block LOWERED the fake score -> evidence for FAKE
          < 0  blurring the block RAISED the fake score  -> evidence for REAL
        Log-odds rather than P(fake): when a model is ~100% sure, P(fake) stays
        at 1.000 whatever is blurred, but the log-odds still show the effect.
        Blurring removes the fine pixel detail detectors rely on, while keeping
        the image's layout.

        Why occlusion and not Grad-CAM for these transformer models, measured
        with a faithfulness test on 30 detected fakes per model (blur the
        hottest 20% of the image vs a random 20% of blocks):
          - Grad-CAM: P(fake) drop 0.092 vs 0.039 random (SwinV2), 0.016 vs
            0.015 (SigLIP: no better than random);
          - this occlusion map: fake log-odds drop 4.01 vs 0.29 (SwinV2) and
            2.31 vs -0.54 (SigLIP), better than random on 100% of images.
        """
        from PIL import ImageFilter

        image = image.convert("RGB")
        original = np.asarray(image)
        blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(8)))
        h, w = original.shape[:2]
        variants = []
        for r in range(grid):
            for c in range(grid):
                a = original.copy()
                ys, ye, xs, xe = r * h // grid, (r + 1) * h // grid, c * w // grid, (c + 1) * w // grid
                a[ys:ye, xs:xe] = blurred[ys:ye, xs:xe]
                variants.append(Image.fromarray(a))
        m0 = float(self.margin_batch([image])[0])
        m_blurred = np.concatenate([self.margin_batch(variants[i:i + batch_size])
                                    for i in range(0, len(variants), batch_size)])
        return m0, (m0 - m_blurred).reshape(grid, grid)


def heatmaps_from_occlusion(change: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Turn an occlusion grid into two smooth heatmaps (values 0..1) of image size (w, h).

    Evidence for FAKE = positive changes, evidence for REAL = negative changes.
    Both are scaled by the SAME maximum, so a weak map stays faint instead of
    being blown up to look important.
    """
    scale = max(np.abs(change).max(), 1e-6)
    out = []
    for part in (np.clip(change, 0, None), np.clip(-change, 0, None)):
        t = torch.tensor(part / scale, dtype=torch.float32)[None, None]
        t = torch.nn.functional.interpolate(t, size=(size[1], size[0]), mode="bicubic", align_corners=False)
        out.append(t[0, 0].clamp(0, 1).numpy())
    return out[0], out[1]


def app_detector(run: str, device: torch.device) -> HFDetector:
    """One of the pretrained detectors offered in the demo app (see APP_DETECTORS)."""
    d = APP_DETECTORS[run]
    return HFDetector(d["model_id"], d["revision"], device)


def _split_logits(detector: PretrainedDetector, split: str, batch_size: int = 64):
    """(labels, logits, groups) for every image of one split of OUR dataset."""
    from src.data import FaceDataset

    ds = FaceDataset(split)
    labels, logits = [], []
    for i in tqdm(range(0, len(ds.samples), batch_size), desc=split, leave=False):
        batch = ds.samples[i:i + batch_size]
        logits.append(detector.logits([Image.open(ds.root / p) for p, _ in batch]))
        labels += [label for _, label in batch]
    return np.array(labels), np.concatenate(logits), ds.groups


def main() -> None:
    """Evaluate the pretrained detector exactly like our own models (same images,
    same metrics, same files), so compare.py can put it side by side."""
    from sklearn.metrics import roc_auc_score

    from src.calibration import (binary_nll, expected_calibration_error, fit_temperature,
                                 sigmoid)
    from src.evaluate import compute_metrics, per_group_results, to_builtin
    from src.plots import plot_per_group
    from src.utils import get_device

    device = get_device()
    detector = PretrainedDetector(device)
    out_dir = config.RESULTS_DIR / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, logits, groups = _split_logits(detector, "test")
    probs = sigmoid(logits)
    metrics = compute_metrics(labels, probs, 0.5)
    metrics["per_group"] = per_group_results(groups, labels, probs, 0.5)
    metrics["unseen"] = {}
    real_probs = probs[labels == 0]
    for split in config.UNSEEN_SPLITS:
        u_labels, u_logits, u_groups = _split_logits(detector, split)
        u_probs = sigmoid(u_logits)
        g = u_groups[0]
        metrics["unseen"][g] = {
            "name": config.GROUP_NAMES.get(g, g), "n": int(len(u_labels)),
            "detection_rate": float((u_probs >= 0.5).mean()),
            "roc_auc_vs_test_reals": float(roc_auc_score(
                np.r_[np.zeros(len(real_probs)), u_labels], np.r_[real_probs, u_probs])),
        }

    val_labels, val_logits, _ = _split_logits(detector, "val")
    t = fit_temperature(val_logits, val_labels)
    calibration = {"temperature": t, "fitted_on": "validation set",
                   "val_nll_before": binary_nll(val_logits, val_labels),
                   "val_nll_after": binary_nll(val_logits / t, val_labels),
                   "test_ece_before": expected_calibration_error(probs, labels),
                   "test_ece_after": expected_calibration_error(sigmoid(logits / t), labels)}
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))
    report = {"model": "pretrained", "run": RUN_NAME, "source": f"{MODEL_ID}@{REVISION}",
              "num_test_images": int(len(labels)), "threshold": 0.5, "positive_class": "fake",
              **metrics, "calibration": calibration}
    (out_dir / "metrics.json").write_text(json.dumps(to_builtin(report), indent=2))
    plot_per_group(metrics["per_group"], metrics["unseen"], out_dir / "per_generator.png", LABEL)

    print(f"\n{LABEL}\nTest set: accuracy {metrics['accuracy']:.4f}, F1 {metrics['f1']:.4f}, "
          f"ROC-AUC {metrics['roc_auc']:.4f}")
    for g in metrics["per_group"].values():
        print(f"    {g['name']:32s} {g['correct_rate']:7.1%}   (n={g['n']}, {g['class']})")
    for g in metrics["unseen"].values():
        print(f"    {g['name']:32s} {g['detection_rate']:7.1%}   (n={g['n']}, fake, AUC vs reals "
              f"{g['roc_auc_vs_test_reals']:.3f})")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
