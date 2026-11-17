"""
image_quality.py - Warn when an image is too poor for ANY detector to judge.

Detectors look for tiny pixel-level traces left by image generators. Those
traces are destroyed when an image is small (e.g. a screenshot), very dark
(little detail, lots of noise) or heavily JPEG-compressed (e.g. shared via
WhatsApp). The in-the-wild test (src/wild_eval.py) showed every detector
failing on such images, often with a confident-looking score. This module
spots those conditions so the app can say "this verdict is unreliable"
instead of showing false certainty.

The thresholds are simple rules of thumb, not learned values.
"""

import io

import numpy as np
from PIL import Image, ImageOps

MIN_SIDE = 400        # shortest side in pixels
MIN_BRIGHTNESS = 60   # mean brightness 0-255
MIN_JPEG_QUALITY = 70

# Standard JPEG luminance quantisation table (quality 50); scaled versions of it
# reveal the quality a JPEG was saved with.
_STD_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
             14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
             18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
             49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]


def jpeg_quality(image: Image.Image) -> int | None:
    """Estimate the quality (1-100) a JPEG was saved with, or None if not a JPEG."""
    tables = getattr(image, "quantization", None)
    if not tables:
        return None
    actual = sorted(tables[0])
    best, best_err = None, float("inf")
    for q in range(1, 101):
        scale = 5000 / q if q < 50 else 200 - 2 * q
        expected = sorted(min(255, max(1, (v * scale + 50) // 100)) for v in _STD_LUMA)
        err = sum(abs(a - b) for a, b in zip(expected, actual))
        if err < best_err:
            best, best_err = q, err
    return best


def quality_issues(image_bytes: bytes) -> list[str]:
    """Plain-language reasons why a verdict on this image would be unreliable."""
    raw = Image.open(io.BytesIO(image_bytes))
    quality = jpeg_quality(raw)
    image = ImageOps.exif_transpose(raw).convert("RGB")
    issues = []
    w, h = image.size
    if min(w, h) < MIN_SIDE:
        issues.append(f"the image is small ({w}×{h} px), e.g. a screenshot or thumbnail, "
                      "so fine details have been lost")
    brightness = float(np.asarray(image.convert("L")).mean())
    if brightness < MIN_BRIGHTNESS:
        issues.append(f"the image is very dark (brightness {brightness:.0f}/255), "
                      "which hides detail and adds noise")
    if quality is not None and quality < MIN_JPEG_QUALITY:
        issues.append(f"it is heavily JPEG-compressed (quality ≈ {quality}), as happens when "
                      "images are shared on messaging apps")
    return issues
