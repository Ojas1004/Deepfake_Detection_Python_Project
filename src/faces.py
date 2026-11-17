"""
faces.py - Find the face in an image and crop it exactly like the training data.

Why: every face in FFHQ (our real photos) and in the StyleGAN fakes is
"aligned": rotated so the eyes are level, scaled and centred so the eyes and
mouth always land in the same place. Images from other sources (wide shots,
head-and-shoulders portraits, phone photos) are framed differently. If we fed
them in unaligned, the model could learn "framing" instead of "fakeness", and
it would perform badly on uploads that aren't framed like FFHQ.

How:
  1. Detect the face with YuNet, a small, fast CNN face detector that ships
     with OpenCV (assets/face_detection_yunet_2023mar.onnx, MIT licence).
     It returns a box plus 5 landmarks: both eyes, nose tip, mouth corners.
  2. Apply the FFHQ alignment recipe (Karras et al., 2019, the code that built
     FFHQ) to those landmarks: it defines an oriented square around the face
     from the eye-to-eye and eye-to-mouth vectors.
  3. Warp that square to an upright image and resize it to the output size.

Used by prepare_data.py (all new sources) and by the demo app (uploads).
"""

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from src import config


@dataclass
class AlignedFace:
    image: Image.Image   # aligned face crop, out_size x out_size, RGB
    crop_size: float     # side length of the face square in the ORIGINAL image (px)
    score: float         # detector confidence (0..1)
    n_faces: int         # how many faces were detected in the image


class FaceAligner:
    """Detect the largest face in an image and return an FFHQ-style aligned crop."""

    # Detection is tried at these image sizes (longest side), smallest first.
    # YuNet misses faces that fill most of a big image (measured: 21% of 1024px
    # close-up portraits missed at full size, 1% at 512px), while small faces
    # in big photos need the larger size. The crop itself is always taken from
    # the full-resolution image; only detection uses the shrunk copy.
    DETECT_SIDES = (512, 1280)

    def __init__(self, score_threshold: float = 0.8):
        # score_threshold: ignore detections the model is less sure about.
        self.detector = cv2.FaceDetectorYN.create(
            str(config.YUNET_MODEL), "", (320, 320), score_threshold, 0.3, 5000)

    def _detect_at(self, rgb: np.ndarray, max_side: int):
        h, w = rgb.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        small = cv2.resize(rgb, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA) if scale < 1 else rgb
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        return (faces, scale) if faces is not None and len(faces) else (None, scale)

    def detect(self, rgb: np.ndarray):
        """Return (landmarks 5x2, score, n_faces) of the LARGEST face, or None."""
        for max_side in self.DETECT_SIDES:
            faces, scale = self._detect_at(rgb, max_side)
            if faces is not None:
                # Each row: x, y, w, h, 5 landmarks (x, y), score. Keep the biggest face.
                best = max(faces, key=lambda f: f[2] * f[3])
                landmarks = best[4:14].reshape(5, 2) / scale
                return landmarks, float(best[14]), len(faces)
            if max(rgb.shape[:2]) <= max_side:  # image already fully checked at native size
                return None
        return None

    @staticmethod
    def ffhq_quad(landmarks: np.ndarray) -> np.ndarray:
        """The FFHQ alignment square (4 corners) computed from 5 landmarks.

        Same maths as NVIDIA's ffhq-dataset/download_ffhq.py, with eye centres
        and mouth corners from YuNet instead of dlib's 68 landmarks.
        """
        eyes = landmarks[:2][np.argsort(landmarks[:2, 0])]   # sort: image-left eye first
        eye_left, eye_right = eyes[0], eyes[1]
        mouth_avg = landmarks[3:5].mean(axis=0)
        eye_avg = (eye_left + eye_right) / 2
        eye_to_eye = eye_right - eye_left
        eye_to_mouth = mouth_avg - eye_avg

        # x: the square's horizontal axis (rotated with the head), y: its vertical axis.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        return np.stack([c - x - y, c - x + y, c + x + y, c + x - y])  # TL, BL, BR, TR

    def align(self, image: Image.Image, out_size: int = 256) -> AlignedFace | None:
        """Aligned face crop of the largest face, or None if no face is found."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.asarray(image)
        found = self.detect(rgb)
        if found is None:
            return None
        landmarks, score, n_faces = found
        quad = self.ffhq_quad(landmarks)
        crop_size = float(np.hypot(*(quad[3] - quad[0])))

        # Warp the square to an upright image at roughly its native resolution
        # (so no detail is thrown away yet). Areas outside the photo are filled
        # by mirroring, as FFHQ does.
        native = max(out_size, int(round(crop_size)))
        dst = np.float32([[0, 0], [0, native], [native, native]])
        matrix = cv2.getAffineTransform(np.float32(quad[:3]), dst)
        warped = cv2.warpAffine(rgb, matrix, (native, native), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
        # Then downscale with LANCZOS, the same high-quality filter used everywhere else.
        face = Image.fromarray(warped).resize((out_size, out_size), Image.Resampling.LANCZOS)
        return AlignedFace(face, crop_size, score, n_faces)


def jpeg_roundtrip(image: Image.Image, quality: int = config.JPEG_QUALITY) -> Image.Image:
    """Compress and decompress as JPEG, so every image ends with the same compression."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def standardize_face(image: Image.Image, aligner: FaceAligner,
                     min_crop: int = 0) -> tuple[Image.Image | None, AlignedFace | None, str]:
    """The ONE preprocessing path for every image (dataset building AND the demo app):

        align face -> FACE_SIZE x FACE_SIZE -> JPEG (quality JPEG_QUALITY) -> IMG_SIZE x IMG_SIZE

    Returns (224x224 image or None, alignment info or None, status) where
    status is "ok", "no_face" or "too_small".
    """
    aligned = aligner.align(image, out_size=config.FACE_SIZE)
    if aligned is None:
        return None, None, "no_face"
    if aligned.crop_size < min_crop:
        return None, aligned, "too_small"
    face = jpeg_roundtrip(aligned.image)
    face = face.resize((config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.LANCZOS)
    return face, aligned, "ok"
