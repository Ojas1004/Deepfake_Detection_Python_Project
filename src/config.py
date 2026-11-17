"""
config.py - Central place for paths and constants shared by every script.

Keeping these in one file means data prep, training, evaluation and the demo
app always agree on image size, label meaning, normalisation, etc.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, wherever it lives)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # original downloads, one folder per source
PROCESSED_DIR = DATA_DIR / "processed"  # our aligned, balanced, pre-resized 224x224 dataset
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# YuNet face detector (OpenCV model zoo, MIT licence), used by src/faces.py.
YUNET_MODEL = PROJECT_ROOT / "assets" / "face_detection_yunet_2023mar.onnx"

# ---------------------------------------------------------------------------
# Data sources (see README "Datasets" for why each was chosen)
# ---------------------------------------------------------------------------
# 1. 140k Real and Fake Faces: real FFHQ photos + StyleGAN (2019) fakes.
KAGGLE_DATASET = "xhlulu/140k-real-and-fake-faces"
# 2. SFHQ-T2I: modern text-to-image fakes (FLUX.1, SDXL, DALL-E 3), MIT licence.
#    Read remotely from one 23 GB zip; only the images we use are downloaded.
SFHQ_T2I_ZIP = ("https://huggingface.co/datasets/bitmind/SyntheticFacesHighQuality-T2I"
                "/resolve/main/SyntheticFacesHighQuality.zip")
# 3. DeepFakeFace: real celebrity photos (IMDB-WIKI) + 3 kinds of fakes made
#    from them: Stable Diffusion 1.5 text-to-image, SD inpainting, InsightFace swaps.
DFF_ZIP = "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/{part}.zip"
# 4. StyleGAN3 fakes: used ONLY for testing (a generator the model never sees).
STYLEGAN3_KAGGLE = "troykueh/real-vs-fake-faces-stylegan3"
# 5. OpenFake (ComplexDataLab, CC BY-NC 4.0): fakes from ~70 generators released
#    2022-2025 (SD 3.5, FLUX.1/2, Midjourney 6, GPT-image-1, Imagen 4, ...) and
#    real photos from LAION and Pexels. The full dataset is ~4 TB; we use the
#    first OPENFAKE_SHARDS shards of core/train (~5.6 GB each). Its test splits
#    are NOT used here: src/wild_eval.py tests on them (OpenFakeTiny).
OPENFAKE_URL = ("https://huggingface.co/datasets/ComplexDataLab/OpenFake/resolve/main/"
                "core/train-00000-of-00032-{shard:05d}.parquet")
OPENFAKE_SHARDS = 7
OPENFAKE_DIR = RAW_DIR / "openfake"
# Generators that make drawings/anime or noise, not photo-like images: skipped.
OPENFAKE_EXCLUDE = r"anim|ghibli|noobai|look-back|illustrious|^pony$|tiny-random"
# Each OpenFake image goes to exactly one split, decided by a hash of its content.
OPENFAKE_SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Image standardisation: EVERY image, from every source, goes through
#   detect face -> FFHQ-style align to FACE_SIZE -> JPEG at JPEG_QUALITY -> IMG_SIZE
# so real and fake images share one processing history (no format shortcuts).
# ---------------------------------------------------------------------------
IMG_SIZE = 224        # EfficientNet-B0's native ImageNet resolution
FACE_SIZE = 256       # aligned face size; matches the 140k dataset's images
JPEG_QUALITY = 95     # matches the 140k dataset's JPEG quality
MIN_FACE_CROP = 128   # skip faces smaller than this in the original photo (too blurry)

# ---------------------------------------------------------------------------
# Dataset composition: number of images per (split, group).
# Each split is 50% real / 50% fake. Groups map to sources in prepare_data.py.
# ---------------------------------------------------------------------------
REAL_GROUPS = ["ffhq", "dff_real", "of_real"]
FAKE_GROUPS = ["stylegan1", "flux_schnell", "sdxl", "flux_dev", "flux_pro",
               "sd15_t2i", "sd_inpaint", "insightface", "of_fake"]

SPLIT_PLAN = {
    "train": {"ffhq": 12_000, "dff_real": 6_000,
              "stylegan1": 6_000,
              "flux_schnell": 2_000, "sdxl": 2_000, "flux_dev": 1_500, "flux_pro": 500,
              "sd15_t2i": 2_000, "sd_inpaint": 2_000, "insightface": 2_000,
              "of_real": 4_100, "of_fake": 4_100},
    "val":   {"ffhq": 2_000, "dff_real": 1_000,
              "stylegan1": 1_000,
              "flux_schnell": 333, "sdxl": 333, "flux_dev": 250, "flux_pro": 84,
              "sd15_t2i": 334, "sd_inpaint": 333, "insightface": 333,
              "of_real": 500, "of_fake": 500},
    "test":  {"ffhq": 2_000, "dff_real": 1_000,
              "stylegan1": 1_000,
              "flux_schnell": 333, "sdxl": 333, "flux_dev": 250, "flux_pro": 84,
              "sd15_t2i": 334, "sd_inpaint": 333, "insightface": 333,
              "of_real": 450, "of_fake": 450},
    # Generators NEVER used for training or validation: an honest test of how
    # well the detector generalises to fakes it has not seen. Fakes only; their
    # real counterpart for metrics like ROC-AUC is the "test" split's reals.
    "unseen_dalle3": {"dalle3": 1_123},       # all DALL-E 3 images in SFHQ-T2I
    "unseen_stylegan3": {"stylegan3": 2_000},
}
UNSEEN_SPLITS = ["unseen_dalle3", "unseen_stylegan3"]

# Human-readable names for reports and plots.
GROUP_NAMES = {
    "ffhq": "Real: FFHQ (Flickr)", "dff_real": "Real: IMDB-WIKI celebrities",
    "stylegan1": "StyleGAN (2019)", "stylegan3": "StyleGAN3 (unseen)",
    "flux_schnell": "FLUX.1 schnell", "flux_dev": "FLUX.1 dev", "flux_pro": "FLUX.1 pro",
    "sdxl": "Stable Diffusion XL", "dalle3": "DALL-E 3 (unseen)",
    "sd15_t2i": "Stable Diffusion 1.5", "sd_inpaint": "SD inpainting (edit)",
    "insightface": "InsightFace face swap",
    "of_real": "Real: LAION / Pexels photos",
    "of_fake": "OpenFake: 2022-25 generators",
}

# Label convention: "fake" is the positive class (1), since the task is
# *detecting* fakes. Precision/recall/F1 are reported for the fake class.
CLASSES = ["real", "fake"]          # index == label
CLASS_TO_IDX = {"real": 0, "fake": 1}

# ImageNet mean/std. EfficientNet's pretrained weights expect inputs normalised
# this way. We use the same for SimpleCNN so both models get identical inputs.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NUM_WORKERS = 6

# ---------------------------------------------------------------------------
# Verdict rules for OUR models, used by the web API (src/api.py).
# They only say FAKE at or above OWN_SURE_FAKE;
# P(fake) from 0.5 up to it is shown as "leans fake - uncertain".
# ---------------------------------------------------------------------------
OWN_SURE_FAKE = 0.95
# What that cut-off means in practice, measured on the frontier benchmark faces
# (204 real, 352 fake images incl. WhatsApp/Instagram copies), calibrated P(fake):
# run -> ((real photos flagged, fakes caught) at OWN_SURE_FAKE, (same) at 0.5).
OWN_MEASURED = {"clip": ((0.01, 0.42), (0.07, 0.63)), "effnet_v3": ((0.03, 0.17), (0.27, 0.60))}
