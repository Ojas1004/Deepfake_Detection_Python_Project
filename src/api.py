"""
api.py - HTTP API for the web front-end in web/: runs the models and explains their verdicts.

    python -m src.api                      # http://127.0.0.1:8000
    python -m src.api --port 8000

Endpoints:
  GET  /api/health       {"ok": true, "device": "cuda"}
  GET  /api/models       the models the UI can offer, with their measured results
  POST /api/analyze      multipart form: image=<file>, model=<run name>
                         -> verdict, P(fake), heatmaps, plain-language explanation
  GET  /api/benchmarks   result tables for the Benchmarks page

Built on Starlette + uvicorn. One request is analysed at a time (one GPU), and at most
MAX_LOADED models stay loaded, so an 8 GB GPU is enough.
"""

import argparse
import base64
import io
import json
import os
import threading
from collections import OrderedDict
from functools import lru_cache
from importlib.util import find_spec

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from src import config
from src.calibration import load_temperature
from src.explain import GAN_ARTIFACT_HINTS, confidence_level, explain_image
from src.faces import FaceAligner
from src.image_quality import quality_issues
from src.models import list_runs, load_checkpoint, run_label

PRETRAINED_AVAILABLE = find_spec("transformers") is not None
if PRETRAINED_AVAILABLE:
    from src import pretrained

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_LOADED = 3  # models kept in GPU memory at once

# Verdict rules for our own models: see src/config.py.
OWN_SURE_FAKE, OWN_MEASURED = config.OWN_SURE_FAKE, config.OWN_MEASURED
DEFAULT_ORDER = ("clip", "pretrained_swinv2", "effnet_v3")  # offered first, in this order
SHORT_NAMES = {"clip": "CLIP + head", "effnet_v3": "EfficientNet v3", "effnet": "EfficientNet v2",
               "cnn": "SimpleCNN v2", "effnet_v1": "EfficientNet v1", "cnn_v1": "SimpleCNN v1",
               "pretrained_swinv2": "SwinV2", "pretrained_siglip": "SigLIP"}

_lock = threading.Lock()   # one analysis at a time: the GPU and the face detector are shared
_loaded: OrderedDict = OrderedDict()
_aligner: FaceAligner | None = None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


@lru_cache(maxsize=None)
def _architecture(run: str) -> str:
    """Model architecture of a run, read from its checkpoint without building the model."""
    ckpt = torch.load(config.CHECKPOINT_DIR / f"{run}_best.pt", map_location="cpu", weights_only=True)
    return ckpt["model_name"]


def _benchmarks_for(name_matches) -> dict:
    """OpenFakeTiny + frontier (original face images) results of one detector, if measured."""
    out = {}
    wild = _read_json(config.RESULTS_DIR / "wild_eval.json") or {}
    w = next((v for k, v in wild.items() if name_matches(k)), None)
    if w:
        out["openfaketiny"] = {"fakes_caught": (w["fakes_caught_core"] + w["fakes_caught_reddit"]) / 2,
                               "reals_kept": w["reals_correct"], "roc_auc": w["roc_auc"]}
    frontier = (_read_json(config.RESULTS_DIR / "frontier" / "results.json") or {}).get("results", {})
    f = next((v for k, v in frontier.items() if name_matches(k)), None)
    if f and "original | faces" in f:
        r = f["original | faces"]
        out["frontier"] = {"fakes_caught": r["fakes_caught"], "reals_kept": r["reals_kept"],
                           "roc_auc": r["roc_auc"], "detected_at_1pct_fp": r["detected_at_1pct_fp"]}
    return out


def _short(label: str) -> str:
    """Short display name for a detector label from any results file."""
    for m in _catalog_cached():
        if label == m["label"] or (m["kind"] == "external" and m["short"] in label):
            return m["short"]
    return label.split(" (")[0]


@lru_cache(maxsize=1)
def _catalog_cached() -> tuple:
    return tuple(model_catalog())


def model_catalog() -> list[dict]:
    """Every model the UI can offer, most useful first."""
    models = []
    for run in list_runs(config.CHECKPOINT_DIR):
        arch = _architecture(run)
        label = run_label(run, arch)
        metrics = _read_json(config.RESULTS_DIR / run / "metrics.json")
        test = None
        if metrics:
            test = {"accuracy": metrics["accuracy"], "roc_auc": metrics["roc_auc"],
                    "precision": metrics["precision"], "recall": metrics["recall"],
                    "n_images": metrics["num_test_images"],
                    "unseen": [{"name": g["name"].replace(" (unseen)", ""), "detection_rate": g["detection_rate"]}
                               for g in metrics.get("unseen", {}).values()]}
        note = None
        if run in OWN_MEASURED:
            (sr, sf), (hr, hf) = OWN_MEASURED[run]
            note = (f"On the frontier benchmark faces: FAKE (P ≥ {OWN_SURE_FAKE}) flags about {sr:.0%} of real "
                    f"photos and catches about {sf:.0%} of fakes; at 0.5 it flags {hr:.0%} and catches {hf:.0%}.")
        models.append({
            "id": run, "label": label, "short": SHORT_NAMES.get(run, run), "kind": "ours", "architecture": arch,
            "input": "aligned face", "sure_fake": OWN_SURE_FAKE,
            "calibrated": load_temperature(run) is not None,
            "heatmap": "Occlusion map" if arch in ("clip", "dinov2") else "Grad-CAM",
            "test": test, "benchmarks": _benchmarks_for(lambda k, label=label: k == f"[ours] {label}"),
            "note": note,
        })
    if PRETRAINED_AVAILABLE:
        for run, d in pretrained.APP_DETECTORS.items():
            models.append({
                "id": run, "label": d["label"], "short": SHORT_NAMES.get(run, d["short"]), "kind": "external",
                "architecture": d["model_id"],
                "input": "whole image", "sure_fake": d["sure_fake"], "calibrated": False,
                "heatmap": "Occlusion map", "test": None, "licence": d["licence"],
                "benchmarks": _benchmarks_for(lambda k, key=d["wild_key"]: k.startswith("[external]") and key in k),
                "note": d["limits"],
            })
    rank = {r: i for i, r in enumerate(DEFAULT_ORDER)}
    models.sort(key=lambda m: (rank.get(m["id"], len(rank)), m["kind"] != "ours"))
    return models


def _get(run: str):
    """Load a model (or reuse it), keeping at most MAX_LOADED in memory."""
    global _aligner
    if _aligner is None:
        _aligner = FaceAligner()
    if run in _loaded:
        _loaded.move_to_end(run)
        return _loaded[run]
    while len(_loaded) >= MAX_LOADED:
        _loaded.popitem(last=False)
        torch.cuda.empty_cache()
    if PRETRAINED_AVAILABLE and run in pretrained.APP_DETECTORS:
        _loaded[run] = pretrained.app_detector(run, DEVICE)
    else:
        _loaded[run] = load_checkpoint(config.CHECKPOINT_DIR / f"{run}_best.pt", DEVICE)
    return _loaded[run]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _heat_png(cam: np.ndarray) -> str:
    """Heatmap (values 0..1) as a grayscale PNG; the front-end draws it in black and white."""
    return _png(Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8), mode="L"))


def _verdict(p_fake: float, sure: float) -> str:
    return "fake" if p_fake >= sure else ("uncertain" if p_fake >= 0.5 else "real")


def _analyse_own(run: str, image_bytes: bytes) -> dict:
    model, checkpoint = _get(run)
    e = explain_image(model, checkpoint["model_name"], Image.open(io.BytesIO(image_bytes)), DEVICE,
                      temperature=load_temperature(run), aligner=_aligner, run_name=run)
    return {
        "p_fake": e.p_fake, "confidence": e.confidence, "level": e.level, "calibrated": e.calibrated,
        "verdict": _verdict(e.p_fake, OWN_SURE_FAKE), "sure_fake": OWN_SURE_FAKE,
        "face_found": e.face_found, "image": _png(e.image), "image_caption": "Aligned face the model saw (224×224)",
        "heat_fake": _heat_png(e.cam_fake), "heat_real": _heat_png(e.cam_real), "heatmap": e.cam_method,
        "summary": e.summary,
        "regions": [{"name": r.name, "attention": r.attention, "blurred_confidence": r.verdict_prob_blurred,
                     "drop": r.drop, "hint": GAN_ARTIFACT_HINTS.get(r.name)} for r in e.regions],
    }


def _analyse_pretrained(run: str, image_bytes: bytes) -> dict:
    """Verdict of a downloaded detector on the ORIGINAL image + occlusion heatmaps."""
    import cv2

    detector = _get(run)
    d = pretrained.APP_DETECTORS[run]
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    p_fake = float(detector.p_fake(image))
    is_fake = p_fake >= 0.5
    confidence = p_fake if is_fake else 1 - p_fake

    shown = image.copy()
    shown.thumbnail((768, 768))
    margin, change = detector.occlusion_map(shown)
    cam_fake, cam_real = pretrained.heatmaps_from_occlusion(change, shown.size)

    verdict_map, word = (cam_fake, "FAKE") if is_fake else (cam_real, "REAL")
    strongest = change.max() if is_fake else -change.min()
    relative = strongest / max(abs(margin), 1e-6)
    summary = []
    if relative < 0.05:
        summary.append(f"No single area changes the verdict noticeably: the evidence for **{word}** "
                       "is spread over the whole image (typical of texture-level traces).")
    else:
        summary.append(f"Blurring the single most important block (1/64 of the image) removed about "
                       f"**{min(relative, 1):.0%}** of the model's evidence for {word}.")
        found = _aligner.detect(np.asarray(shown))
        if found is not None:
            quad = _aligner.ffhq_quad(found[0]).astype(np.int32)
            mask = np.zeros(verdict_map.shape, np.uint8)
            cv2.fillPoly(mask, [quad], 1)
            share = float(verdict_map[mask == 1].sum() / max(verdict_map.sum(), 1e-9))
            summary.append(f"**{share:.0%}** of the evidence for {word} lies on the face and head "
                           f"(which cover {float(mask.mean()):.0%} of the image); **{1 - share:.0%}** "
                           "lies elsewhere (background, body, clothing).")
    return {
        "p_fake": p_fake, "confidence": confidence, "level": confidence_level(confidence), "calibrated": False,
        "verdict": _verdict(p_fake, d["sure_fake"]), "sure_fake": d["sure_fake"], "face_found": True,
        "image": _png(shown), "image_caption": "The whole image (the model crops it itself)",
        "heat_fake": _heat_png(cam_fake), "heat_real": _heat_png(cam_real), "heatmap": "Occlusion map",
        "summary": summary, "regions": [],
    }


def analyse(run: str, image_bytes: bytes) -> dict:
    with _lock:
        is_pretrained = PRETRAINED_AVAILABLE and run in pretrained.APP_DETECTORS
        result = (_analyse_pretrained if is_pretrained else _analyse_own)(run, image_bytes)
    result["model"] = run
    result["quality_issues"] = quality_issues(image_bytes)
    return result


# ---------------------------------------------------------------------------
# Benchmarks page
# ---------------------------------------------------------------------------
def benchmarks() -> dict:
    runs = [r for r in list_runs(config.CHECKPOINT_DIR) if (config.RESULTS_DIR / r / "metrics.json").exists()]
    test = []
    for run in runs:
        m = _read_json(config.RESULTS_DIR / run / "metrics.json")
        rob = _read_json(config.RESULTS_DIR / run / "robustness.json")
        groups = m.get("per_group", {})
        test.append({
            "id": run, "label": run_label(run, m["model"]), "short": SHORT_NAMES.get(run, run),
            "n_images": m["num_test_images"],
            "accuracy": m["accuracy"], "roc_auc": m["roc_auc"],
            "openfake_fakes": groups.get("of_fake", {}).get("correct_rate"),
            "openfake_reals": groups.get("of_real", {}).get("correct_rate"),
            "unseen": {k: v["detection_rate"] for k, v in m.get("unseen", {}).items()},
            "robustness": {c: v["accuracy"] for c, v in rob["conditions"].items()} if rob else None,
        })
    test.sort(key=lambda r: -r["roc_auc"])

    def clean(name: str) -> tuple[str, str]:
        kind, _, label = name.partition("] ")
        return _short(label), ("ours" if kind == "[ours" else "external")

    wild = []
    for name, r in (_read_json(config.RESULTS_DIR / "wild_eval.json") or {}).items():
        label, kind = clean(name)
        wild.append({"label": label, "kind": kind, "fakes_caught_modern": r["fakes_caught_core"],
                     "fakes_caught_reddit": r["fakes_caught_reddit"], "reals_kept": r["reals_correct"],
                     "balanced_accuracy": r["balanced_accuracy"], "roc_auc": r["roc_auc"]})
    wild.sort(key=lambda r: -r["roc_auc"])

    frontier = []
    for name, variants in ((_read_json(config.RESULTS_DIR / "frontier" / "results.json") or {})
                           .get("results", {}).items()):
        label, kind = clean(name)
        rows = {v.split(" | ")[0]: r for v, r in variants.items() if v.endswith("| faces")}
        if "original" in rows:
            frontier.append({"label": label, "kind": kind, "variants": rows})
    frontier.sort(key=lambda r: -r["variants"]["original"]["roc_auc"])
    return {"test": test, "openfaketiny": wild, "frontier": frontier}


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "device": DEVICE.type})


async def models_endpoint(request: Request) -> JSONResponse:
    return JSONResponse({"models": await run_in_threadpool(model_catalog)})


async def benchmarks_endpoint(request: Request) -> JSONResponse:
    return JSONResponse(await run_in_threadpool(benchmarks))


async def analyze_endpoint(request: Request) -> JSONResponse:
    try:
        form = await request.form(max_files=1, max_fields=4, max_part_size=MAX_UPLOAD_BYTES)
    except Exception:
        return _error("The upload could not be read. Images up to 20 MB are accepted.", 400)
    upload, run = form.get("image"), form.get("model")
    if upload is None or not hasattr(upload, "read"):
        return _error("No image was uploaded.", 400)
    valid = {m["id"] for m in await run_in_threadpool(model_catalog)}
    if run not in valid:
        return _error(f"Unknown model '{run}'.", 400)
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        return _error("This image is larger than 20 MB. Please upload a smaller file.", 413)
    try:
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return _error("This file could not be read as an image. Try a JPEG, PNG or WebP photo.", 400)
    try:
        return JSONResponse(await run_in_threadpool(analyse, run, data))
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return _error("The GPU ran out of memory. Close other GPU programs and try again.", 503)


def build_app() -> Starlette:
    # Websites allowed to call this API: WEB_ORIGINS (comma-separated, exact) and,
    # optionally, WEB_ORIGIN_REGEX (e.g. Vercel preview URLs).
    origins = [o.strip() for o in os.environ.get("WEB_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")]
    origin_regex = os.environ.get("WEB_ORIGIN_REGEX") or None
    return Starlette(
        routes=[Route("/api/health", health), Route("/api/models", models_endpoint),
                Route("/api/benchmarks", benchmarks_endpoint),
                Route("/api/analyze", analyze_endpoint, methods=["POST"])],
        middleware=[Middleware(CORSMiddleware, allow_origins=origins, allow_origin_regex=origin_regex,
                               allow_methods=["GET", "POST"], allow_headers=["*"])],
    )


app = build_app()

if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser(description="HTTP API for the web front-end.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
