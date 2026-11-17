# Next steps: retrain with OpenFake (written 2026-11-15)

Where things stand, and what to do next session. Run every command from the
project folder.

## 1. Do this next: train and evaluate `effnet_v3`

```fish
source .venv/bin/activate.fish; and python -m src.train --model effnet --name effnet_v3 --epochs 15; and python -m src.evaluate --name effnet_v3
```

- Takes about 18-20 minutes on the RTX 5060 (GPU). Ctrl+C keeps the best checkpoint so far.
- `--name effnet_v3` keeps the current v2 model (`checkpoints/effnet_best.pt`) for comparison.
- Skip SimpleCNN: it reached only 78% as v2.

## 2. Then measure it on current generators

These are the numbers that show whether the retrain helped:

```fish
python -m src.wild_eval          # OpenFakeTiny: 785 recent fakes, 65 real photos  -> results/wild_eval.md
python -m src.frontier_eval      # 19 current generators + social-media copies     -> results/frontier_eval.md
```

Both overwrite their existing report files. They load several large detectors
one after another; close other heavy apps first (see "Memory warning").

Optional, for the report:

```fish
python -m src.robustness --names effnet_v3 effnet effnet_v1
python -m src.compare
```

## 3. Numbers to beat

Fakes caught / real photos kept as real, threshold 0.5.

| Model | Own test set (6,000 images) | Frontier faces (88 fake, 51 real) | OpenFakeTiny (785 fake, 65 real) |
|---|---|---|---|
| EfficientNet v1 (StyleGAN only) | 64.3% accuracy | 12.5% / 100% | about 5% / 91% |
| EfficientNet v2 (`effnet`) | 97.3% accuracy | 31.8% / 86.3% | 47.6% / 75.4% |
| SimpleCNN v2 (`cnn`) | 77.9% accuracy | 39.8% / 92.2% | not measured |
| SwinV2 (downloaded, app default) | - | 78.4% / 94.1% | 67.6% / 87.7% |

EfficientNet v2 per generator: FLUX.1 100%, SDXL 99.7%, DALL-E 3 (never seen) 98.2%,
face swaps 92.5%, SD inpainting 90.4%, StyleGAN3 (never seen) 9.2%.

The goal of v3: catch more than 32-48% of current-generator fakes, and stop
flagging about 25% of unfamiliar real photos.

## 4. What changed this session

**Trained:** `effnet` and `cnn` (v2), evaluated into `results/effnet/` and `results/cnn/`.

**New data source: OpenFake** (ComplexDataLab, CC BY-NC 4.0)
- 7 shards of `core/train` downloaded to `data/raw/openfake/` (39 GB, 27,919 images).
- Aligned faces extracted to `data/raw/openfake/aligned/` (12,548 usable, cached).
- 10,100 of them added to `data/processed/`:

| Split | Real (LAION / Pexels) | Fake (63 generators) | Split total now |
|---|---|---|---|
| train | 4,100 | 4,100 | 44,200 |
| val | 500 | 500 | 7,000 |
| test | 450 | 450 | 6,900 |

- Fakes are picked round-robin across generators, so the common ones (SD 3.5, FLUX.1,
  GPT-image-1, DALL-E 3, Ideogram 3, ...) give about 182 faces each instead of a few dominating.
- Anime/drawing generators are excluded (`OPENFAKE_EXCLUDE` in `src/config.py`).
- Old manifest backed up: `data/processed/manifest_before_openfake.csv`.

**Code**
- `src/openfake.py` (new): download, face extraction, split planning.
- `src/prepare_data.py`: OpenFake is source 5; new `--add-openfake` mode.
- `src/config.py`: OpenFake settings, groups `of_real` / `of_fake`, split sizes.

## 5. Known issues and caveats

- **Test set changed** (6,900 images, was 6,000). Overall accuracy of `effnet_v3` is not
  directly comparable with v2's 97.3%; the per-generator rows are.
- **Label noise in OpenFake:** some LAION "real" images are drawings, some faces are in
  profile, a few fakes are stylised.
- **Newest generators are thin** in the training data: FLUX.2 about 200 faces, Imagen 4 has 90,
  nano-banana 8. Expect the biggest gains on 2022-2025 generators.
- **`check_gpu.py` is broken:** line 1 starts with a stray `321` and a backtick. Delete them.
- **App sidebar for v1 models** still shows 99.5% / 94.5% from the old StyleGAN-only test set.
  `python -m src.evaluate --name effnet_v1` (and `cnn_v1`) would replace them with honest
  numbers, but overwrites `results/effnet_v1/`.
- **README does not mention OpenFake or v3 yet.**
- **Combining detectors was tested and rejected:** "flag if EfficientNet v2 or SwinV2 says fake"
  caught 77.7% of fakes but kept only 65.8% of real photos (SwinV2 alone: 66.0% / 93.3%).
  Worth re-testing once v3 exists.

## 6. Memory warning

This machine has 15 GB RAM. When memory and swap fill up, the system kills the whole
VSCodium session (it happened twice on 2026-11-15, caused by the face extraction; that
leak is fixed). For any heavy job, run it in its own capped group so only the job dies:

```fish
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python -m src.wild_eval
```

Training itself was fine in earlier runs (uses the GPU, 6 data-loader workers).

## 7. If v3 is still weak on the newest generators

In order of expected payoff:
1. More OpenFake shards, or the `supplement/t2i` folder (17.5 GB, newest generators).
   Raise `OPENFAKE_SHARDS` in `src/config.py`, then rebuild with `--force`.
2. Add StyleGAN2/3 to training (v2 catches only 9% of StyleGAN3).
3. Swap the backbone for a frozen CLIP or DINOv2 ViT with a small trained head.
4. Train on full-resolution crops instead of 224 px resized faces.

## 8. Status on 2026-11-17: CLIP model added, best of our own models

Fakes caught / real photos kept as real at threshold 0.5, and ROC-AUC:

| Model | Own test set | Frontier faces (88 fake, 51 real) | OpenFakeTiny (785 fake, 65 real) |
|---|---|---|---|
| EfficientNet v3 (`effnet_v3`) | 95.1% accuracy | 63.6% / 76.5%, AUC 0.770 | 61.5% / 83.1%, AUC 0.791 |
| **CLIP ViT-L/14 frozen + head (`clip`)** | 98.9% accuracy | 86.4% / 84.3%, AUC 0.931 | 78.0% / 87.7%, AUC 0.925 |
| SwinV2 (downloaded) | - | 78.4% / 94.1%, AUC 0.904 | 67.6% / 87.7%, AUC 0.861 |

- `clip` still misses 40% of StyleGAN3 fakes (never seen in training) and keeps fewer frontier real
  photos than SwinV2 (84% vs 94%, only 51 photos).
- Train: `python -m src.train_frozen --model clip` (features are cached in `data/features/clip/`, 3.8 GB;
  with the cache the head trains in under a minute). Then `python -m src.evaluate --name clip --num-workers 2`.
- The head design (7 CLIP layers, one hidden layer of 256, dropout 0.5) was chosen using the validation
  set and OpenFakeTiny. The frontier benchmark was not looked at until the design was fixed.
- DINOv2 ViT-B/14 was also tried (`--model dinov2`, features cached): OpenFakeTiny AUC up to 0.89,
  below CLIP. DINOv2-large and CLIP+DINOv2 combined were not tried.
- The demo opens on `clip`. Our own models say FAKE only
  at P(fake) >= 0.95 (`OWN_SURE_FAKE`); 0.5-0.95 shows "leans fake - uncertain". The `clip` model's
  heatmap is an occlusion map, not Grad-CAM.
- Run GPU jobs one at a time: two feature extractions plus the app ran the 8 GB GPU out of memory.
- `src.evaluate` needs `--num-workers 2` and an 8 GB cap; with 6 workers it is killed at 6 GB.
- `frontier_eval` shrinks photos above 2048 px before cropping the face, the app does not. `effnet_v3`'s
  score on the same photo can differ a lot between the two (0.02 vs 0.65 on one image).
- 2026-11-17: all six models re-evaluated on the current 6,900-image test set (old results backed up in
  `results/_old_testset_backup/`), then `src.robustness` and `src.compare` run on all of them:
  `results/robustness.md`, `results/comparison.md`. The app sidebar now shows these honest numbers.
- README updated for OpenFake, v3 and `clip`; `check_gpu.py` line 1 fixed (2026-11-17).

## 9. Web front-end (2026-11-17)

- `web/`: Next.js 16 + React 19 + Tailwind 4 UI, black and white. Pages: Detect (`/`) and Benchmarks.
- `src/api.py`: Starlette API it calls (`python -m src.api`, port 8000). No new Python dependencies.
- `OWN_SURE_FAKE` and `OWN_MEASURED` live in `src/config.py`.
- 2026-11-17: the Streamlit app (`app.py`, `.streamlit/`) was removed; the web UI is the only demo.
- Tested in headless Chromium: upload, model switch, light/dark, 375 px phone width, Benchmarks page; no console errors.
