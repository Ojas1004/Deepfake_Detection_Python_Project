# Deepfake Face Detection: Real vs AI-Generated Faces

A binary image classifier that decides whether a face photo is **real** or
**AI-generated**, with a calibrated confidence score, suspicious-area heatmaps
and plain-language explanations. Three architectures are trained and compared:

| Model | Approach | Parameters |
|---|---|---|
| **SimpleCNN** | 4 convolution blocks (BatchNorm + Dropout), trained from scratch | 1.17 M |
| **EfficientNet-B0** | Pretrained on ImageNet, fine-tuned (transfer learning) | 4.01 M |
| **CLIP + trained head** | OpenAI's CLIP ViT-L/14 image encoder, **frozen**; only a small head on top is trained | 1.84 M trained (+ 303 M frozen) |

The project went through several versions, and all are kept for comparison:

| Version | Run names | Training data | Notes |
|---|---|---|---|
| **v1** | `cnn_v1`, `effnet_v1` | 140k Real and Fake Faces only (FFHQ vs 2019 StyleGAN) | 99.5% on its own StyleGAN-only test set, 62% on the current one |
| **v2** | `cnn`, `effnet` | 4 sources, 11 generators/manipulations, face-aligned, robustness augmentation | built to fix v1's weaknesses (see *Why v2*) |
| **v3** | `effnet_v3` | v2 data + **OpenFake** (63 generators from 2022-25, varied real photos) | better on current generators, but still flags many unfamiliar real photos |
| **CLIP** | `clip` | same data as v3 | **best model**; the demo app opens on it (see *Why the CLIP model*) |

---

## Results at a glance

**Own test set** (6,900 aligned faces, 50% real / 50% fake, all models on the same images; `results/comparison.md`):

| Model | Accuracy | ROC-AUC | OpenFake fakes caught | OpenFake real photos kept | StyleGAN3 caught (never seen in training) |
|---|---|---|---|---|---|
| **CLIP + head (`clip`)** | **98.9%** | **0.999** | **94.9%** | **97.1%** | **60.1%** |
| EfficientNet v3 | 95.1% | 0.990 | 81.8% | 88.7% | 14.6% |
| EfficientNet v2 | 93.5% | 0.978 | 49.6% | 87.6% | 9.2% |
| SimpleCNN v2 | 75.9% | 0.835 | 48.4% | 76.9% | 44.0% |
| EfficientNet v1 | 62.1% | 0.560 | 6.0% | 89.6% | 20.8% |
| SimpleCNN v1 | 61.0% | 0.627 | 2.4% | 97.3% | 14.9% |

**Current generators, never seen in training** (face images only, threshold 0.5; fakes caught / real photos kept):

| Detector | Frontier benchmark (88 fake, 51 real) | OpenFakeTiny (785 fake, 65 real) |
|---|---|---|
| **CLIP + head (ours)** | **86.4% / 84.3%**, AUC 0.931 | 78.0% / 87.7%, AUC **0.925** |
| SwinV2 (downloaded, not trained here) | 78.4% / **94.1%**, AUC 0.904 | 67.6% / 87.7%, AUC 0.861 |
| EfficientNet v3 (ours) | 63.6% / 76.5%, AUC 0.770 | 61.5% / 83.1%, AUC 0.791 |
| EfficientNet v2 (ours) | 31.8% / 86.3%, AUC 0.689 | 47.6% / 75.4%, AUC 0.717 |

On the frontier benchmark, when only 1% of real photos may be flagged, the CLIP model still catches
**62.5%** of the fakes (SwinV2: 9.1%, EfficientNet v3: 25.0%). Details: `results/frontier_eval.md`, `results/wild_eval.md`.

---

## 1. Setup

Tested on Arch Linux with an RTX 5060 Laptop GPU (Blackwell, 8 GB), 15 GB RAM and Python 3.14.
Either use the virtual environment (exact pinned versions) or Arch's system packages.

**Option A: virtual environment**
```fish
python -m venv .venv
source .venv/bin/activate.fish        # bash/zsh: source .venv/bin/activate
pip install -r requirements.txt
```
`requirements.txt` installs the **CUDA 12.8** build of PyTorch. The RTX 50-series
(compute capability `sm_120`) needs CUDA 12.8 or newer.

**Option B: Arch system packages**
```fish
sudo pacman -S python-pytorch-cuda python-torchvision-cuda python-scikit-learn python-matplotlib \
    python-pillow python-numpy python-tqdm python-starlette python-uvicorn python-python-multipart python-opencv
```
The CLIP model and the benchmarks also need `transformers` and `pyarrow`. If your distribution
doesn't package them, use Option A.

Note: the API needs `python-python-multipart` for uploads. The similarly named
`python-multipart` package is a different library and does not work.

The web UI also needs **Node.js 20 or newer** and npm.

**Check the GPU first**
```fish
python check_gpu.py
```
This must print `[OK]` for CUDA, a GPU matmul, AMP (float16) and a cuDNN convolution.

**Accounts.** You need a Kaggle API token (kaggle.com → *Settings* → *API* →
*Generate New Token*), saved as `~/.kaggle/access_token` with `chmod 600`.
A Hugging Face login (`hf auth login`) is optional; it only raises download rate limits.

Run every command from the project folder; scripts in `src/` run as modules (`python -m src.<name>`).

**Memory on a 15 GB machine.** Some jobs use a lot of RAM. When RAM and swap fill up, Linux's
out-of-memory killer can take the whole desktop session with it. Run heavy jobs in their own
capped group, so only the job is stopped:
```fish
systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 .venv/bin/python -m src.evaluate --name clip --num-workers 2
```
`src.evaluate` peaks at about 6.6 GB with `--num-workers 2` (the default of 6 workers needs more).
Run only one GPU job at a time on an 8 GB GPU, including the demo app.

---

## 2. Datasets

Many public "real vs fake face" datasets contain **shortcuts**: differences between
the real and fake halves that have nothing to do with fakeness (a different
resolution, JPEG quality, colour vs grayscale, EXIF metadata, even watermarks).
A model learns those instead of the real cues. Candidates were audited before
use, and several were rejected for exactly this reason. Chosen sources:

| Source | Real | Fake | Used for |
|---|---|---|---|
| [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | FFHQ (Flickr photos) | StyleGAN (2019) | train / val / test |
| [SFHQ-T2I](https://huggingface.co/datasets/bitmind/SyntheticFacesHighQuality-T2I) (MIT) | – | FLUX.1 schnell/dev/pro, SDXL | train / val / test |
| ″ | – | **DALL-E 3** | **test only (unseen generator)** |
| [DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace) | IMDB-WIKI celebrity photos | Stable Diffusion 1.5, SD **inpainting** (edits of the real photo), **InsightFace face swaps** | train / val / test |
| [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (CC BY-NC 4.0) | LAION and Pexels photos | 63 generators from 2022-25 (SD 3.5, FLUX.1/2, GPT-image-1, DALL-E 3, Ideogram 3, Imagen 4, ...) | train / val / test (from v3) |
| [StyleGAN3 faces](https://www.kaggle.com/datasets/troykueh/real-vs-fake-faces-stylegan3) | – | StyleGAN3 | **test only (unseen generator)** |

Composition (each split is 50% real / 50% fake):

| Split | Real | Fake | Total |
|---|---|---|---|
| train | 12,000 FFHQ + 6,000 IMDB-WIKI + 4,100 LAION/Pexels | 6,000 StyleGAN + 6,000 FLUX/SDXL + 6,000 SD1.5/inpainting/face-swap + 4,100 OpenFake | 44,200 |
| val | 2,000 FFHQ + 1,000 IMDB-WIKI + 500 LAION/Pexels | 1,000 + 1,000 + 1,000 + 500 (same mix) | 7,000 |
| test | 2,000 FFHQ + 1,000 IMDB-WIKI + 450 LAION/Pexels | 1,000 + 1,000 + 1,000 + 450 (same mix) | 6,900 |
| unseen_dalle3 | – | all DALL-E 3 faces that pass the face filter (1,094) | 1,094 |
| unseen_stylegan3 | – | 2,000 StyleGAN3 | 2,000 |

**OpenFake details.** Only the first 7 parquet shards of `core/train` are used (39 GB, 27,919 images);
12,548 of those images contain a usable face. Fakes are picked **round-robin across generators**, so common
generators don't dominate: the frequent ones give about 182 faces each, while rare, newer ones contribute
every face they have. Anime and drawing generators are excluded (`OPENFAKE_EXCLUDE` in `src/config.py`).
The licence is **non-commercial** (CC BY-NC 4.0).

**Two benchmark sets of current generators** are used only for testing, never for training:
- **OpenFakeTiny** (CC BY-NC 4.0, the OpenFake team's test sets): 850 face images, of which 297 are fakes from
  current generators (GPT-image-1.5, Midjourney 7, FLUX.2, Z-Image, Seedream 5, Veo 3 frames), 488 are AI images
  posted on Reddit and 65 are real photos.
- **[WasItAIGenerated Frontier benchmark](https://huggingface.co/datasets/wasitaigeneratedcom/frontier-image-video-bench-2026-10)**
  (public split, October 2026): 261 fakes from 19 current generators (GPT-Image-2, Nano Banana Pro/2.1, FLUX.3,
  Seedream 5 Pro, Qwen-Image 3, ...) and 486 real photos. Each image also comes as a WhatsApp, Instagram and
  re-shared copy. 139 originals contain a usable face (88 fake, 51 real). It was built by a company that sells a
  detector, and it is not face-specific.

---

## 3. Prepare the data (once)

```fish
python -m src.prepare_data     # ~4-5 GB of downloads + processing; can be re-run, downloads are cached
python -m src.prepare_data --add-openfake   # add OpenFake (~39 GB of downloads) to an existing data/processed
python -m src.data             # sanity check: composition, a batch, a preview grid
```

`python -m src.openfake` downloads the OpenFake shards and extracts their faces on its own; the
`--add-openfake` step runs it if needed. Raise `OPENFAKE_SHARDS` in `src/config.py` for more OpenFake
data, then rebuild with `--force`.

What `prepare_data.py` does:
1. **Plan:** decides with a fixed seed which images go into which split, before touching any pixels.
2. **Download only what's needed.** The two Hugging Face face datasets are single huge zips (SFHQ-T2I is 23 GB).
   `src/remote_zip.py` reads the zip's table of contents over HTTP and downloads individual images with
   HTTP range requests, so only about 10k of the 122k SFHQ images are fetched.
3. **Standardise every image identically** (`src/faces.py`):
   detect the face (YuNet) → FFHQ-style alignment → 256×256 → JPEG quality 95 → 224×224 PNG.
   Every source, real or fake, ends with the same processing history, so size, framing or
   compression can't give the class away.
4. **Keep** the first N good images per split and source, and write `data/processed/manifest.csv`.
   Faces that aren't found, or are smaller than 128 px in the original photo, are skipped in **all** sources.

Data-leakage rules:
- **140k:** the dataset's own train/valid/test folders are used.
- **DeepFakeFace:** every fake is made from one real photo, and there is one photo per person.
  Photos are split first, so a person's real image and its fakes are always in the same split.
- **OpenFake:** the split is decided by a hash of the image content, so an image and any exact duplicate
  always land in the same split.
- **Unseen generators:** DALL-E 3 and StyleGAN3 never appear in training or validation.

Two DeepFakeFace problems were found by inspecting the data and are corrected:
- **Stretched fakes:** inpainting and face-swap fakes were stretched to 512×512 from the original photo's shape (median 1.33×),
  which distorts the face. They are resized back to the original photo's size before alignment.
- **Small faces:** many real photos are wide shots with small faces, which would become blurry crops and a "real" giveaway.
  The minimum face size filter removes them.

---

## 4. Train

**SimpleCNN and EfficientNet** (`src/train.py`):
```fish
python -m src.train --model effnet --name effnet_v3 --epochs 15   # about 17 minutes on the RTX 5060
python -m src.train --model cnn    --epochs 30
```

| Option | Default | Meaning |
|---|---|---|
| `--model` | (required) | `cnn` or `effnet` |
| `--name` | model name | run name for `checkpoints/<name>_best.pt` and `results/<name>/` |
| `--epochs` | 20 | maximum epochs (early stopping may finish sooner) |
| `--batch-size` | 64 | images per step |
| `--lr` | 1e-3 | learning rate (EfficientNet: while the backbone is frozen) |
| `--finetune-lr` | 1e-4 | EfficientNet: learning rate after unfreezing |
| `--freeze-epochs` | 2 | EfficientNet: epochs with the backbone frozen |
| `--patience` | 5 | early stopping patience (epochs without val-loss improvement) |
| `--num-workers` | 6 | DataLoader worker processes |
| `--seed` | 42 | random seed |

The best checkpoint and `results/<name>/history.json` are written after every
epoch, so Ctrl+C at any time keeps the best model so far.

**CLIP + trained head** (`src/train_frozen.py`):
```fish
python -m src.train_frozen --model clip
```

The CLIP backbone (about 1.6 GB) is downloaded from Hugging Face on first use, pinned to one commit.
Because the backbone never changes, training works in two steps:
1. **Features (slow, once):** every training and validation image goes through CLIP, and the result is saved to
   `data/features/clip/` (3.8 GB). Each training image is used three times: once clean and twice with the random
   robustness augmentation. This takes about 20 minutes.
2. **Head (seconds):** the small head is trained on the saved features. With the cache in place, a new head design
   trains in under a minute.

| Option | Default | Meaning |
|---|---|---|
| `--model` | (required) | `clip`, or `dinov2` (DINOv2 ViT-B/14, tried and found weaker) |
| `--name` | model name | run name, as above |
| `--layers` | 6 9 12 15 18 21 24 | which transformer layers the head reads |
| `--patch-mean` | off | also use each layer's average patch token, not just its CLS token |
| `--hidden` | 256 | size of the head's hidden layer (0 = a single linear layer) |
| `--dropout` | 0.5 | dropout on the features |
| `--views` | 2 | augmented copies of each training image, on top of the clean one |
| `--epochs` | 25 | epochs (the checkpoint with the best validation loss is kept) |

The checkpoint stores only the head (7 MB); the backbone is loaded from Hugging Face when the model is used.

---

## 5. Evaluate

```fish
python -m src.evaluate --name clip --num-workers 2      # any trained run
python -m src.evaluate --name effnet_v3 --num-workers 2
python -m src.robustness --names clip effnet_v3 effnet effnet_v1 cnn cnn_v1 --batch-size 64 --num-workers 2
python -m src.compare                                   # side-by-side tables + chart for the report
```

`evaluate.py` writes to `results/<name>/`:

| File | Content |
|---|---|
| `metrics.json` | accuracy, precision, recall, F1, ROC-AUC, specificity, confusion matrix, **per-source results**, **unseen-generator results**, calibration |
| `per_generator.png` | share of each source/generator classified correctly; unseen generators in orange |
| `confusion_matrix.png`, `roc_curve.png` | standard test-set figures |
| `loss_curve.png`, `accuracy_curve.png` | training curves, best epoch and unfreeze point marked |
| `calibration.json`, `reliability_diagram.png` | confidence calibration (temperature fitted on the **validation** set) |

`robustness.py` re-runs the test set with JPEG compression (q90/70/50), 50% downscaling,
blur, WhatsApp-like processing and zoom. It saves `results/<name>/robustness.json` and
`results/robustness.md`. `compare.py` collects everything into `results/comparison.md`
and `results/comparison.png`.

Every model must be evaluated on the **same** test set for `compare.py` to be fair. When the test set
changes (it grew from 4,000 to 6,000 to 6,900 images), re-run `src.evaluate` for every run.
The old results from before the last change are kept in `results/_old_testset_backup/`.

Robustness of the CLIP model: it loses at most 3.3 points of accuracy under any of these conditions
(worst: WhatsApp-like, 95.6%). EfficientNet v3 drops to 85.5% when the face is zoomed out.

**Fake is the positive class:**
- **Precision** = of the images flagged fake, the share that really were fake.
- **Recall** = of all fakes, the share caught.
- **ROC-AUC** = how well the scores separate the classes at any threshold.

**Unseen generators** contain only fakes, so they report the detection rate, plus ROC-AUC against the test-set reals.

### Pretrained reference model (not trained in this project)

```fish
python -m src.pretrained        # evaluates it on the same test + unseen sets -> results/pretrained_communityforensics/
```

[CommunityForensics DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT)
is a published detector ("Community Forensics: Using Thousands of Generators to Train Fake Image Detectors",
CVPR 2025; MIT licence; ViT-Small trained on 2.7M images from 4,803 generators). It is included **only as an
external reference** to compare our own models against. The weights are pinned to one commit for reproducibility.

Measured on the v2 test set (6,000 images, before OpenFake was added), fed our aligned 224 px crops like our models:
- Accuracy 77.1%, ROC-AUC 0.870.
- Excellent on GAN faces: StyleGAN 100%, StyleGAN3 100%, DALL-E 3 84%.
- Weak on FLUX/SDXL crops (about 43–49%) and on face swaps (26%).
- 18% of real FFHQ photos flagged as fake.

On the **original full-resolution files** it does much better: FLUX 82%, SDXL 76%, real FFHQ 91%.
It was designed for whole images, so the demo app gives it the original upload.
It is not designed for face swaps (5% caught on the original files).

### Current generators: two benchmarks

```fish
python -m src.wild_eval                     # OpenFakeTiny -> results/wild_eval.md
python -m src.wild_eval --extra photo.png   # + each detector's verdict on your own image(s)
python -m src.frontier_eval                 # frontier benchmark -> results/frontier_eval.md
```

Our test set contains many older, easy fakes, so these two scripts test every detector on the two benchmark
sets of current generators described in *Datasets*. Both overwrite their report files. Our own models see the
aligned face; the downloaded detectors see the whole image, as they were designed.

OpenFakeTiny results (`results/wild_eval.md`):

| Detector | Fakes caught (modern / Reddit) | Reals kept | Balanced accuracy | ROC-AUC |
|---|---|---|---|---|
| **CLIP + head (ours)** | 79% / 77% | 88% | **82.8%** | **0.925** |
| SwinV2 (haywoodsloan) | 61% / 72% | 88% | 77.7% | 0.861 |
| EfficientNet v3 (ours) | 52% / 67% | 83% | 72.3% | 0.791 |
| SigLIP (Ateeqq) | 80% / 92% | 46% | 66.9% | 0.824 |
| EfficientNet v2 (ours) | 32% / 57% | 75% | 61.5% | 0.717 |
| CommunityForensics ViT | 23% / 25% | 95% | 59.6% | 0.771 |
| Our EfficientNet v1 | 3% / 7% | 91% | 48.3% | 0.525 |

Frontier benchmark, face images, originals (`results/frontier_eval.md`):

| Detector | ROC-AUC | Fakes caught @ 1% of reals flagged | Fakes caught (0.5) | Reals kept (0.5) |
|---|---|---|---|---|
| **CLIP + head (ours)** | **0.931** | **62.5%** | **86.4%** | 84.3% |
| CommunityForensics ViT | 0.909 | 53.4% | 44.3% | 100.0% |
| SwinV2 (haywoodsloan) | 0.904 | 9.1% | 78.4% | 94.1% |
| SigLIP (Ateeqq) | 0.772 | 11.4% | 73.9% | 68.6% |
| EfficientNet v3 (ours) | 0.770 | 25.0% | 63.6% | 76.5% |
| EfficientNet v2 (ours) | 0.689 | 8.0% | 31.8% | 86.3% |

On WhatsApp, Instagram and re-shared copies the CLIP model catches 47-61% of fakes and keeps 96% of real photos;
SwinV2 catches 52-60% and keeps 94-96%. Re-compression erases much of the evidence for every detector.

No detector is reliable on the most convincing modern fakes, especially small, dark or re-compressed images
such as screenshots. Detecting state-of-the-art AI images is still an open research problem.

**Caveats on these numbers.** Both benchmarks have few real face photos (51 and 65), so "reals kept" can be off by
about ±10 points. The CLIP model's head design was chosen partly using OpenFakeTiny, so its OpenFakeTiny numbers are
slightly optimistic. The frontier benchmark was not looked at until that design was fixed. `frontier_eval.py` shrinks
photos larger than 2,048 px before finding the face; the app does not. That changed EfficientNet v3's score a lot
on some photos (0.02 vs 0.65 on one), but the CLIP model's by about 0.01.

---

## 6. Explain predictions (figures for the report)

```fish
python -m src.explain --name effnet --test-samples 8          # Grad-CAM grid of 8 test images
python -m src.explain --name effnet --image path/to/photo.jpg # one image + region table
```

---

## 7. Run the demo

A black-and-white web UI built with Next.js and React, in `web/`. It needs the Python API (`src/api.py`),
which runs the models, alongside it:

```fish
.venv/bin/python -m src.api          # terminal 1: http://127.0.0.1:8000
cd web; npm install; npm run dev     # terminal 2: http://localhost:3000
```

Open http://localhost:3000, pick a model and upload any photo with a face. The **CLIP model** is selected
first; all trained runs and two downloaded detectors (SwinV2 and SigLIP) are also listed. A second page,
**Benchmarks**, shows the results tables. Details in `web/README.md`.
For our own models, the face is **detected and aligned automatically**, exactly like the training data;
the largest face is used. The downloaded detectors get the original image.

The page shows:
1. **Verdict & confidence:** Real / Fake / "leans fake – uncertain", a calibrated confidence (high/moderate/low) and P(fake).
2. **Where the evidence is:** a greyscale heatmap; switch between evidence for FAKE and evidence for REAL.
   Grad-CAM for SimpleCNN and EfficientNet, an occlusion map for the CLIP model and the downloaded detectors.
3. **Why:** plain-language reasons and a table of each facial region's influence.

**"Leans fake – uncertain" zone.** A hard FAKE verdict is shown only above a per-model threshold; scores between
0.5 and that threshold are shown as **uncertain**:
- **Our models: P(fake) ≥ 0.95** (`OWN_SURE_FAKE` in `src/config.py`). On the frontier benchmark faces (including
  social-media copies), the CLIP model flags about 1% of real photos at this threshold and catches about 42% of
  fakes; at 0.5 it flags 7% and catches 63%. EfficientNet v3 flags 27% of real photos at 0.5, so a plain
  0.5 cut-off would mislabel many real photos.
- **SwinV2 ≥ 0.95, SigLIP ≥ 0.9999:** chosen so that fewer than 1% of the 1,932 real benchmark photos
  (originals + social-media copies) score that high.

Example: a real Pixel 8a selfie scored 0.926 on SwinV2. Modern phones process every photo with neural networks
(multi-frame HDR, noise reduction), which leaves AI-like texture.

**Image-quality warning** (`src/image_quality.py`), for every model. Detectors fail on:
- small images (shortest side < 400 px),
- very dark images (mean brightness < 60/255),
- heavily compressed JPEGs (quality < 70).

For these, the app states that the verdict is unreliable whatever the score says.

**Heatmaps for transformer models.** The CLIP model, SwinV2 and SigLIP are transformers. Plain Grad-CAM proved
unreliable on them: in a faithfulness test, blurring its "hottest" 20% of the image hurt SigLIP's fake score no more
than blurring a random 20%. The app therefore uses an **occlusion map** instead. Each block of an 8×8 grid is blurred
in turn, and the change in the model's fake score (log-odds) shows which areas are evidence for FAKE and which for REAL.

On 30 detected fakes per model, blurring the 20% this map marks hottest lowered the fake log-odds by 4.01 (SwinV2) and
2.31 (SigLIP), against 0.29 and −0.54 for a random 20%. That is better than random on 100% of images for both models.
This test has not yet been repeated for the CLIP model. For the downloaded detectors, the app also reports how much
of the evidence lies on the face versus the background.

---

## Project structure

```
check_gpu.py          step 0: verify that PyTorch can use the GPU
assets/               YuNet face-detector model (OpenCV model zoo, MIT licence)
src/
  config.py           paths, dataset sources, split plan, constants
  utils.py            seeding (reproducibility) and device selection
  remote_zip.py       download single files out of huge remote zips (HTTP range requests)
  faces.py            face detection + FFHQ-style alignment + the shared standardisation
  openfake.py         OpenFake: download shards, extract faces, plan the splits
  prepare_data.py     plan, download, standardise, write data/processed/manifest.csv
  data.py             Dataset, robustness augmentation, DataLoaders
  models.py           SimpleCNN, EfficientNet-B0, FrozenViT (CLIP/DINOv2 + head), runs, checkpoint loading
  train.py            training loop for SimpleCNN / EfficientNet (AMP, Adam, ReduceLROnPlateau, early stopping)
  train_frozen.py     feature caching + head training for the CLIP model
  evaluate.py         test metrics, per-source + unseen-generator results, calibration
  robustness.py       accuracy under JPEG / resize / blur / zoom
  compare.py          side-by-side comparison of all runs
  calibration.py      temperature scaling, expected calibration error (ECE)
  explain.py          Grad-CAM, occlusion maps, facial-region analysis, explanation text
  image_quality.py    warnings for small, dark or heavily compressed uploads
  pretrained.py       downloaded detectors (CommunityForensics, SwinV2, SigLIP)
  wild_eval.py        OpenFakeTiny benchmark
  frontier_eval.py    frontier benchmark, incl. social-media copies
  api.py              HTTP API the web UI calls: models, analysis, benchmark results (Starlette + uvicorn)
  plots.py            all figures
web/                  Next.js + React front-end (black and white)
data/                 raw downloads, processed dataset, cached CLIP features (git-ignored)
checkpoints/          best model weights per run (git-ignored)
results/              history, metrics, figures per run + comparison + benchmark reports
```

---

## Why v2: what was wrong with v1

v1 scored 99.5% on its test set but performed poorly on real-world uploads. Measuring it made the reasons clear:

- **Fine detail:** it relied on fragile pixel-level detail. After a 50% downscale it caught only 24% of fakes; after blur, 66%.
- **JPEG confusion:** it mistook JPEG compression for GAN artifacts. At JPEG quality 50 it called 69% of *real* photos fake.
- **One generator:** it had only ever seen one 2019 generator (StyleGAN), so modern diffusion faces (FLUX, SDXL, DALL-E 3) looked unfamiliar.
- **Framing:** it expected FFHQ-style framing. With the face smaller in the frame, it caught only 45% of fakes.

v2 addresses each one:
1. **More generators:** GAN, diffusion, inpainting and face swaps.
2. **Face alignment:** applied to every image and every upload.
3. **Identical processing for every image:** removes format shortcuts.
4. **Robustness augmentation:** random JPEG, downscaling, blur and crops (Wang et al., CVPR 2020).
5. **Honest testing:** on generators never seen in training.

On the current test set, v1 reaches only 62% accuracy: it catches 0-11% of the diffusion and face-swap fakes.

## Why v3 and the CLIP model

**v2's problem:** on current generators it caught only 32-48% of fakes and flagged about 25% of unfamiliar real
photos, because it had seen a handful of generators and real photos from only two sources (Flickr, celebrities).

**v3** adds OpenFake: 63 more generators and varied LAION/Pexels real photos. It catches about twice as many
current-generator fakes as v2, but still flags 27% of the frontier benchmark's real photos at threshold 0.5.
Its score for real and fake photos overlaps too much to fix with a different threshold.

**The CLIP model** keeps the same data but replaces the network. EfficientNet learned what a "real photo" looks
like only from our ~22,000 real training photos. CLIP was trained by OpenAI on about 400 million image-caption
pairs, so ordinary photos of every kind are familiar to it. Its weights are **frozen**, so it cannot overfit to our
data; only a small head on top is trained. This approach is used in published detectors that generalise to unseen
generators (e.g. Ojha et al., CVPR 2023, frozen CLIP features with a linear classifier). It raised accuracy on OpenFake real photos from 88.7% to 97.1%, and on OpenFake fakes from 81.8% to 94.9%.

## Design decisions (and why)

**Data**
- **Balanced classes (50/50)** make accuracy meaningful: always guessing one class scores exactly 50%.
- **Robustness augmentation** (training only), applied randomly each time an image is loaded:
  - RandomResizedCrop (80–100% area)
  - horizontal flip, colour jitter
  - downscale-and-upscale (25% of images)
  - Gaussian blur (20%)
  - JPEG re-compression at quality 30–95 (50%), applied last, as happens in real life
- **The test set is only used in `evaluate.py`.** Model selection (best checkpoint, early stopping, LR schedule) uses validation data only.

**Models**
- **One output logit + `BCEWithLogitsLoss`** is the standard binary setup; `sigmoid(logit)` = P(fake).
- **SimpleCNN** doubles its channels (32→64→128→256) while each block halves the resolution (224→14).
  Global average pooling feeds a tiny classifier (256→1).
- **EfficientNet-B0** reuses features learned on 1.2M ImageNet images.

**EfficientNet fine-tuning in two phases**
1. *Epochs 1–2, backbone frozen:* only the new head (1,281 parameters) trains, at lr 1e-3.
   The backbone's BatchNorm layers stay in eval mode, so their ImageNet statistics are kept.
2. *From epoch 3, everything unfrozen:* all 4M parameters train at lr 1e-4, with a fresh Adam optimizer.

**CLIP model (`FrozenViT` in `src/models.py`)**
- **Input:** the same aligned 224 px face as the other models, re-normalised to CLIP's own pixel mean/std.
  The ViT cuts it into 16×16 patches of 14 px, plus one summary ("CLS") token.
- **Features from 7 layers, not just the last:** the CLS token of layers 6, 9, 12, 15, 18, 21 and 24 is joined into one
  7,168-number vector. The last layer mostly describes *what* is in the image; middle layers still carry the
  low-level texture traces that generators leave. Using all 7 layers clearly beat any single layer
  (OpenFakeTiny ROC-AUC up to 0.86 for the best single layer).
- **Head:** each feature is standardised with training-set statistics, then dropout 0.5 → Linear 7,168→256 → GELU →
  dropout 0.3 → Linear 256→1. Trained with AdamW (weight decay 0.01), cosine learning-rate schedule, 25 epochs; the
  checkpoint with the best validation loss is kept (epoch 14).
- **Backbone runs in float16** on the GPU, about 3× faster with the same features.
- **How the design was chosen:** about 30 head variants on two backbones were compared on the validation set and
  OpenFakeTiny. DINOv2 ViT-B/14 reached at most 0.89 ROC-AUC on OpenFakeTiny, below CLIP. DINOv2-large and a
  CLIP+DINOv2 combination were not tried.

**Training**
- **Adam** adapts each parameter's step size.
- **ReduceLROnPlateau** halves the LR after 2 epochs without improvement.
- **Early stopping** keeps the best checkpoint.
- **AMP mixed precision:** float16 on tensor cores, with `GradScaler` against gradient underflow.

**Reproducibility**
- All random generators (Python, NumPy, PyTorch CPU/CUDA, each DataLoader worker) are seeded, and cuDNN runs in deterministic mode.
- The dataset plan uses sorted file lists and a fixed seed.
- All downloaded models (CLIP, DINOv2, the reference detectors) are pinned to one Hugging Face commit.

**Explainable AI** (`src/explain.py`, `src/calibration.py`)
- **Calibrated confidence:** *temperature scaling* divides the logits by one number T before the sigmoid.
  T is fitted on validation data; predictions don't change, only the confidence values.
  ECE and the reliability diagram measure the improvement.
- **Grad-CAM** (Selvaraju et al., 2017) highlights where the evidence for *fake* or for *real* is.
  It is verified to match an independent implementation exactly. Used for SimpleCNN and EfficientNet.
- **Occlusion map** for the CLIP model: blur each block of an 8×8 grid in turn and measure the score change.
- **Region explanation with two methods:**
  1. heatmap attention per facial region;
  2. an occlusion test that blurs one region at a time and measures the confidence drop.

  Because every face is aligned, each pixel maps to a named region.
  The "what to look for" hints are general knowledge about AI faces, **not** model output.
- **Limits:** these methods show *where* the evidence is, not *what* it is.
  EfficientNet's Grad-CAM grid is only 7×7.

**Limitations**
- **New generators:** a detector only knows the generators it was trained on.
  The unseen-generator results (DALL-E 3, StyleGAN3) and the two benchmarks show how well it transfers; future generators may still fool it.
  Even the CLIP model misses 40% of StyleGAN3 faces, and the newest generators are thin in the training data
  (FLUX.2 about 200 faces, Imagen 4 90, nano-banana 8).
- **Faces only:** our models judge only the aligned face crop. For images without a clear face, use SwinV2.
- **Face filter:** faces smaller than 128 px were excluded from the data, so very small or low-quality faces are outside what the model learned.
- **Dataset biases:** real photos come from Flickr (FFHQ), celebrity photos (IMDB-WIKI) and LAION/Pexels; other kinds of photos may behave differently.
- **Label noise in OpenFake:** some LAION "real" images are drawings, some faces are in profile, and a few fakes are stylised.
- **Licences:** OpenFake and OpenFakeTiny are CC BY-NC 4.0, so models trained on them are for non-commercial use.
