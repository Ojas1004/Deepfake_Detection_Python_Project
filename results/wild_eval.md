# In-the-wild evaluation (OpenFakeTiny, faces only)

850 images: 65 real, 785 fake. Balanced accuracy = average of fake and real accuracy (50% = guessing). Only 65 real photos, so 'reals kept as real' is approximate.

| Detector | Modern-generator fakes caught | Reddit fakes caught | Reals kept as real | Balanced accuracy | ROC-AUC |
|---|---|---|---|---|---|
| [ours] CLIP ViT-L/14 (frozen) + trained head | 79.1% | 77.3% | 87.7% | 82.8% | 0.925 |
| [external] SwinV2 AI detector (haywoodsloan) | 60.9% | 71.7% | 87.7% | 77.7% | 0.861 |
| [external] SigLIP AI-vs-human (Ateeqq) | 79.8% | 92.4% | 46.2% | 66.9% | 0.824 |
| [ours] EfficientNet-B0 (fine-tuned) [effnet_v3] | 52.2% | 67.2% | 83.1% | 72.3% | 0.791 |
| [external] CommunityForensics ViT (CVPR 2025) | 22.6% | 24.6% | 95.4% | 59.6% | 0.771 |
| [ours] EfficientNet-B0 (fine-tuned) - v2: multi-source + robust | 32.0% | 57.2% | 75.4% | 61.5% | 0.717 |
| [ours] SimpleCNN (from scratch) - v2: multi-source + robust | 50.5% | 54.7% | 69.2% | 61.2% | 0.662 |
| [ours] EfficientNet-B0 (fine-tuned) - v1: StyleGAN only | 3.4% | 7.4% | 90.8% | 48.3% | 0.525 |
| [external] ViT deepfake detector (prithivMLmods) | 97.6% | 96.9% | 0.0% | 48.6% | 0.479 |
| [ours] SimpleCNN (from scratch) - v1: StyleGAN only | 1.3% | 0.8% | 100.0% | 50.5% | 0.426 |
