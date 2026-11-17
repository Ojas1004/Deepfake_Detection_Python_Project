# In-the-wild evaluation (OpenFakeTiny, faces only)

850 images: 65 real, 785 fake. Balanced accuracy = average of fake and real accuracy (50% = guessing). Only 65 real photos, so 'reals kept as real' is approximate.

| Detector | Modern-generator fakes caught | Reddit fakes caught | Reals kept as real | Balanced accuracy | ROC-AUC |
|---|---|---|---|---|---|
| [external] SwinV2 AI detector (haywoodsloan) | 60.9% | 71.7% | 87.7% | 77.7% | 0.861 |
| [external] SigLIP AI-vs-human (Ateeqq) | 79.8% | 92.4% | 46.2% | 66.9% | 0.824 |
| [external] CommunityForensics ViT (CVPR 2025) | 22.6% | 24.6% | 95.4% | 59.6% | 0.771 |
| [ours] EfficientNet-B0 (fine-tuned) - v1: StyleGAN only | 3.4% | 7.4% | 90.8% | 48.3% | 0.525 |
| [external] ViT deepfake detector (prithivMLmods) | 97.6% | 96.9% | 0.0% | 48.6% | 0.479 |
| [ours] SimpleCNN (from scratch) - v1: StyleGAN only | 1.3% | 0.8% | 100.0% | 50.5% | 0.426 |

## Your own images: P(fake) per detector

| Detector | 7.png |
|---|---|
| [ours] SimpleCNN (from scratch) - v1: StyleGAN only | 0.099 |
| [ours] EfficientNet-B0 (fine-tuned) - v1: StyleGAN only | 0.000 |
| [external] CommunityForensics ViT (CVPR 2025) | 0.000 |
| [external] SigLIP AI-vs-human (Ateeqq) | 1.000 |
| [external] SwinV2 AI detector (haywoodsloan) | 0.049 |
| [external] ViT deepfake detector (prithivMLmods) | 0.802 |
