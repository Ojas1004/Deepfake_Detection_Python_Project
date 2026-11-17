# Folder test: tpdne

2000 images from `data/raw/tpdne/images`, all fake.

| Detector | Fakes detected | Median P(fake) |
|---|---|---|
| [external] ViT deepfake detector (prithivMLmods) | 63.2% | 0.615 |
| [ours] EfficientNet-B0 (fine-tuned) - v1: StyleGAN only | 63.1% | 0.837 |
| [external] CommunityForensics ViT (CVPR 2025) | 63.1% | 0.787 |
| [external] SigLIP AI-vs-human (Ateeqq) | 16.6% | 0.002 |
| [external] SwinV2 AI detector (haywoodsloan) | 16.0% | 0.036 |
| [ours] SimpleCNN (from scratch) - v1: StyleGAN only | 10.9% | 0.054 |
