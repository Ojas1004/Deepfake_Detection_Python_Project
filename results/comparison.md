# Model comparison

All numbers are on the same test set (data/processed, split `test`) and the same unseen-generator sets.

## Overall (test set)

| | SimpleCNN (from scratch) - v2: multi-source + robust | EfficientNet-B0 (fine-tuned) - v2: multi-source + robust | CLIP ViT-L/14 (frozen) + trained head | SimpleCNN (from scratch) - v1: StyleGAN only | EfficientNet-B0 (fine-tuned) - v1: StyleGAN only | EfficientNet-B0 (fine-tuned) [effnet_v3] |
|---|---|---|---|---|---|---|
| Accuracy | 0.759 | 0.935 | 0.989 | 0.610 | 0.621 | 0.951 |
| F1 (fake) | 0.743 | 0.933 | 0.989 | 0.406 | 0.462 | 0.951 |
| ROC-AUC | 0.835 | 0.978 | 0.999 | 0.627 | 0.560 | 0.990 |
| Precision (fake) | 0.797 | 0.964 | 0.991 | 0.852 | 0.795 | 0.958 |
| Recall (fake) | 0.697 | 0.905 | 0.988 | 0.266 | 0.326 | 0.943 |
| Specificity (real) | 0.822 | 0.966 | 0.991 | 0.954 | 0.916 | 0.959 |
| Calibration error (ECE, after) | 0.014 | 0.015 | 0.002 | 0.025 | 0.126 | 0.007 |

## By source / generator (share classified correctly)

| | SimpleCNN (from scratch) - v2: multi-source + robust | EfficientNet-B0 (fine-tuned) - v2: multi-source + robust | CLIP ViT-L/14 (frozen) + trained head | SimpleCNN (from scratch) - v1: StyleGAN only | EfficientNet-B0 (fine-tuned) - v1: StyleGAN only | EfficientNet-B0 (fine-tuned) [effnet_v3] |
|---|---|---|---|---|---|---|
| Real: FFHQ (Flickr) | 82.2% | 98.9% | 99.8% | 94.8% | 96.2% | 98.2% |
| Real: IMDB-WIKI celebrities | 84.8% | 96.2% | 98.7% | 95.5% | 83.4% | 94.6% |
| Real: LAION / Pexels photos | 76.9% | 87.6% | 97.1% | 97.3% | 89.6% | 88.7% |
| StyleGAN (2019) | 66.6% | 97.2% | 99.8% | 86.2% | 99.1% | 96.5% |
| FLUX.1 schnell | 99.4% | 100.0% | 100.0% | 2.4% | 0.0% | 100.0% |
| Stable Diffusion XL | 95.8% | 99.7% | 100.0% | 0.9% | 1.2% | 99.4% |
| FLUX.1 dev | 98.4% | 100.0% | 100.0% | 3.6% | 0.0% | 100.0% |
| FLUX.1 pro | 98.8% | 100.0% | 100.0% | 1.2% | 0.0% | 100.0% |
| Stable Diffusion 1.5 | 80.2% | 95.2% | 98.8% | 3.6% | 11.1% | 94.0% |
| SD inpainting (edit) | 48.6% | 90.4% | 99.1% | 1.5% | 9.3% | 89.8% |
| InsightFace face swap | 33.0% | 92.5% | 96.7% | 2.4% | 10.5% | 93.4% |
| OpenFake: 2022-25 generators | 48.4% | 49.6% | 94.9% | 2.4% | 6.0% | 81.8% |

## Generators never seen in training

Detection rate (share of fakes caught); ROC-AUC against the test-set reals.

| | SimpleCNN (from scratch) - v2: multi-source + robust | EfficientNet-B0 (fine-tuned) - v2: multi-source + robust | CLIP ViT-L/14 (frozen) + trained head | SimpleCNN (from scratch) - v1: StyleGAN only | EfficientNet-B0 (fine-tuned) - v1: StyleGAN only | EfficientNet-B0 (fine-tuned) [effnet_v3] |
|---|---|---|---|---|---|---|
| DALL-E 3 (unseen) | 86.1% (AUC 0.919) | 98.2% (AUC 0.998) | 99.6% (AUC 1.000) | 1.7% (AUC 0.382) | 0.1% (AUC 0.122) | 99.5% (AUC 0.999) |
| StyleGAN3 (unseen) | 44.0% (AUC 0.699) | 9.2% (AUC 0.645) | 60.1% (AUC 0.979) | 14.9% (AUC 0.611) | 20.8% (AUC 0.674) | 14.6% (AUC 0.695) |

## Robustness (test-set accuracy under real-world degradations)

| | SimpleCNN (from scratch) - v2: multi-source + robust | EfficientNet-B0 (fine-tuned) - v2: multi-source + robust | CLIP ViT-L/14 (frozen) + trained head | SimpleCNN (from scratch) - v1: StyleGAN only | EfficientNet-B0 (fine-tuned) - v1: StyleGAN only | EfficientNet-B0 (fine-tuned) [effnet_v3] |
|---|---|---|---|---|---|---|
| clean | 75.9% | 93.6% | 98.9% | 61.0% | 62.1% | 95.1% |
| JPEG q90 | 76.0% | 93.2% | 98.2% | 59.5% | 54.8% | 94.4% |
| JPEG q70 | 75.1% | 92.3% | 97.3% | 58.1% | 49.9% | 93.4% |
| JPEG q50 | 74.2% | 91.6% | 95.9% | 56.7% | 50.3% | 92.5% |
| low-res (50%) | 69.3% | 91.3% | 97.2% | 51.6% | 52.2% | 92.4% |
| blur (r=1) | 72.2% | 92.2% | 97.4% | 53.8% | 56.6% | 93.3% |
| WhatsApp-like | 72.2% | 90.8% | 95.6% | 54.1% | 55.2% | 91.9% |
| zoom in 1.3x | 76.2% | 91.2% | 97.8% | 56.6% | 57.0% | 92.6% |
| zoom out 0.75x | 69.4% | 85.8% | 96.5% | 52.0% | 55.1% | 85.5% |
