# Frontier benchmark (19 current generators, public split)

Fakes from GPT-Image-2, Nano Banana Pro/2.1, FLUX.3, FLUX.2, Seedream 5, Qwen-Image 3, Ideogram 4.5, Grok Imagine 2 and more, vs real photos; each also as WhatsApp / Instagram / re-shared copies. 'Detected @ 1% FP' = share of fakes caught when only 1% of real photos may be flagged. Our models are scored on face images only.

## All images

| Detector | Variant | ROC-AUC | Detected @ 1% FP | Fakes caught (0.5) | Reals kept (0.5) | n fake / real |
|---|---|---|---|---|---|---|
| [external] CommunityForensics ViT (CVPR 2025) | original | 0.781 | 25.3% | 36.0% | 96.3% | 261 / 486 |
| [external] CommunityForensics ViT (CVPR 2025) | whatsapp | 0.707 | 19.9% | 10.3% | 99.8% | 261 / 482 |
| [external] CommunityForensics ViT (CVPR 2025) | instagram | 0.700 | 21.8% | 16.1% | 99.6% | 261 / 482 |
| [external] CommunityForensics ViT (CVPR 2025) | reshared | 0.702 | 20.3% | 12.3% | 99.8% | 261 / 482 |
| [external] SigLIP AI-vs-human (Ateeqq) | original | 0.805 | 13.8% | 59.8% | 84.0% | 261 / 486 |
| [external] SigLIP AI-vs-human (Ateeqq) | whatsapp | 0.808 | 14.6% | 60.9% | 82.8% | 261 / 482 |
| [external] SigLIP AI-vs-human (Ateeqq) | instagram | 0.802 | 14.2% | 60.2% | 81.5% | 261 / 482 |
| [external] SigLIP AI-vs-human (Ateeqq) | reshared | 0.803 | 12.6% | 59.8% | 82.0% | 261 / 482 |
| [external] SwinV2 AI detector (haywoodsloan) | original | 0.926 | 64.4% | 68.2% | 98.4% | 261 / 486 |
| [external] SwinV2 AI detector (haywoodsloan) | whatsapp | 0.903 | 42.1% | 37.2% | 99.2% | 261 / 482 |
| [external] SwinV2 AI detector (haywoodsloan) | instagram | 0.894 | 39.5% | 45.2% | 98.8% | 261 / 482 |
| [external] SwinV2 AI detector (haywoodsloan) | reshared | 0.894 | 38.7% | 39.8% | 98.5% | 261 / 482 |

## Only images with a face

| Detector | Variant | ROC-AUC | Detected @ 1% FP | Fakes caught (0.5) | Reals kept (0.5) | n fake / real |
|---|---|---|---|---|---|---|
| [ours] SimpleCNN (from scratch) - v2: multi-source + robust | original | 0.720 | 15.9% | 39.8% | 92.2% | 88 / 51 |
| [ours] SimpleCNN (from scratch) - v2: multi-source + robust | whatsapp | 0.717 | 20.5% | 43.2% | 86.3% | 88 / 51 |
| [ours] SimpleCNN (from scratch) - v2: multi-source + robust | instagram | 0.645 | 9.1% | 40.9% | 76.5% | 88 / 51 |
| [ours] SimpleCNN (from scratch) - v2: multi-source + robust | reshared | 0.677 | 20.5% | 43.2% | 82.4% | 88 / 51 |
| [ours] EfficientNet-B0 (fine-tuned) - v2: multi-source + robust | original | 0.689 | 8.0% | 31.8% | 86.3% | 88 / 51 |
| [ours] EfficientNet-B0 (fine-tuned) - v2: multi-source + robust | whatsapp | 0.594 | 3.4% | 37.5% | 70.6% | 88 / 51 |
| [ours] EfficientNet-B0 (fine-tuned) - v2: multi-source + robust | instagram | 0.539 | 3.4% | 35.2% | 62.7% | 88 / 51 |
| [ours] EfficientNet-B0 (fine-tuned) - v2: multi-source + robust | reshared | 0.537 | 1.1% | 37.5% | 58.8% | 88 / 51 |
| [ours] CLIP ViT-L/14 (frozen) + trained head | original | 0.931 | 62.5% | 86.4% | 84.3% | 88 / 51 |
| [ours] CLIP ViT-L/14 (frozen) + trained head | whatsapp | 0.831 | 37.5% | 46.6% | 96.1% | 88 / 51 |
| [ours] CLIP ViT-L/14 (frozen) + trained head | instagram | 0.897 | 58.0% | 61.4% | 96.1% | 88 / 51 |
| [ours] CLIP ViT-L/14 (frozen) + trained head | reshared | 0.870 | 52.3% | 58.0% | 96.1% | 88 / 51 |
| [ours] SimpleCNN (from scratch) - v1: StyleGAN only | original | 0.599 | 10.2% | 9.1% | 100.0% | 88 / 51 |
| [ours] SimpleCNN (from scratch) - v1: StyleGAN only | whatsapp | 0.632 | 12.5% | 11.4% | 98.0% | 88 / 51 |
| [ours] SimpleCNN (from scratch) - v1: StyleGAN only | instagram | 0.584 | 10.2% | 10.2% | 100.0% | 88 / 51 |
| [ours] SimpleCNN (from scratch) - v1: StyleGAN only | reshared | 0.578 | 15.9% | 12.5% | 100.0% | 88 / 51 |
| [ours] EfficientNet-B0 (fine-tuned) - v1: StyleGAN only | original | 0.609 | 14.8% | 12.5% | 100.0% | 88 / 51 |
| [ours] EfficientNet-B0 (fine-tuned) - v1: StyleGAN only | whatsapp | 0.611 | 17.0% | 27.3% | 92.2% | 88 / 51 |
| [ours] EfficientNet-B0 (fine-tuned) - v1: StyleGAN only | instagram | 0.631 | 14.8% | 21.6% | 96.1% | 88 / 51 |
| [ours] EfficientNet-B0 (fine-tuned) - v1: StyleGAN only | reshared | 0.600 | 23.9% | 23.9% | 98.0% | 88 / 51 |
| [ours] EfficientNet-B0 (fine-tuned) [effnet_v3] | original | 0.770 | 25.0% | 63.6% | 76.5% | 88 / 51 |
| [ours] EfficientNet-B0 (fine-tuned) [effnet_v3] | whatsapp | 0.740 | 14.8% | 56.8% | 80.4% | 88 / 51 |
| [ours] EfficientNet-B0 (fine-tuned) [effnet_v3] | instagram | 0.697 | 10.2% | 60.2% | 64.7% | 88 / 51 |
| [ours] EfficientNet-B0 (fine-tuned) [effnet_v3] | reshared | 0.704 | 10.2% | 60.2% | 70.6% | 88 / 51 |
| [external] CommunityForensics ViT (CVPR 2025) | original | 0.909 | 53.4% | 44.3% | 100.0% | 88 / 51 |
| [external] CommunityForensics ViT (CVPR 2025) | whatsapp | 0.869 | 38.6% | 13.6% | 100.0% | 88 / 51 |
| [external] CommunityForensics ViT (CVPR 2025) | instagram | 0.861 | 35.2% | 20.5% | 100.0% | 88 / 51 |
| [external] CommunityForensics ViT (CVPR 2025) | reshared | 0.865 | 43.2% | 17.0% | 100.0% | 88 / 51 |
| [external] SigLIP AI-vs-human (Ateeqq) | original | 0.772 | 11.4% | 73.9% | 68.6% | 88 / 51 |
| [external] SigLIP AI-vs-human (Ateeqq) | whatsapp | 0.785 | 11.4% | 75.0% | 70.6% | 88 / 51 |
| [external] SigLIP AI-vs-human (Ateeqq) | instagram | 0.773 | 11.4% | 73.9% | 64.7% | 88 / 51 |
| [external] SigLIP AI-vs-human (Ateeqq) | reshared | 0.770 | 11.4% | 75.0% | 66.7% | 88 / 51 |
| [external] SwinV2 AI detector (haywoodsloan) | original | 0.904 | 9.1% | 78.4% | 94.1% | 88 / 51 |
| [external] SwinV2 AI detector (haywoodsloan) | whatsapp | 0.890 | 10.2% | 52.3% | 96.1% | 88 / 51 |
| [external] SwinV2 AI detector (haywoodsloan) | instagram | 0.897 | 8.0% | 60.2% | 96.1% | 88 / 51 |
| [external] SwinV2 AI detector (haywoodsloan) | reshared | 0.887 | 14.8% | 58.0% | 94.1% | 88 / 51 |

## Fakes caught per generator (originals, threshold 0.5)

| Generator | [external] CommunityForensics ViT (CVPR 2025) | [external] SigLIP AI-vs-human (Ateeqq) | [external] SwinV2 AI detector (haywoodsloan) |
|---|---|---|---|
| alibaba/qwen-image-3 | 33% | 80% | 73% |
| blackforestlabs/flux-3 | 9% | 55% | 55% |
| bytedance/seedream/v5/pro | 0% | 71% | 14% |
| flux-2 | 92% | 69% | 92% |
| flux-2-pro | 18% | 71% | 53% |
| google/nano-banana-2.1 | 43% | 36% | 79% |
| hidream-o1-image | 100% | 75% | 94% |
| ideogram/v4.5 | 35% | 65% | 53% |
| kling-image/v3 | 60% | 90% | 90% |
| krea-2/turbo | 81% | 75% | 94% |
| luma/agent/uni-1/v1 | 23% | 38% | 54% |
| meta/muse-image | 0% | 58% | 58% |
| microsoft/mai-image-2.5 | 25% | 83% | 92% |
| nano-banana-pro | 15% | 30% | 65% |
| openai/gpt-image-2 | 42% | 50% | 67% |
| qwen-image-2512 | 60% | 40% | 100% |
| recraft/v4.1 | 0% | 38% | 75% |
| xai/grok-imagine-image/v2.0 | 0% | 67% | 42% |
| z-image/turbo | 55% | 55% | 55% |
