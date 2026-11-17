"""
check_gpu.py - Step 0: verify that PyTorch can see and use the GPU.

Run this before anything else:
    python check_gpu.py

The RTX 5060 is a Blackwell GPU (compute capability sm_120). Only PyTorch
builds compiled against CUDA 12.8 or newer include kernels for sm_120, so an
older build may import fine but fail as soon as it touches the GPU. That's why
this script also runs a real computation on the GPU, not just
torch.cuda.is_available().
"""

import sys

import torch


def main() -> int:
    print(f"Python        : {sys.version.split()[0]}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Built for CUDA: {torch.version.cuda}")

    # 1) Can PyTorch talk to the NVIDIA driver at all?
    if not torch.cuda.is_available():
        print("\n[FAIL] torch.cuda.is_available() is False.")
        print("       Check the NVIDIA driver (nvidia-smi) and that you installed")
        print("       the CUDA build of torch (see requirements.txt).")
        return 1
    print("\n[OK] torch.cuda.is_available() = True")

    # 2) Which GPU, and does this torch build ship kernels for it?
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    major, minor = torch.cuda.get_device_capability(idx)
    vram_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    print(f"GPU           : {name}")
    print(f"Compute cap.  : sm_{major}{minor}")
    print(f"VRAM          : {vram_gb:.1f} GB")
    print(f"Arch list     : {torch.cuda.get_arch_list()}")

    # 3) Run a real computation on the GPU. With an incompatible build this is
    #    where "no kernel image is available for execution" would show up.
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print(f"[OK] matmul on GPU works (result sum = {c.sum().item():.2f})")

    # 4) Test mixed precision (AMP), which train.py uses.
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        d = a @ b
    print(f"[OK] AMP autocast works (dtype = {d.dtype})")

    # 5) Check cuDNN, which the convolution layers rely on.
    conv = torch.nn.Conv2d(3, 8, kernel_size=3).cuda()
    _ = conv(torch.randn(2, 3, 224, 224, device="cuda"))
    print(f"[OK] cuDNN conv works (cuDNN version {torch.backends.cudnn.version()})")

    print("\nAll GPU checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
