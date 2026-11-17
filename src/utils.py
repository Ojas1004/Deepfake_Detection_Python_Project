"""
utils.py - Small helpers shared across scripts (reproducibility, device).
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every random number generator we use so runs are repeatable.

    - random / numpy / torch: control shuffling, augmentation, weight init.
    - cudnn.benchmark = False: stops cuDNN from picking different (and
      non-deterministic) convolution algorithms on each run.
    - cudnn.deterministic = True: asks cuDNN to use deterministic algorithms.

    Note: on a GPU some operations can still differ by tiny floating-point
    amounts between runs. Results will be very close but not always
    bit-identical, which is normal and expected.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker process its own, reproducible seed.

    PyTorch seeds torch's RNG in each worker, but not numpy's or Python's
    `random`. Deriving them from torch's per-worker seed keeps any
    augmentation that uses them reproducible too.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    """Return the GPU if available, otherwise the CPU (with a warning)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: CUDA not available - running on CPU (this will be slow).")
    return torch.device("cpu")
