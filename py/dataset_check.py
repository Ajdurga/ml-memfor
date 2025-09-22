from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset, random_split
from torchvision import datasets

ROOT = Path("data")
SEED = 1337


def main() -> None:
    train_raw = datasets.CIFAR10(root=ROOT, train=True, download=True)
    test_raw = datasets.CIFAR10(root=ROOT, train=False, download=True)

    full = ConcatDataset([train_raw, test_raw])
    N = len(full)
    sizes = [int(0.8 * N), int(0.1 * N)]
    sizes.append(N - sum(sizes))

    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(full, sizes, generator=gen)

    print("total", N)
    print("split sizes", [len(train_ds), len(val_ds), len(test_ds)])
    sample0 = full[0]
    if isinstance(sample, tuple) and len(sample) == 2:
        x0, y0 = sample0
        print("image types:", type(x0).__name__, "label type:", type(y0).__name__)
    else:
        print("unexpected sample repr:", repr(sample)[:200])

if __name__ == "__main__":
    main()