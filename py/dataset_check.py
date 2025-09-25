from pathlib import Path
import torch
from torch.utils.data import ConcatDataset, random_split
from torchvision import datasets
ROOT = Path("data")
SEED = 1337

train_raw = datasets.CIFAR10(root=ROOT, train=True, download=True)
test_raw = datasets.CIFAR10(root=ROOT, train=False, download=True)

full = ConcatDataset([train_raw, test_raw])
N = len(full)
sizes = [int(0.8*N), int(0.1*N)]
sizes.append(N-sum(sizes))

gen = torch.Generator().manual_seed(SEED)
train_ds, val_ds, test_ds = random_split(full, sizes, generator=gen)

print("total", N)
print("split sizes", [len(train_ds), len(val_ds), len(test_ds)])