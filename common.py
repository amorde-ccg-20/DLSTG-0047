from pathlib import Path

import torch
from torch.utils.data import Subset
from torchvision.datasets import MovingMNIST

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"


def load(download=False):
    videos = MovingMNIST(root=DATA, split=None, download=download,
                         transform=lambda x: x.to(torch.float32) / 255)
    # The manual split that was asked for
    return (Subset(videos, range(8000)), Subset(videos, range(8000, 9000)),
            Subset(videos, range(9000, 10000)))


def mirror(x):
    return x.flip(-1)


def rotate(x, k=1):
    return torch.rot90(x, k, (-2, -1))


def window(x, t, k):
    frames = x[t - 1::k][:10]
    return frames[:5], frames[5:]
