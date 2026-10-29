import torch
from torch import nn
from torch.utils.data import Dataset

from common import mirror, rotate, window
from model import Model


def build(name):
    model = Model()
    if name == 'D':
        model.first.conv = nn.Conv2d(80, 256, 3, padding=2, dilation=2)
        model.second.conv = nn.Conv2d(128, 256, 3, padding=2, dilation=2)
    return model


class Windows(Dataset):
    def __init__(self, videos, name):
        self.videos = videos
        self.name = name

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        if self.name == 'B':
            speed = 1
            start = torch.randint(1, 12, ()).item()
        else:
            speed = torch.randint(1, 4, ()).item()
            start = torch.randint(1, (14, 8, 2)[speed - 1] + 1, ()).item()
        x, y = window(self.videos[i], start, speed)
        if self.name == 'B':
            g = torch.randint(0, 8, ()).item()
            x, y = rotate(x, g % 4), rotate(y, g % 4)
            if g >= 4:
                x, y = mirror(x), mirror(y)
        count = len(y)
        target = torch.zeros_like(x)
        target[:count] = y
        return x, target, torch.arange(5) < count


def loss(pred, target, mask):
    errors = (pred - target).square().mean(dim=(2, 3, 4))
    return ((errors * mask).sum(dim=1) / mask.sum(dim=1)).mean()
