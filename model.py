import torch
from torch import nn
from torch.nn import functional as F


class Cell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels + 64, 256, 3, padding=1)

    def forward(self, x, h, c):
        i, f, o, g = self.conv(torch.cat((x, h), dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = Cell(16)
        self.second = Cell(64)
        self.out = nn.Conv2d(64, 16, 1)

    def forward(self, x):
        batch = x.shape[0]
        h1 = x.new_zeros(batch, 64, 16, 16)
        c1 = x.new_zeros(batch, 64, 16, 16)
        h2 = x.new_zeros(batch, 64, 16, 16)
        c2 = x.new_zeros(batch, 64, 16, 16)
        predictions = []
        for step in range(9):
            frame = x[:, step] if step < 5 else prediction
            h1, c1 = self.first(F.pixel_unshuffle(frame, 4), h1, c1)
            h2, c2 = self.second(h1, h2, c2)
            prediction = torch.sigmoid(F.pixel_shuffle(self.out(h2), 4))
            if step >= 4:
                predictions.append(prediction)
        return torch.stack(predictions, dim=1)
