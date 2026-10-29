import csv
import math
import time

import torch
from torch.utils.data import DataLoader, Dataset

from common import OUT, load, mirror, rotate, window
from models_part4 import build


class Windows(Dataset):
    def __init__(self, videos, speed):
        self.videos = videos
        self.speed = speed

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        return window(self.videos[i], 1, self.speed)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def measure(model, videos, device):
    model.eval()
    sums = torch.zeros(8, dtype=torch.float64)
    elapsed = 0.0
    with torch.no_grad():
        for speed in (1, 2, 3):
            loader = DataLoader(Windows(videos, speed), batch_size=32)
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                if speed == 1:
                    sync(device)
                    start = time.perf_counter()
                pred = model(x)
                if speed == 1:
                    sync(device)
                    elapsed += time.perf_counter() - start
                e2 = (pred[:, :2] - y[:, :2]).square().mean((1, 2, 3, 4))
                sums[4 + speed] += e2.sum().item()
                if speed == 1:
                    sums[0] += (pred - y).square().mean((1, 2, 3, 4)).sum().item()
                    for transform, i in ((mirror, 1), (rotate, 3)):
                        transformed = model(transform(x))
                        sums[i] += (transformed - transform(pred)).square().mean((1, 2, 3, 4)).sum().item()
                        sums[i + 1] += (transformed - transform(y)).square().mean((1, 2, 3, 4)).sum().item()
    return (sums / len(videos)).tolist(), elapsed


def fmt(value):
    value = float(value)
    if value == 0:
        return '0'
    if abs(value) < 1e-3:
        return f'{value:.2e}'
    return f'{value:.{max(0, 2 - math.floor(math.log10(abs(value))))}f}'


def pair(values):
    a, b = values
    return f'{fmt((a + b) / 2)} ({fmt(a)}, {fmt(b)})'


if __name__ == '__main__':
    for name in ('a', 'b', 'c', 'd'):
        for seed in (0, 1):
            for prefix, suffix in (('model', 'pt'), ('time', 'txt')):
                path = OUT / f'{prefix}_{name}_{seed}.{suffix}'
                if not path.exists():
                    raise FileNotFoundError(f'Missing {path}. Train the required model first.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hardware = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    _, _, test = load()
    header = ['Model', 'E(5) 1x', 'C_M', 'E_M', 'C_R', 'E_R',
              'E(2) 1x', 'E(2) 2x', 'E(2) 3x', 'Parameters',
              'Training seconds/epoch', 'Inference seconds']
    rows = []
    for name in ('A', 'B', 'C', 'D'):
        values, times, training = [], [], []
        for seed in (0, 1):
            model = build(name).to(device)
            model.load_state_dict(torch.load(OUT / f'model_{name.lower()}_{seed}.pt', map_location=device, weights_only=True))
            metrics, inference = measure(model, test, device)
            values.append(metrics)
            times.append(inference)
            training.append(float((OUT / f'time_{name.lower()}_{seed}.txt').read_text()))
        parameters = sum(p.numel() for p in model.parameters())
        row = [name] + [pair((values[0][i], values[1][i])) for i in range(8)]
        row += [f'{parameters} ({parameters}, {parameters})', pair(training), pair(times)]
        rows.append(row)
        print(f'{name}: complete', flush=True)
    with (OUT / 'table5.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    (OUT / 'table5_hardware.txt').write_text(hardware + '\n')
    print('Hardware:', hardware)
    print('Saved outputs/table5.csv')
