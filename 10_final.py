import csv
import importlib.util

import torch
from torch import nn

from common import OUT, load, mirror, rotate
from models_part4 import build


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Symmetric(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        total = None
        for flipped in (False, True):
            for k in range(4):
                transformed = rotate(x, k)
                if flipped:
                    transformed = mirror(transformed)
                prediction = self.model(transformed)
                if flipped:
                    prediction = mirror(prediction)
                prediction = rotate(prediction, -k)
                total = prediction if total is None else total + prediction
        return total / 8


def gap_pair(values):
    a, b = values
    return f'{(a + b) / 2:.2e} ({a:.2e}, {b:.2e})'


def selection():
    paths = (OUT / 'figure3_videos.txt', OUT / 'figure5_video.txt')
    if not all(path.exists() for path in paths):
        raise FileNotFoundError('Run the Part 3 scripts first to record the Figure 3 and Figure 5 video indices.')
    chosen = []
    for line in paths[0].read_text().splitlines():
        name, number = line.split()
        chosen.append((name, int(number)))
    speed_video = int(paths[1].read_text().strip())
    if len(chosen) != 4 or [x[0] for x in chosen].count('Mirror') != 2 or [x[0] for x in chosen].count('Rotation') != 2:
        raise ValueError('Figure 3 requires two mirror and two rotation videos.')
    if any(index < 9000 or index > 9999 for _, index in chosen) or not 9000 <= speed_video <= 9999:
        raise ValueError('Figure indices must be test videos 9000–9999.')
    return chosen, speed_video


if __name__ == '__main__':
    for name in ('b', 'd'):
        for seed in (0, 1):
            for prefix, suffix in (('model', 'pt'), ('time', 'txt')):
                path = OUT / f'{prefix}_{name}_{seed}.{suffix}'
                if not path.exists():
                    raise FileNotFoundError(f'Missing {path}. Run Part 4 training first.')
    chosen, speed_video = selection()
    table = module('table5', '9_table5.py')
    symmetry = module('symmetry', '6_symmetry.py')
    speed = module('speed', '7_speed.py')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hardware = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    _, _, test = load()
    header = ['Model', 'E(5) 1x', 'C_M', 'E_M', 'C_R', 'E_R',
              'E(2) 1x', 'E(2) 2x', 'E(2) 3x', 'Parameters',
              'Training seconds/epoch', 'Inference seconds']
    rows = []
    final = None
    for name in ('B', 'D'):
        values, inference, training = [], [], []
        for seed in (0, 1):
            model = build(name).to(device)
            model.load_state_dict(torch.load(OUT / f'model_{name.lower()}_{seed}.pt', map_location=device, weights_only=True))
            wrapped = Symmetric(model).to(device).eval()
            metrics, seconds = table.measure(wrapped, test, device)
            values.append(metrics)
            inference.append(seconds)
            training.append(float((OUT / f'time_{name.lower()}_{seed}.txt').read_text()))
            if name == 'D' and seed == 0:
                final = wrapped
        columns = []
        for i in range(8):
            v = (values[0][i], values[1][i])
            columns.append(gap_pair(v) if i in (1, 3) else table.pair(v))
        parameters = sum(p.numel() for p in model.parameters())
        rows.append([f'Symmetrised {name}'] + columns +
                    [f'{parameters} ({parameters}, {parameters})', table.pair(training), table.pair(inference)])
        print(f'Symmetrised {name}: complete', flush=True)
    with (OUT / 'table6.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    (OUT / 'table6_hardware.txt').write_text(hardware + '\n')
    for name, index in chosen:
        transform = mirror if name == 'Mirror' else rotate
        path = symmetry.panel(final, test[index - 9000], index, name, transform, device, prefix='figure6')
        print(f'{name}: video {index}, GIF {path}', flush=True)
    speed.show(final, test[speed_video - 9000], speed_video, device, prefix='figure6_speed')
    print(f'Saved Table 6 and Figure 6. Final model: symmetrised D; speed video: {speed_video}.')
