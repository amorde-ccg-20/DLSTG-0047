import time

import torch
from torch.utils.data import DataLoader

from common import OUT, load
from models_part4 import Windows, build, loss


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}", flush=True)
    train, _, _ = load()
    for name in ('B', 'C', 'D'):
        for seed in (0, 1):
            torch.manual_seed(seed)
            model = build(name).to(device)
            print(f'{name}, seed {seed}, parameters: {sum(p.numel() for p in model.parameters())}', flush=True)
            loader = DataLoader(Windows(train, name), batch_size=32, shuffle=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            start = time.perf_counter()
            for epoch in range(30):
                model.train()
                total = 0.0
                for x, y, mask in loader:
                    x, y, mask = x.to(device), y.to(device), mask.to(device)
                    optimizer.zero_grad()
                    error = loss(model(x), y, mask)
                    error.backward()
                    optimizer.step()
                    total += error.item() * len(x)
                print(f'{name}, seed {seed}, epoch {epoch + 1}/30: {total / len(train):.3g}', flush=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            seconds = (time.perf_counter() - start) / 30
            torch.save(model.state_dict(), OUT / f'model_{name.lower()}_{seed}.pt')
            (OUT / f'time_{name.lower()}_{seed}.txt').write_text(f'{seconds:.6f}\n')
            print(f'Mean training time: {seconds:.2f} seconds/epoch', flush=True)
