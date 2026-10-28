import time

import torch
from torch.utils.data import DataLoader, Dataset

from common import OUT, load, window
from model import Model


class Windows(Dataset):
    def __init__(self, videos):
        self.videos = videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        t = torch.randint(1, 12, ()).item()
        return window(self.videos[i], t, 1)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _, _ = load()
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}", flush=True)
    for seed in (0, 1):
        torch.manual_seed(seed)
        model = Model().to(device)
        print(f"Seed {seed}; trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}", flush=True)
        loader = DataLoader(Windows(train), batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        start = time.perf_counter()
        for epoch in range(30):
            model.train()
            total = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = (model(x) - y).square().mean()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(x)
            print(f"Seed {seed}, epoch {epoch + 1}/30, loss: {total / len(train):.3g}", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 30
        torch.save(model.state_dict(), OUT / f"model_a_{seed}.pt")
        (OUT / f"time_a_{seed}.txt").write_text(f"{elapsed:.6f}\n")
        print(f"Mean training time: {elapsed:.2f} seconds/epoch", flush=True)
