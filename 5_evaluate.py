import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        return window(self.videos[i], 1, 1)


def fmt(value):
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    digits = max(0, 2 - math.floor(math.log10(abs(value))))
    return f"{value:.{digits}f}"


def score(model, loader, device):
    errors = torch.zeros(5, dtype=torch.float64)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            errors += (pred - y).square().mean(dim=(2, 3, 4)).sum(dim=0).cpu().double()
    return errors / len(loader.dataset)


def show(model, videos, indices, device):
    fig, axes = plt.subplots(6, 10, figsize=(15, 9))
    with torch.no_grad():
        for row, i in enumerate(indices):
            x, y = window(videos[i], 1, 1)
            pred = model(x.unsqueeze(0).to(device))[0].cpu()
            for col in range(10):
                for offset, future in enumerate((y, pred)):
                    frame = x[col] if col < 5 else future[col - 5]
                    axes[2 * row + offset, col].imshow(frame[0], cmap="gray", vmin=0, vmax=1)
                    axes[2 * row + offset, col].axis("off")
            axes[2 * row, 0].set_ylabel(f"Video {9000 + i}\nTrue")
            axes[2 * row + 1, 0].set_ylabel("Prediction")
            axes[2 * row, 0].text(-0.12, 0.5, f"Video {9000 + i}\nTrue", transform=axes[2 * row, 0].transAxes, ha="right", va="center")
            axes[2 * row + 1, 0].text(-0.12, 0.5, "Prediction", transform=axes[2 * row + 1, 0].transAxes, ha="right", va="center")
    for col in range(10):
        axes[0, col].set_title(f"Input {col + 1}" if col < 5 else f"Future {col - 4}")
    fig.tight_layout()
    fig.savefig(OUT / "figure2.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure2.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs=3, type=int, required=True)
    args = parser.parse_args()
    if any(i < 9000 or i > 9999 for i in args.videos):
        parser.error("Video indices must be between 9000 and 9999.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test = load()
    loader = DataLoader(Windows(test), batch_size=32)
    models, scores = [], []
    for seed in (0, 1):
        model = Model().to(device)
        model.load_state_dict(torch.load(OUT / f"model_a_{seed}.pt", map_location=device, weights_only=True))
        model.eval()
        models.append(model)
        scores.append(score(model, loader, device))
    blank = score(lambda x: torch.zeros_like(x), loader, device)
    copy = score(lambda x: x[:, -1:].expand(-1, 5, -1, -1, -1), loader, device)
    values = torch.stack(scores)
    lines = ["Model | MSE1 | MSE2 | MSE3 | MSE4 | MSE5 | E(5) | Parameters"]
    columns = torch.cat((values, values.mean(dim=1, keepdim=True)), dim=1)
    entries = [f"{fmt(col.mean())} ({fmt(col[0])}, {fmt(col[1])})" for col in columns.T]
    lines.append("A | " + " | ".join(entries) + f" | {sum(p.numel() for p in models[0].parameters())}")
    for name, result in (("Blank", blank), ("Copy-last", copy)):
        entries = [fmt(v) for v in torch.cat((result, result.mean().view(1)))]
        lines.append(name + " | " + " | ".join(entries) + " | 0")
    text = "\n".join(lines) + "\n"
    print(text)
    (OUT / "table2.txt").write_text(text)
    show(models[0], test, [i - 9000 for i in args.videos], device)
