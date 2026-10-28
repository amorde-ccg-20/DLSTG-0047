import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from common import OUT, load, mirror, rotate, window
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
    return f"{value:.{max(0, 2 - math.floor(math.log10(abs(value))))}f}"


def measure(model, loader, device):
    results = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            columns = [(pred - y).square().mean((1, 2, 3, 4))]
            for transform in (mirror, rotate):
                transformed = model(transform(x))
                columns.append((transformed - transform(pred)).square().mean((1, 2, 3, 4)))
                columns.append((transformed - transform(y)).square().mean((1, 2, 3, 4)))
            results.append(torch.stack(columns, dim=1).cpu())
    return torch.cat(results)


def rows(model, video, transform, device):
    x, y = window(video, 1, 1)
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device))[0].cpu()
        transformed = model(transform(x).unsqueeze(0).to(device))[0].cpu()
    tx = transform(x)
    return [(x, pred), (tx, transform(y)), (tx, transformed), (tx, transform(pred))]


def panel(model, video, index, name, transform, device):
    pairs = rows(model, video, transform, device)
    labels = ["X, f(X)", "T(X), T(Y)", "T(X), f(T(X))", "T(X), T(f(X))"]
    fig, axes = plt.subplots(4, 10, figsize=(15, 6))
    for row, (x, y) in enumerate(pairs):
        for col, frame in enumerate(torch.cat((x, y))):
            axes[row, col].imshow(frame[0], cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
        axes[row, 0].text(-0.1, 0.5, labels[row], transform=axes[row, 0].transAxes, ha="right", va="center")
    fig.suptitle(f"{name}, test video {index}")
    fig.tight_layout()
    fig.savefig(OUT / f"figure3_{name}_{index}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"figure3_{name}_{index}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    frames = []
    for h in range(5):
        fig, axes = plt.subplots(4, 6, figsize=(9, 6))
        for row, (x, y) in enumerate(pairs):
            for col, frame in enumerate(torch.cat((x, y[h:h + 1]))):
                axes[row, col].imshow(frame[0], cmap="gray", vmin=0, vmax=1)
                axes[row, col].axis("off")
            axes[row, 0].text(-0.1, 0.5, labels[row], transform=axes[row, 0].transAxes, ha="right", va="center")
        fig.suptitle(f"{name}, video {index}, horizon {h + 1}")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB"))
        plt.close(fig)
    path = OUT / f"figure3_{name}_{index}.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=800, loop=0)
    return path


def hist(values, chosen):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, name in zip(axes, (1, 3), ("Mirror", "Rotation")):
        ax.hist(values[:, col].numpy(), bins=30, edgecolor="white")
        for index, color in zip(chosen[name], ("red", "darkorange")):
            ax.axvline(values[index, col], color=color, linestyle="--", label=f"Video {9000 + index}")
        ax.set(xlabel="Per-video consistency gap", ylabel="Videos", title=name)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figure4.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure4.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test = load()
    loader = DataLoader(Windows(test), batch_size=32)
    models, values = [], []
    for seed in (0, 1):
        model = Model().to(device)
        model.load_state_dict(torch.load(OUT / f"model_a_{seed}.pt", map_location=device, weights_only=True))
        model.eval()
        models.append(model)
        values.append(measure(model, loader, device))
    blank = measure(lambda x: torch.zeros_like(x), loader, device)
    copy = measure(lambda x: x[:, -1:].expand(-1, 5, -1, -1, -1), loader, device)
    means = torch.stack([x.mean(dim=0) for x in values])
    lines = ["Model | E(5) | C_M | E_M | C_R | E_R"]
    entries = [f"{fmt(col.mean())} ({fmt(col[0])}, {fmt(col[1])})" for col in means.T]
    lines.append("A | " + " | ".join(entries))
    for name, result in (("Blank", blank), ("Copy-last", copy)):
        lines.append(name + " | " + " | ".join(fmt(v) for v in result.mean(dim=0)))
    text = "\n".join(lines) + "\n"
    print(text)
    (OUT / "table3.txt").write_text(text)
    chosen = {"Mirror": values[0][:, 1].topk(2).indices.tolist(),
              "Rotation": values[0][:, 3].topk(2).indices.tolist()}
    (OUT / "figure3_videos.txt").write_text("\n".join(f"{name} {9000 + i}" for name in ("Mirror", "Rotation") for i in chosen[name]) + "\n")
    for name, transform in (("Mirror", mirror), ("Rotation", rotate)):
        for i in chosen[name]:
            path = panel(models[0], test[i], 9000 + i, name, transform, device)
            print(f"{name}: video {9000 + i}, GIF {path}")
    hist(values[0], chosen)
