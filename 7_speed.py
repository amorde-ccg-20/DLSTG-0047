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
    def __init__(self, videos, speed):
        self.videos = videos
        self.speed = speed

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        return window(self.videos[i], 1, self.speed)


def fmt(value):
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.{max(0, 2 - math.floor(math.log10(abs(value))))}f}"


def score(model, loader, device):
    errors = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)[:, :y.shape[1]]
            errors.append((pred - y).square().mean((2, 3, 4)).cpu())
    errors = torch.cat(errors)
    return errors[:, :2].mean().item(), errors.mean().item()


def show(model, video, index, device):
    fig, axes = plt.subplots(6, 7, figsize=(11, 9))
    with torch.no_grad():
        for row, speed in enumerate((1, 2, 3)):
            x, y = window(video, 1, speed)
            pred = model(x.unsqueeze(0).to(device))[0].cpu()
            for offset, future in enumerate((y[:2], pred[:2])):
                for col, frame in enumerate(torch.cat((x, future))):
                    ax = axes[2 * row + offset, col]
                    ax.imshow(frame[0], cmap="gray", vmin=0, vmax=1)
                    ax.axis("off")
                axes[2 * row + offset, 0].text(-0.1, 0.5, f"{speed}x {'true' if offset == 0 else 'predicted'}", transform=axes[2 * row + offset, 0].transAxes, ha="right", va="center")
    for col in range(7):
        axes[0, col].set_title(f"Input {col + 1}" if col < 5 else f"Future {col - 4}")
    fig.suptitle(f"Test video {index}")
    fig.tight_layout()
    fig.savefig(OUT / "figure5.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure5.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=int, required=True)
    args = parser.parse_args()
    if args.video < 9000 or args.video > 9999:
        parser.error("The test video index must be between 9000 and 9999.")
    OUT.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test = load()
    models = []
    for seed in (0, 1):
        model = Model().to(device)
        model.load_state_dict(torch.load(OUT / f"model_a_{seed}.pt", map_location=device, weights_only=True))
        model.eval()
        models.append(model)
    copy = lambda x: x[:, -1:].expand(-1, 5, -1, -1, -1)
    values, references = [], []
    for speed in (1, 2, 3):
        loader = DataLoader(Windows(test, speed), batch_size=32)
        values.append([score(model, loader, device) for model in models])
        references.append(score(copy, loader, device))
    lines = ["Model | E(2) 1x | E(2) 2x | E(2) 3x | E(5) 1x | E(5) 2x"]
    columns = [(values[k][0][0], values[k][1][0]) for k in range(3)]
    columns += [(values[k][0][1], values[k][1][1]) for k in range(2)]
    lines.append("A | " + " | ".join(f"{fmt((a+b)/2)} ({fmt(a)}, {fmt(b)})" for a, b in columns))
    values = [references[k][0] for k in range(3)] + [references[k][1] for k in range(2)]
    lines.append("Copy-last | " + " | ".join(fmt(x) for x in values))
    text = "\n".join(lines) + "\n"
    print(text)
    (OUT / "table4.txt").write_text(text)
    show(models[0], test[args.video - 9000], args.video, device)
    (OUT / "figure5_video.txt").write_text(f"{args.video}\n")
