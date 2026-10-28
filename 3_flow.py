import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUT, load

OUT.mkdir(exist_ok=True)
train, _, _ = load()
directions, speeds = [], []
for i in range(50):
    video = train[i].squeeze(1).numpy()
    for s in range(19):
        a = (video[s] * 255).astype(np.uint8)
        b = (video[s + 1] * 255).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx, dy = flow[..., 0], flow[..., 1]
        speed = np.hypot(dx, dy)
        keep = (video[s] > 0.2) & (video[s + 1] > 0.2) & (speed > 0.5)
        directions.append(np.degrees(np.arctan2(-dy[keep], dx[keep])) % 360)
        speeds.append(speed[keep])

direction = np.concatenate(directions)
speed = np.concatenate(speeds)
median = np.median(speed)
weights = np.full(len(speed), 1 / len(speed))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(direction, bins=np.arange(0, 361, 20), weights=weights, edgecolor="white")
axes[0].axhline(1 / 18, color="red", linestyle="--", label="Uniform: 1/18")
axes[0].set(xlabel="Direction (degrees)", ylabel="Fraction of kept pixels", xlim=(0, 360), title="(a) Flow direction")
axes[0].legend()
axes[1].hist(speed, bins=40, weights=weights, edgecolor="white")
axes[1].axvline(median, color="red", linestyle="--", label=f"Median: {median:.3g} pixels/frame")
axes[1].set(xlabel="Speed (pixels/frame)", ylabel="Fraction of kept pixels", title="(b) Flow speed")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT / "figure1.pdf", bbox_inches="tight")
fig.savefig(OUT / "figure1.png", dpi=160, bbox_inches="tight")
print(f"Kept pixels: {len(speed)}; median speed: {median:.3g} pixels/frame")
(OUT / "flow.txt").write_text(f"Kept pixels: {len(speed)}\nMedian speed: {median:.3g} pixels/frame\n")
