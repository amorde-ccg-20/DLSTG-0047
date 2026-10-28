import torch

from common import OUT, mirror, rotate, window

OUT.mkdir(exist_ok=True)
torch.manual_seed(0)
x = torch.rand(20, 1, 64, 64)
p = torch.zeros(20, 1, 64, 64)
p[7, 0, 10, 20] = 1

pairs = [(mirror(mirror(x)), x), (rotate(x, 4), x),
         (mirror(rotate(mirror(x))), rotate(x, -1))]
values = []
for a, b in pairs:
    diff = (a - b).abs().max().item()
    values.append(f"{diff:g}")

for y in [mirror(p), rotate(p)]:
    index = torch.nonzero(y)[0].tolist()
    position = (index[0], index[2], index[3])
    values.append(str(position))

frames = torch.arange(1, 21).view(20, 1, 1, 1).expand(20, 1, 64, 64)
for k in (1, 2, 3):
    a, b = window(frames, 1, k)
    result = (a[:, 0, 0, 0].tolist(), b[:, 0, 0, 0].tolist())
    values.append(f"{result[0]} / {result[1]}")

labels = ["(a) M(M(x))", "(b) R4(x)", "(c) MRM(x)",
          "(d) M(p) pixel", "(e) R(p) pixel", "(f) 1x frames",
          "(g) 2x frames", "(h) 3x frames"]
for label, value in zip(labels, values):
    print(f"{label}: {value}")
(OUT / "table1.txt").write_text("\n".join(f"{a}: {b}" for a, b in zip(labels, values)) + "\n")
