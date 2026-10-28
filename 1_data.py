from common import DATA, load

train, val, test = load(download=True)
x = train[0]
print(f"Data folder: {DATA}")
print(f"Videos: train={len(train)}, validation={len(val)}, test={len(test)}")
print(f"First training video: shape={tuple(x.shape)}, dtype={x.dtype}")
print(f"Intensity range: {x.min().item():.3f} to {x.max().item():.3f}")
