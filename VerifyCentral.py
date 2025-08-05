import torch
import os

SAVE_DIR = "data/central_testset"

x_path = os.path.join(SAVE_DIR, "x.pt")
y_path = os.path.join(SAVE_DIR, "y.pt")
indices_path = os.path.join(SAVE_DIR, "indices.pt")

if not (os.path.exists(x_path) and os.path.exists(y_path)):
    print("Central test set files not found.")
    exit(1)

x = torch.load(x_path)
y = torch.load(y_path)
indices = torch.load(indices_path) if os.path.exists(indices_path) else None

print(f"Central test set:")
print(f"  x shape: {x.shape}")
print(f"  y shape: {y.shape}")
print(f"  Unique labels: {torch.unique(y).tolist()}")
print(f"  Pixel value range: min={x.min().item():.3f}, max={x.max().item():.3f}")
if indices is not None:
    print(f"  Indices length: {len(indices)}")
print("-" * 40)