import torch
import os

NUM_CLIENTS = 5  

for client_id in range(1, NUM_CLIENTS + 1):
    data_dir = f"data/clients_data/client_{client_id}"
    x_path = os.path.join(data_dir, "x_train.pt")
    y_path = os.path.join(data_dir, "y_train.pt")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"[!] Data missing for client_{client_id}")
        continue

    x = torch.load(x_path)
    y = torch.load(y_path)

    print(f"Client {client_id}:")
    print(f"  x_train shape: {x.shape}")
    print(f"  y_train shape: {y.shape}")
    print(f"  Unique labels: {torch.unique(y).tolist()}")
    print(f"  Pixel value range: min={x.min().item():.3f}, max={x.max().item():.3f}")
    print("-" * 40)
