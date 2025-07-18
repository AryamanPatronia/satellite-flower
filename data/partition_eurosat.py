import torch
import os
from torchvision.datasets import EuroSAT
from torchvision.transforms import ToTensor
from torch.utils.data import Subset
import random
from collections import defaultdict

NUM_CLIENTS = 5
SAMPLES_PER_CLIENT = 1000 

def partition_dataset():
    dataset = EuroSAT(root="data/raw", download=False, transform=ToTensor())

    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_to_indices[label].append(idx)

    os.makedirs("data/clients_data", exist_ok=True)

    # Assign 2 classes per client (non-IID setup)
    class_ids = list(class_to_indices.keys())
    random.seed(42)
    random.shuffle(class_ids)

    for client_id in range(NUM_CLIENTS):
        client_classes = class_ids[client_id::NUM_CLIENTS]  # non-overlapping assignment
        indices = []

        for cls in client_classes:
            indices.extend(class_to_indices[cls][:SAMPLES_PER_CLIENT // len(client_classes)])

        subset = Subset(dataset, indices)

        x_data = torch.stack([subset[i][0] for i in range(len(subset))])
        y_data = torch.tensor([subset[i][1] for i in range(len(subset))])

        client_dir = f"data/clients_data/client_{client_id}"
        os.makedirs(client_dir, exist_ok=True)
        torch.save(x_data, f"{client_dir}/x_train.pt")
        torch.save(y_data, f"{client_dir}/y_train.pt")

        # For simplicity, using same data for test set now...
        torch.save(x_data, f"{client_dir}/x_test.pt")
        torch.save(y_data, f"{client_dir}/y_test.pt")

        print(f"Saved data for client_{client_id}: {len(subset)} samples")

if __name__ == "__main__":
    partition_dataset()
