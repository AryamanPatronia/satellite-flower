import torch
import os
from torchvision.datasets import EuroSAT
from torchvision import transforms
from torch.utils.data import Subset
import random
from collections import defaultdict

NUM_CLIENTS = 5
SAMPLES_PER_CLIENT = 5000
CLASSES_PER_CLIENT = 4  # Overlap: each client gets 4 classes

def partition_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3444, 0.3809, 0.4082], std=[0.1859, 0.1556, 0.1349])
    ])
    dataset = EuroSAT(root="data/raw", download=False, transform=transform)

    # Load central test indices
    central_test_indices_path = "data/central_testset/indices.pt"
    if os.path.exists(central_test_indices_path):
        central_test_indices = set(torch.load(central_test_indices_path))
    else:
        central_test_indices = set()

    # Group indices by class
    class_to_indices = defaultdict(list)
    # When building class_to_indices:
    for idx, (_, label) in enumerate(dataset):
        if idx not in central_test_indices:
            class_to_indices[label].append(idx)

    os.makedirs("data/clients_data", exist_ok=True)

    class_ids = list(class_to_indices.keys())
    num_classes = len(class_ids)
    random.seed(42)

    for i in range(NUM_CLIENTS):
        client_id = i + 1  # Start from 1
        # Overlapping assignment: sliding window
        start = (i * 2) % num_classes
        client_classes = [class_ids[(start + j) % num_classes] for j in range(CLASSES_PER_CLIENT)]
        indices = []
        for cls in client_classes:
            cls_indices = class_to_indices[cls]
            per_client = min(SAMPLES_PER_CLIENT // CLASSES_PER_CLIENT, len(cls_indices) // NUM_CLIENTS)
            indices.extend(cls_indices[i * per_client:(i + 1) * per_client])

        # Shuffle and split into train and test (80/20 split)
        random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_indices = indices[:split]
        test_indices = indices[split:]

        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)

        x_train = torch.stack([train_subset[i][0] for i in range(len(train_subset))])
        y_train = torch.tensor([train_subset[i][1] for i in range(len(train_subset))])

        x_test = torch.stack([test_subset[i][0] for i in range(len(test_subset))])
        y_test = torch.tensor([test_subset[i][1] for i in range(len(test_subset))])

        client_dir = f"data/clients_data/client_{client_id}"
        os.makedirs(client_dir, exist_ok=True)
        torch.save(x_train, f"{client_dir}/x_train.pt")
        torch.save(y_train, f"{client_dir}/y_train.pt")
        torch.save(x_test, f"{client_dir}/x_test.pt")
        torch.save(y_test, f"{client_dir}/y_test.pt")

        print(f"Saved data for client_{client_id}: {len(train_subset)} train + {len(test_subset)} test samples, classes: {client_classes}")

if __name__ == "__main__":
    partition_dataset()
