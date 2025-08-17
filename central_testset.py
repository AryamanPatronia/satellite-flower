import torch
from torchvision.datasets import EuroSAT
from torchvision import transforms
from collections import defaultdict
import random
import os

SAMPLES_PER_CLASS = 200  # or adjust based on balance
NUM_CLASSES = 10
SAVE_DIR = "data/central_testset"

# Create a central test set with 10 classes, each with SAMPLES_PER_CLASS samples
# Normalize the dataset with the mean and std of EuroSAT

def create_central_testset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3444, 0.3809, 0.4082], std=[0.1859, 0.1556, 0.1349])
    ])
    dataset = EuroSAT(root="data/raw", download=True, transform=transform)
    class_to_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset):
        class_to_indices[label].append(idx)


    test_indices = []
    val_indices = []
    val_ratio = 0.2  # 20% for validation
    for cls in range(NUM_CLASSES):
        indices = class_to_indices[cls]
        selected = random.sample(indices, SAMPLES_PER_CLASS)
        split = int((1 - val_ratio) * SAMPLES_PER_CLASS)
        test_indices.extend(selected[:split])
        val_indices.extend(selected[split:])

    # Shuffle indices
    random.shuffle(test_indices)
    random.shuffle(val_indices)

    x_test = torch.stack([dataset[i][0] for i in test_indices])
    y_test = torch.tensor([dataset[i][1] for i in test_indices])
    x_val = torch.stack([dataset[i][0] for i in val_indices])
    y_val = torch.tensor([dataset[i][1] for i in val_indices])

    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(x_test, f"{SAVE_DIR}/x.pt")
    torch.save(y_test, f"{SAVE_DIR}/y.pt")
    torch.save(test_indices, f"{SAVE_DIR}/indices.pt")
    torch.save(x_val, f"{SAVE_DIR}/x_val.pt")
    torch.save(y_val, f"{SAVE_DIR}/y_val.pt")
    torch.save(val_indices, f"{SAVE_DIR}/val_indices.pt")
    print(f"Saved global test set with {len(y_test)} samples (10 classes)")
    print(f"Saved global validation set with {len(y_val)} samples (10 classes)")

if __name__ == "__main__":
    create_central_testset()
