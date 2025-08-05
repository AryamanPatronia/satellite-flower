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

    selected_indices = []
    for cls in range(NUM_CLASSES):
        indices = class_to_indices[cls]
        selected = random.sample(indices, SAMPLES_PER_CLASS)
        selected_indices.extend(selected)

    # Shuffle all selected indices
    random.shuffle(selected_indices)

    x = torch.stack([dataset[i][0] for i in selected_indices])
    y = torch.tensor([dataset[i][1] for i in selected_indices])

    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(x, f"{SAVE_DIR}/x.pt")
    torch.save(y, f"{SAVE_DIR}/y.pt")
    torch.save(selected_indices, f"{SAVE_DIR}/indices.pt")
    print(f"Saved global test set with {len(y)} samples (10 classes)")

if __name__ == "__main__":
    create_central_testset()
