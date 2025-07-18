import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import os

# Basic CNN model for EuroSAT (RGB, 64x64)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)  # 3 input channels (RGB)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(20 * 13 * 13, 50)  # Automatically calculated for 64x64 input
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Output: [batch, 10, 30, 30]
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Output: [batch, 20, 13, 13]
        x = x.view(x.size(0), -1)                   # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load local data (each client has its own data folder)
def load_data(client_id):
    path = f"data/clients_data/client_{client_id}/"
    x_train = torch.load(os.path.join(path, "x_train.pt"))
    y_train = torch.load(os.path.join(path, "y_train.pt"))
    x_test = torch.load(os.path.join(path, "x_test.pt"))
    y_test = torch.load(os.path.join(path, "y_test.pt"))
    return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)

def train(model, train_loader, epochs, device):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

def test(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


if __name__ == "__main__":
    print("Testing local model and dataset loading...")

    test_cid = "0"  # Use any existing client folder under data/clients_data/
    trainset, testset = load_data(test_cid)

    print(f"Loaded trainset size: {len(trainset)}")
    print(f"Loaded testset size: {len(testset)}")

    model = Net()
    sample_x, sample_y = next(iter(trainset))
    print(f"Sample input shape: {sample_x.shape}")
    print(f"Sample label: {sample_y}")
    print("Model architecture:")
    print(model)
