import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os

# Custom simple CNN for EuroSAT (Second Simulation Settings)
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)  # (3, 64, 64) -> (10, 60, 60)
        self.pool = nn.MaxPool2d(2)                  # (10, 60, 60) -> (10, 30, 30)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # (10, 30, 30) -> (20, 26, 26)
        # pool again: (20, 26, 26) -> (20, 13, 13)
        self.fc1 = nn.Linear(20 * 13 * 13, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Load local data (each client has its own data folder)
def load_data(client_id):
    path = f"data/clients_data/client_{client_id}/"
    x_train = torch.load(os.path.join(path, "x_train.pt"))
    y_train = torch.load(os.path.join(path, "y_train.pt"))
    x_test = torch.load(os.path.join(path, "x_test.pt"))
    y_test = torch.load(os.path.join(path, "y_test.pt"))

    # Data is already normalized during creation, so no further transforms are needed here.
    return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)

def train(model, train_loader, epochs, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {acc:.4f}")

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
    test_cid = "1"
    trainset, testset = load_data(test_cid)

    print(f"Loaded trainset size: {len(trainset)}")
    print(f"Loaded testset size: {len(testset)}")

    model = Net()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)

    train(model, trainloader, epochs=3, device=DEVICE)
    accuracy = test(model, testloader, device=DEVICE)
    print(f"Local test accuracy after training: {accuracy:.4f}")
