from torchvision.datasets import EuroSAT
from torchvision.transforms import ToTensor

def download_dataset():
    EuroSAT(root="data/raw", download=True, transform=ToTensor())
    print("EuroSAT dataset downloaded to data/raw/")

if __name__ == "__main__":
    download_dataset()
