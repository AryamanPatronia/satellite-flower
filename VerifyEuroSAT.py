import torch
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# Check any client's (satallite) if it is carrying the EuroSAT data.
CLIENT_ID = "5"
DATA_DIR = f"data/clients_data/client_{CLIENT_ID}"

# Load tensors
x_path = os.path.join(DATA_DIR, "x_train.pt")
y_path = os.path.join(DATA_DIR, "y_train.pt")

x = torch.load(x_path)
y = torch.load(y_path)

print(f"Loaded x_train shape: {x.shape}") 
print(f"Loaded y_train shape: {y.shape}")
print(f"Unique labels: {torch.unique(y)}")
print(f"Pixel value range: min={x.min().item()}, max={x.max().item()}")

# Show first image
img = x[0]
label = y[0].item()

plt.imshow(TF.to_pil_image(img))
plt.title(f"Client {CLIENT_ID} - Label: {label}")
plt.axis('off')
plt.show()
