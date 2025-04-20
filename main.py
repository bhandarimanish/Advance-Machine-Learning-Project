import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------
# Dataset Loader
# ---------------------
class RoadDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        self.target_paths = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir)])

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.input_paths[idx], cv2.IMREAD_GRAYSCALE) / 255.0
        y = cv2.imread(self.target_paths[idx], cv2.IMREAD_GRAYSCALE) / 255.0
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return x, y

# ---------------------
# Minimal U-Net
# ---------------------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, 2, stride=2), nn.ReLU(),
            nn.Conv2d(8, 1, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.sigmoid(x)

# ---------------------
# Training Loop
# ---------------------
def train():
    input_dir = "data/input"
    target_dir = "data/target"
    dataset = RoadDataset(input_dir, target_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(5):
        model.train()
        epoch_loss = 0
        for x, y in dataloader:
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/model.pth")

    # Save some sample predictions
    model.eval()
    with torch.no_grad():
        for i in range(3):
            x, y = dataset[i]
            pred = model(x.unsqueeze(0)).squeeze().numpy()
            pred_bin = (pred > 0.5).astype(np.uint8)

            fig, axs = plt.subplots(1, 3, figsize=(10, 3))
            axs[0].imshow(x.squeeze(), cmap='gray'); axs[0].set_title("Input")
            axs[1].imshow(y.squeeze(), cmap='gray'); axs[1].set_title("Ground Truth")
            axs[2].imshow(pred_bin, cmap='gray'); axs[2].set_title("Prediction")
            for ax in axs: ax.axis('off')
            plt.savefig(f'results/sample_{i}.png')
            plt.close()

if __name__ == "__main__":
    train()
