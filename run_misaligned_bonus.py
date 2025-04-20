import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm

# -----------------------
# Dataset with GT shifting
# -----------------------
class RoadDataset(Dataset):
    def __init__(self, input_dir, target_dir, shift_targets=False):
        self.input_paths = sorted(glob(os.path.join(input_dir, "*.png")))
        self.target_paths = sorted(glob(os.path.join(target_dir, "*.png")))
        self.shift_targets = shift_targets

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.input_paths[idx], cv2.IMREAD_GRAYSCALE) / 255.0
        y = cv2.imread(self.target_paths[idx], cv2.IMREAD_GRAYSCALE) / 255.0

        if self.shift_targets:
            shift_x = np.random.randint(-2, 3)
            shift_y = np.random.randint(-2, 3)
            y = np.roll(y, shift=(shift_y, shift_x), axis=(0, 1))

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return x, y

# -----------------------
# U-Net
# -----------------------
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

# -----------------------
# Dice Loss
# -----------------------
class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1 - ((2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth))

# -----------------------
# Train with Misaligned GT
# -----------------------
def train_misaligned():
    dataset = RoadDataset("data/input", "data/target", shift_targets=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    bce = nn.BCELoss()
    dice = DiceLoss()
    loss_fn = lambda p, t: 0.5 * bce(p, t) + 0.5 * dice(p, t)

    os.makedirs("results/misaligned_run", exist_ok=True)
    loss_history = []

    for epoch in range(10):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Save model + curve
    torch.save(model.state_dict(), "results/misaligned_run/model.pth")
    plt.plot(loss_history)
    plt.title("Misaligned GT Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("results/misaligned_run/loss_curve.png")
    plt.close()

    # Save 3 predictions
    model.eval()
    with torch.no_grad():
        for i in range(3):
            x, y = dataset[i]
            pred = model(x.unsqueeze(0)).squeeze().numpy()
            fig, axs = plt.subplots(1, 3, figsize=(10, 3))
            axs[0].imshow(x.squeeze(), cmap='gray')
            axs[0].set_title("Input")
            axs[1].imshow(y.squeeze(), cmap='gray')
            axs[1].set_title("Shifted GT")
            axs[2].imshow(pred, cmap='gray')
            axs[2].set_title("Prediction")
            for ax in axs: ax.axis('off')
            plt.savefig(f"results/misaligned_run/sample_{i}.png")
            plt.close()

if __name__ == "__main__":
    train_misaligned()
