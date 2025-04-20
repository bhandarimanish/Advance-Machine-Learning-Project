import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
import json

# -----------------------
# Dataset
# -----------------------
class RoadDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_paths = sorted(glob(os.path.join(input_dir, "*.png")))
        self.target_paths = sorted(glob(os.path.join(target_dir, "*.png")))

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.input_paths[idx], cv2.IMREAD_GRAYSCALE) / 255.0
        y = cv2.imread(self.target_paths[idx], cv2.IMREAD_GRAYSCALE) / 255.0
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
# Losses
# -----------------------
class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1 - ((2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth))

def mse(pred, target):
    return ((pred - target) ** 2).mean()

def dice_coefficient(pred, target, smooth=1):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# -----------------------
# Train and Evaluate
# -----------------------
def run_experiment(name, loss_fn, save_dir):
    print(f"\n Running experiment: {name}")
    os.makedirs(save_dir, exist_ok=True)

    dataset = RoadDataset("data/input", "data/target")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_history = []

    for epoch in range(5):  # short training for ablation
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

    # Save loss curve
    plt.plot(loss_history)
    plt.title(f"{name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # Evaluate
    model.eval()
    all_mse = []
    all_dice = []
    with torch.no_grad():
        for i, (x, y) in enumerate(dataset):
            pred = model(x.unsqueeze(0)).squeeze().numpy()
            pred_bin = (pred > 0.5).astype(np.float32)
            all_mse.append(mse(pred_bin, y.squeeze().numpy()))
            all_dice.append(dice_coefficient(pred_bin, y.squeeze().numpy()))

            if i < 3:
                fig, axs = plt.subplots(1, 3, figsize=(10, 3))
                axs[0].imshow(x.squeeze(), cmap='gray')
                axs[0].set_title("Input")
                axs[1].imshow(y.squeeze(), cmap='gray')
                axs[1].set_title("Ground Truth")
                axs[2].imshow(pred, cmap='gray')
                axs[2].set_title("Prediction (Soft)")
                for ax in axs: ax.axis('off')
                plt.savefig(os.path.join(save_dir, f"sample_{i}.png"))
                plt.close()

    results = {
        "avg_loss": loss_history[-1],
        "avg_mse": float(np.mean(all_mse)),
        "avg_dice": float(np.mean(all_dice))
    }
    print(f"\n{name} - MSE: {results['avg_mse']:.5f}, Dice: {results['avg_dice']:.5f}")
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

# -----------------------
# Run all configs
# -----------------------
if __name__ == "__main__":
    bce = nn.BCELoss()
    dice = DiceLoss()

    run_experiment("BCE Only", bce, "ablation_results/run_bce")
    run_experiment("Dice Only", dice, "ablation_results/run_dice")
    run_experiment("BCE + Dice", lambda p, t: 0.5 * bce(p, t) + 0.5 * dice(p, t), "ablation_results/run_bce_dice")
