import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# ---------------------
# Dataset Loader (with GT shifting)
# ---------------------
class RoadDataset(Dataset):
    def __init__(self, input_dir, target_dir, shift_targets=False):
        self.input_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        self.target_paths = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir)])
        self.shift_targets = shift_targets

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.input_paths[idx], cv2.IMREAD_GRAYSCALE) / 255.0
        y = cv2.imread(self.target_paths[idx], cv2.IMREAD_GRAYSCALE) / 255.0

        if self.shift_targets:
            shift_x = random.randint(-2, 2)
            shift_y = random.randint(-2, 2)
            y = np.roll(y, shift=(shift_y, shift_x), axis=(0, 1))

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return x, y

# ---------------------
# U-Net
# ---------------------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU()
            )

        self.enc1 = block(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = block(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = block(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = block(64, 128)

        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = block(128, 64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = block(64, 32)
        self.upconv1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = block(32, 16)

        self.final_conv = nn.Conv2d(16, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        out = self.final_conv(dec1)
        return self.sigmoid(out)

# ---------------------
# Loss Functions
# ---------------------
class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1 - ((2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth))

class SoftAlignedDiceLoss(nn.Module):
    def __init__(self, max_shift=2):
        super().__init__()
        self.max_shift = max_shift
        self.base_dice = DiceLoss()

    def forward(self, inputs, targets):
        best_loss = float('inf')
        for shift_x in range(-self.max_shift, self.max_shift + 1):
            for shift_y in range(-self.max_shift, self.max_shift + 1):
                shifted_target = torch.roll(targets, shifts=(shift_y, shift_x), dims=(2, 3))
                loss = self.base_dice(inputs, shifted_target)
                if loss < best_loss:
                    best_loss = loss
        return best_loss

def compute_accuracy(preds, targets, threshold=0.5):
    preds_bin = (preds > threshold).float()
    correct = (preds_bin == targets).float().sum()
    total = torch.numel(targets)
    return (correct / total).item()

def compute_precision(preds, targets, threshold=0.5, smooth=1):
    preds_bin = (preds > threshold).float()
    preds_bin = preds_bin.view(-1)
    targets = targets.view(-1)
    true_positive = (preds_bin * targets).sum()
    predicted_positive = preds_bin.sum()
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    return precision.item()

def compute_recall(preds, targets, threshold=0.5, smooth=1):
    preds_bin = (preds > threshold).float()
    preds_bin = preds_bin.view(-1)
    targets = targets.view(-1)
    true_positive = (preds_bin * targets).sum()
    actual_positive = targets.sum()
    recall = (true_positive + smooth) / (actual_positive + smooth)
    return recall.item()

def compute_f1(preds, targets, threshold=0.5, smooth=1):
    precision = compute_precision(preds, targets, threshold, smooth)
    recall = compute_recall(preds, targets, threshold, smooth)
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    return f1

def compute_dice(preds, targets, threshold=0.5, smooth=1):
    preds_bin = (preds > threshold).float()
    preds_bin = preds_bin.view(-1)
    targets = targets.view(-1)
    intersection = (preds_bin * targets).sum()
    dice = (2. * intersection + smooth) / (preds_bin.sum() + targets.sum() + smooth)
    return dice.item()

def compute_iou(preds, targets, threshold=0.5, smooth=1):
    preds_bin = (preds > threshold).float()
    preds_bin = preds_bin.view(-1)
    targets = targets.view(-1)
    intersection = (preds_bin * targets).sum()
    union = preds_bin.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def compute_mse_distance(preds, targets, threshold=0.5):
    """
    Compute Mean Squared Error based on distance transform from ground truth.
    preds: model outputs (batch_size, 1, H, W)
    targets: ground-truth masks (batch_size, 1, H, W)
    """
    batch_mse = []

    for pred, target in zip(preds, targets):
        # Threshold prediction
        pred_bin = (pred > threshold).float()

        # Convert GT to numpy and compute distance transform
        target_np = target.squeeze().cpu().numpy().astype(np.uint8)
        distance_transform = cv2.distanceTransform(1 - target_np, distanceType=cv2.DIST_L2, maskSize=5)

        # Prediction to numpy
        pred_bin_np = pred_bin.squeeze().cpu().numpy()

        # MSE computation only on predicted positives
        mse = np.mean((pred_bin_np * distance_transform) ** 2)
        batch_mse.append(mse)

    return np.mean(batch_mse)


# ---------------------
# Training Loop
# ---------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dir = "data/distorted_input"
    target_dir = "data/target"

    # Load full dataset without shifting
    full_dataset = RoadDataset(input_dir, target_dir, shift_targets=False)

    # Split dataset
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Wrap training dataset to apply random GT shifting
    class ShiftedDataset(torch.utils.data.Dataset):
        def __init__(self, subset):
            self.subset = subset

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            x, y = self.subset[idx]
            shift_x = random.randint(-2, 2)
            shift_y = random.randint(-2, 2)
            y = torch.roll(y, shifts=(shift_y, shift_x), dims=(1, 2))
            return x, y

    train_dataset = ShiftedDataset(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    bce = nn.BCELoss()
    dice = SoftAlignedDiceLoss(max_shift=2)
    loss_fn = lambda pred, target: 0.5 * bce(pred, target) + 0.5 * dice(pred, target)

    os.makedirs("results_misaligned", exist_ok=True)
    loss_history = []

    for epoch in range(20):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "results_misaligned/model.pth")

    # Save loss curve
    plt.plot(loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("results_misaligned/loss_curve.png")
    plt.close()

    # Evaluation
    model.eval()
    total_acc = 0
    total_dice = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_iou = 0
    total_val_mse = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_acc += compute_accuracy(pred, y, threshold=0.3)
            total_dice += compute_dice(pred, y, threshold=0.3)
            total_precision += compute_precision(pred, y, threshold=0.3)
            total_recall += compute_recall(pred, y, threshold=0.3)
            total_f1 += compute_f1(pred, y, threshold=0.3)
            total_iou += compute_iou(pred, y, threshold=0.3)
            total_val_mse += compute_mse_distance(pred, y, threshold=0.3)

    final_acc = total_acc / len(val_loader)
    final_dice = total_dice / len(val_loader)
    final_precision = total_precision / len(val_loader)
    final_recall = total_recall / len(val_loader)
    final_f1 = total_f1 / len(val_loader)
    final_iou = total_iou / len(val_loader)
    final_val_mse = total_val_mse / len(val_loader)


    print(f"Final Validation Accuracy: {final_acc:.4f}")
    print(f"Final Validation Dice Score: {final_dice:.4f}")
    print(f"Final Validation Precision: {final_precision:.4f}")
    print(f"Final Validation Recall: {final_recall:.4f}")
    print(f"Final Validation F1 Score: {final_f1:.4f}")
    print(f"Final Validation IoU: {final_iou:.4f}")
    print(f"Final Validation MSE (Distance Transform): {final_val_mse:.6f}")


    # Save a few predictions with original GT
    for i in range(3):
        x, y = val_dataset[i]
        x_input = x.unsqueeze(0).to(device)
        pred = model(x_input).squeeze().cpu().detach().numpy()

        pred_bin = (pred > 0.3).astype(np.uint8)
        pred_skel = skeletonize(pred_bin).astype(np.uint8) * 255

        # Load original GT from disk
        original_y_path = full_dataset.target_paths[val_dataset.indices[i]]
        original_gt = cv2.imread(original_y_path, cv2.IMREAD_GRAYSCALE) / 255.0

        # Generate shifted GT similar to training
        shift_x = random.randint(-2, 2)
        shift_y = random.randint(-2, 2)
        shifted_gt = np.roll(original_gt, shift=(shift_y, shift_x), axis=(0, 1))

        # Plot all
        fig, axs = plt.subplots(1, 4, figsize=(22, 5))
        axs[0].imshow(x.squeeze().cpu(), cmap='gray')
        axs[0].set_title("Input")

        axs[1].imshow(original_gt, cmap='gray')
        axs[1].set_title("Original Ground Truth")

        axs[2].imshow(shifted_gt, cmap='gray')
        axs[2].set_title("Shifted Ground Truth (Train)")

        axs[3].imshow(pred_skel, cmap='gray')
        axs[3].set_title("Prediction (Skeleton)")

        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"results_misaligned/preview_{i:05d}.png", bbox_inches='tight')
        plt.close()

        cv2.imwrite(f"results_misaligned/sample_{i:05d}.png", pred_skel)


    # ---------------------
    # Visualize and Save Evaluation Metrics
    # ---------------------

    metrics = {
        "Accuracy": final_acc,
        "Precision": final_precision,
        "Recall": final_recall,
        "F1 Score": final_f1,
        "Dice": final_dice,
        "IoU": final_iou,
        "MSE (Dist)": final_val_mse
    }

    # Bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics.keys(), metrics.values())
    plt.ylabel("Score")
    plt.title("Evaluation Metrics")
    plt.ylim(0, max(1.05, max(metrics.values()) + 0.05))
    plt.xticks(rotation=45)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01,
                f"{height:.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("results_misaligned/metrics_bar.png")
    plt.close()




if __name__ == "__main__":
    train()
