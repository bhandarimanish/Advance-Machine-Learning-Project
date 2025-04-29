import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree

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

        # Ground-truth to numpy and distance transform
        target_np = target.squeeze().cpu().numpy().astype(np.uint8)
        distance_transform = cv2.distanceTransform(1 - target_np, distanceType=cv2.DIST_L2, maskSize=5)

        # Prediction to numpy
        pred_bin_np = pred_bin.squeeze().cpu().numpy()

        # MSE computation
        mse = np.mean((pred_bin_np * distance_transform) ** 2)
        batch_mse.append(mse)

    return np.mean(batch_mse)  # Mean over batch


# ---------------------
# Node Precision and Recall (Valence-based)
# ---------------------

def compute_valence_map(skeleton):
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton, -1, kernel)
    valence_map = skeleton * neighbor_count
    return valence_map

def extract_nodes(valence_map, valence):
    coords = np.argwhere(valence_map == valence)
    return coords

def bipartite_match(pred_nodes, gt_nodes, max_dist=3):
    if len(pred_nodes) == 0 or len(gt_nodes) == 0:
        return 0, 0
    tree = cKDTree(gt_nodes)
    dists, indices = tree.query(pred_nodes, distance_upper_bound=max_dist)
    matched_pred = np.sum(dists <= max_dist)
    matched_gt = len(set(indices[dists <= max_dist]))
    return matched_pred, matched_gt

def evaluate_node_precision_recall(pred_skeleton, gt_skeleton, threshold=0.5):
    pred_bin = (pred_skeleton > threshold).float().squeeze().cpu().numpy().astype(np.uint8)
    gt_bin = gt_skeleton.squeeze().cpu().numpy().astype(np.uint8)

    pred_bin = skeletonize(pred_bin).astype(np.uint8)
    gt_bin = skeletonize(gt_bin).astype(np.uint8)

    pred_valence = compute_valence_map(pred_bin)
    gt_valence = compute_valence_map(gt_bin)

    valences = [1, 2, 3, 4]
    results = {}

    for v in valences:
        pred_nodes = extract_nodes(pred_valence, v)
        gt_nodes = extract_nodes(gt_valence, v)
        matched_pred, matched_gt = bipartite_match(pred_nodes, gt_nodes, max_dist=3)

        precision = matched_pred / max(len(pred_nodes), 1)
        recall = matched_gt / max(len(gt_nodes), 1)

        results[v] = {
            "precision": precision,
            "recall": recall,
            "matched_pred": matched_pred,
            "total_pred": len(pred_nodes),
            "matched_gt": matched_gt,
            "total_gt": len(gt_nodes)
        }

    return results

def compute_iou(preds, targets, threshold=0.5, smooth=1):
    preds_bin = (preds > threshold).float()
    preds_bin = preds_bin.view(-1)
    targets = targets.view(-1)
    intersection = (preds_bin * targets).sum()
    union = preds_bin.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


# ---------------------
# Training Loop
# ---------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dir = "data/distorted_input"
    target_dir = "data/target"
    dataset = RoadDataset(input_dir, target_dir)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    bce = nn.BCELoss()
    dice = DiceLoss()
    loss_fn = lambda pred, target: 0.5 * bce(pred, target) + 0.5 * dice(pred, target)

    os.makedirs("results", exist_ok=True)
    loss_history = []

    for epoch in range(3):
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
    torch.save(model.state_dict(), "results/model.pth")

    # Save loss curve
    plt.plot(loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("results/loss_curve.png")
    plt.close()

    # Final Evaluation on Validation Set
    model.eval()
    total_acc = 0
    total_dice = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_val_loss = 0
    total_val_mse = 0
    total_iou = 0 

    # NEW: to collect node precision/recall
    node_precision_recall_results = {1: [], 2: [], 3: [], 4: []}

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            # Regular Metrics
            loss = loss_fn(pred, y)
            total_val_loss += loss.item()
            total_acc += compute_accuracy(pred, y, threshold=0.3)
            total_dice += compute_dice(pred, y, threshold=0.3)
            total_precision += compute_precision(pred, y, threshold=0.3)
            total_recall += compute_recall(pred, y, threshold=0.3)
            total_f1 += compute_f1(pred, y, threshold=0.3)
            total_val_mse += compute_mse_distance(pred, y, threshold=0.3)
            total_iou += compute_iou(pred, y, threshold=0.3)

            # NEW: Node Precision & Recall
            batch_result = evaluate_node_precision_recall(pred, y, threshold=0.3)
            for v in [1, 2, 3, 4]:
                node_precision_recall_results[v].append(batch_result[v])

    # Averages for standard metrics
    final_val_loss = total_val_loss / len(val_loader)
    final_acc = total_acc / len(val_loader)
    final_dice = total_dice / len(val_loader)
    final_precision = total_precision / len(val_loader)
    final_recall = total_recall / len(val_loader)
    final_f1 = total_f1 / len(val_loader)
    final_val_mse = total_val_mse / len(val_loader)
    final_iou = total_iou / len(val_loader)


    # NEW: Averages for node metrics
    final_node_results = {}
    for v in [1, 2, 3, 4]:
        precisions = [r['precision'] for r in node_precision_recall_results[v]]
        recalls = [r['recall'] for r in node_precision_recall_results[v]]
        final_node_results[v] = {
            "precision": np.mean(precisions),
            "recall": np.mean(recalls)
        }

    # Now Print Final Standard Metrics
    print(f"Final Validation Loss (Test Loss): {final_val_loss:.4f}")
    print(f"Final Validation Accuracy: {final_acc:.4f}")
    print(f"Final Validation Dice Score: {final_dice:.4f}")
    print(f"Final Validation IoU: {final_iou:.4f}") 
    print(f"Final Validation Precision: {final_precision:.4f}")
    print(f"Final Validation Recall: {final_recall:.4f}")
    print(f"Final Validation F1 Score: {final_f1:.4f}")
    print(f"Final Validation MSE (Distance Transform): {final_val_mse:.6f}")

    # Print Final Node Precision & Recall
    for v in [1,2,3,4]:
        print(f"{v}-valent Node Precision: {final_node_results[v]['precision']:.4f}, Recall: {final_node_results[v]['recall']:.4f}")


    # Save predictions for a few samples
    for i in range(3):
        x, y = val_dataset[i]
        x = x.unsqueeze(0).to(device)
        pred = model(x).squeeze().cpu().detach().numpy()

        # Threshold prediction
        pred_bin = (pred > 0.3).astype(np.uint8)

        # Skeletonize
        pred_skel = skeletonize(pred_bin).astype(np.uint8) * 255

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(x.squeeze().cpu(), cmap='gray')
        axs[0].set_title("Input")
        axs[1].imshow(y.squeeze().cpu(), cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[2].imshow(pred_skel, cmap='gray')
        axs[2].set_title("Prediction (Skeleton)")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"results/preview_{i:05d}.png", bbox_inches='tight')
        plt.close()

        # Save only skeleton prediction
        cv2.imwrite(f"results/sample_{i:05d}.png", pred_skel)

if __name__ == "__main__":
    train()
