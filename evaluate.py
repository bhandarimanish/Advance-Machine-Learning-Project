import os
import torch
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from main import UNet  # Reuse your U-Net definition

# ----------------------
# Metrics
# ----------------------

def mse_distance_transform(pred_bin, gt_skel):
    """
    Computes MSE between prediction and distance transform of ground truth skeleton.
    The closer prediction is to skeleton, the smaller the loss.
    """
    # Convert skeleton to distance map (invert to compute distance to nearest foreground pixel)
    dist_transform = distance_transform_edt(1 - gt_skel)
    return ((pred_bin * dist_transform) ** 2).mean()

def dice_coefficient(pred, target, smooth=1):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# ----------------------
# Helpers
# ----------------------

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (img / 255.0).astype(np.float32)

# ----------------------
# Evaluation Function
# ----------------------

def evaluate(input_dir, target_dir, model_path):
    model = UNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    input_paths = sorted(glob(os.path.join(input_dir, "*.png")))
    target_paths = sorted(glob(os.path.join(target_dir, "*.png")))

    all_mse = []
    all_dice = []

    for inp_path, tgt_path in tqdm(zip(input_paths, target_paths), total=len(input_paths)):
        x = load_image(inp_path)
        y = load_image(tgt_path)

        x_tensor = torch.tensor(x).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pred = model(x_tensor).squeeze().numpy()

        pred_bin = (pred > 0.5).astype(np.float32)
        gt_skel = skeletonize((y > 0.5)).astype(np.float32)

        mse_score = mse_distance_transform(pred_bin, gt_skel)
        dice_score = dice_coefficient(pred_bin, gt_skel)

        all_mse.append(mse_score)
        all_dice.append(dice_score)

    print(f"\n📊 Evaluation Results:")
    print(f"Avg MSE (DistMap): {np.mean(all_mse):.6f}")
    print(f"Avg Dice:          {np.mean(all_dice):.6f}")

# ----------------------
# Run
# ----------------------

if __name__ == "__main__":
    evaluate("data/input", "data/target", "results/model.pth")
