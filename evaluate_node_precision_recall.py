import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def binarize(path, threshold=100):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return (img > threshold).astype(np.uint8)

def compute_valence_map(skeleton):
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]])
    neighbors = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    return neighbors * skeleton

def get_nodes_by_valence(valence_map, valence):
    return np.column_stack(np.where(valence_map == valence))

def match(pred_nodes, gt_nodes, radius=3):
    matched = 0
    used = set()
    for px, py in pred_nodes:
        for i, (gx, gy) in enumerate(gt_nodes):
            if i in used:
                continue
            if np.hypot(px - gx, py - gy) <= radius:
                matched += 1
                used.add(i)
                break
    return matched, len(pred_nodes), len(gt_nodes)

def print_results(valence_results):
    print("\n🎯 Valence-wise Precision and Recall (radius ≤ 3 pixels):")
    for v in range(1, 5):
        tp, pred, gt = valence_results[v]
        if pred == 0 or gt == 0:
            print(f"{v}-valent → No nodes found")
        else:
            precision = tp / pred
            recall = tp / gt
            print(f"{v}-valent → Precision: {precision:.3f}, Recall: {recall:.3f}")

def plot_bar_chart(valence_results, save_path="results/valence_bar_chart.png"):
    valences = [1, 2, 3, 4]
    precision = []
    recall = []
    for v in valences:
        tp, pred, gt = valence_results[v]
        p = tp / pred if pred else 0
        r = tp / gt if gt else 0
        precision.append(p)
        recall.append(r)

    x = np.arange(len(valences))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, precision, width, label='Precision')
    ax.bar(x + width/2, recall, width, label='Recall')

    ax.set_ylabel('Score')
    ax.set_xlabel('Node Valence')
    ax.set_title('Node Precision & Recall by Valence')
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in valences])
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

def find_first_valid_pair():
    """Return the first valid matching (pred, gt) pair."""
    for i in range(1000):
        pred_path = f"results/sample_{i:05d}.png"
        gt_path = f"data/target/image_{i:05d}.png"
        if os.path.exists(pred_path) and os.path.exists(gt_path):
            return pred_path, gt_path
    raise FileNotFoundError("No matching prediction/ground truth pair found.")

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    try:
        pred_path, gt_path = find_first_valid_pair()
        print(f"✅ Using prediction: {pred_path}")
        print(f"✅ Using ground truth: {gt_path}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        exit(1)

    pred = binarize(pred_path)
    gt = binarize(gt_path)

    pred_skel = skeletonize(pred)
    gt_skel = skeletonize(gt)

    pred_val_map = compute_valence_map(pred_skel)
    gt_val_map = compute_valence_map(gt_skel)

    valence_results = defaultdict(lambda: [0, 0, 0])

    for val in range(1, 5):
        pred_nodes = get_nodes_by_valence(pred_val_map, val)
        gt_nodes = get_nodes_by_valence(gt_val_map, val)
        tp, pred_count, gt_count = match(pred_nodes, gt_nodes, radius=3)
        valence_results[val] = [tp, pred_count, gt_count]

    print_results(valence_results)
    plot_bar_chart(valence_results)
