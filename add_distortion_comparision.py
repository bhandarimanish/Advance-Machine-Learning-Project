import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Paths ===
input_folder = 'data/input'
distorted_folder = 'data/distorted_input'
ground_truth_folder = 'data/target'
save_folder = 'data/comparison_plots'

# Create output folders if not exists
os.makedirs(distorted_folder, exist_ok=True)
os.makedirs(save_folder, exist_ok=True)

# === Distortion Functions ===
def add_blur(image):
    ksize = np.random.choice([3, 5])
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = np.random.uniform(2, 8)
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def adjust_brightness(image):
    value = np.random.randint(-30, 30)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.int16) + value, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    bright_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return bright_img

def adjust_contrast(image):
    alpha = np.random.uniform(0.8, 1.2)
    contrast_img = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return contrast_img

# === Step 1: Distort and Save ===
print("Applying distortions...")

for file_name in os.listdir(input_folder):
    if file_name.endswith('.png'):
        img_path = os.path.join(input_folder, file_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Cannot load {file_name}. Skipping.")
            continue

        img_blur = add_blur(img)
        img_noisy = add_noise(img_blur)
        img_bright = adjust_brightness(img_noisy)
        img_final = adjust_contrast(img_bright)

        output_path = os.path.join(distorted_folder, file_name)
        cv2.imwrite(output_path, img_final)

print("Distorted images saved successfully!")

# === Step 2: Create and Save Separate Comparison Plots ===
print("Creating comparison plots...")

# Pick 3 random images
all_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
random.seed(50)
print(random.sample(range(10), 3)) 
chosen_files = random.sample(all_files, 3)

for idx, file_name in enumerate(chosen_files):
    input_img = cv2.imread(os.path.join(input_folder, file_name))
    distorted_img = cv2.imread(os.path.join(distorted_folder, file_name))
    gt_file_name = file_name.replace('image_', 'target_')
    ground_truth_img = cv2.imread(os.path.join(ground_truth_folder, gt_file_name))

    # Safety check
    if input_img is None or distorted_img is None or ground_truth_img is None:
        print(f"Warning: Couldn't load one of the images for {file_name}. Skipping.")
        continue

    # Convert to RGB
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    distorted_img = cv2.cvtColor(distorted_img, cv2.COLOR_BGR2RGB)
    ground_truth_img = cv2.cvtColor(ground_truth_img, cv2.COLOR_BGR2RGB)

    # Create a figure for each image
    plt.figure(figsize=(15, 5))

    # Plot Input
    plt.subplot(1, 3, 1)
    plt.imshow(input_img)
    plt.title(f'Input: {file_name}')
    plt.axis('off')

    # Plot Distorted
    plt.subplot(1, 3, 2)
    plt.imshow(distorted_img)
    plt.title('Distorted')
    plt.axis('off')

    # Plot Ground Truth
    plt.subplot(1, 3, 3)
    plt.imshow(ground_truth_img)
    plt.title('Ground Truth')
    plt.axis('off')

    # Save each comparison plot separately
    save_path = os.path.join(save_folder, f'comparison_{idx}.png')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

print("All comparison plots created and saved!")
