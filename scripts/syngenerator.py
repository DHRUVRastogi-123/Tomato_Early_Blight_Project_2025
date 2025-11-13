# Cell 1: Write the Synthetic Data Generator Script
import cv2
import numpy as np
import os
import random
from tqdm import tqdm
import albumentations as A
import sys

# --- [FIX] Path Correction for Colab ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
# --- End of Fix ---

# --- Configuration ---
# Source of "stamps": Our background-free Early Blight leaves
SOURCE_CUTOUTS_DIR = os.path.join(PROJECT_ROOT, "data/processed/cls_clean_train/early_blight")

# Source of backgrounds: Our original 'train' images
# Using 'train' as it has many large, healthy plant images
BACKGROUND_IMAGES_DIR = os.path.join(PROJECT_ROOT, "Dataset/Annotated/train/images")

# Destination: Our 'test' folder, which is currently empty
TARGET_IMAGES_DIR = os.path.join(PROJECT_ROOT, "Dataset/Annotated/test/images")
TARGET_LABELS_DIR = os.path.join(PROJECT_ROOT, "Dataset/Annotated/test/labels")

# The class index for "Early Blight"
EARLY_BLIGHT_INDEX = 0

# How many new images to create?
NUM_SYNTHETIC_IMAGES = 30 # Let's make 30 new test images
# ---------------------

def get_random_cutout(cutout_dir):
    """Loads a random cutout and its mask."""
    try:
        cutout_name = random.choice(os.listdir(cutout_dir))
        cutout_path = os.path.join(cutout_dir, cutout_name)
        cutout_img = cv2.imread(cutout_path)
        
        if cutout_img is None:
            return None, None

        # Create a mask from the black background
        cutout_gray = cv2.cvtColor(cutout_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(cutout_gray, 1, 255, cv2.THRESH_BINARY)
        
        return cutout_img, mask
    except:
        return None, None

def get_random_background(bg_dir):
    """Loads a random background image."""
    try:
        bg_name = random.choice(os.listdir(bg_dir))
        bg_path = os.path.join(bg_dir, bg_name)
        bg_img = cv2.imread(bg_path)
        
        # Check if background is too small
        if bg_img is None or bg_img.shape[0] < 200 or bg_img.shape[1] < 200:
            return None
        return bg_img
    except:
        return None

# Albumentations for the cutout
cutout_transform = A.Compose([
    A.Resize(width=random.randint(50, 150), height=random.randint(50, 150), p=1),
    A.Rotate(limit=90, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
])

def generate_synthetic_data():
    print("--- Starting Synthetic Data Generation ---")
    
    os.makedirs(TARGET_IMAGES_DIR, exist_ok=True)
    os.makedirs(TARGET_LABELS_DIR, exist_ok=True)
    
    cutout_files = os.listdir(SOURCE_CUTOUTS_DIR)
    if not cutout_files:
        print(f"Error: No cutout files found in {SOURCE_CUTOUTS_DIR}")
        print("Please run 'build_clean_classifier_dataset.py' (Cell 6) first.")
        return

    bg_files = os.listdir(BACKGROUND_IMAGES_DIR)
    if not bg_files:
        print(f"Error: No background files found in {BACKGROUND_IMAGES_DIR}")
        return

    print(f"Creating {NUM_SYNTHETIC_IMAGES} new test images...")

    for i in tqdm(range(NUM_SYNTHETIC_IMAGES)):
        # 1. Load a random background
        bg_img = get_random_background(BACKGROUND_IMAGES_DIR)
        if bg_img is None:
            continue
            
        h_bg, w_bg = bg_img.shape[:2]
        
        # 2. Load a random cutout and its mask
        cutout_img, mask = get_random_cutout(SOURCE_CUTOUTS_DIR)
        if cutout_img is None:
            continue
            
        # 3. Augment the cutout
        transformed = cutout_transform(image=cutout_img, mask=mask)
        cutout_aug = transformed['image']
        mask_aug = transformed['mask']
        
        h_cut, w_cut = cutout_aug.shape[:2]

        # 4. Find a random place to paste it
        if h_bg <= h_cut or w_bg <= w_cut:
            # Background is smaller than cutout, skip
            continue
            
        x_paste = random.randint(0, w_bg - w_cut)
        y_paste = random.randint(0, h_bg - h_cut)
        
        # 5. Paste the cutout onto the background
        roi = bg_img[y_paste:y_paste+h_cut, x_paste:x_paste+w_cut]
        
        mask_inv = cv2.bitwise_not(mask_aug)
        
        bg_masked = cv2.bitwise_and(roi, roi, mask=mask_inv)
        cutout_masked = cv2.bitwise_and(cutout_aug, cutout_aug, mask=mask_aug)
        
        dst = cv2.add(bg_masked, cutout_masked)
        bg_img[y_paste:y_paste+h_cut, x_paste:x_paste+w_cut] = dst
        
        # 6. Save the new synthetic image
        save_name = f"synthetic_img_{i:04d}"
        img_save_path = os.path.join(TARGET_IMAGES_DIR, save_name + ".jpg")
        cv2.imwrite(img_save_path, bg_img)
        
        # 7. Save the corresponding label file (in the format you need)
        # We need: class_id x_center_norm y_center_norm radius_norm
        
        x_center_pix = x_paste + w_cut / 2
        y_center_pix = y_paste + h_cut / 2
        radius_pix = max(w_cut, h_cut) / 2
        
        x_center_norm = x_center_pix / w_bg
        y_center_norm = y_center_pix / h_bg
        radius_norm = radius_pix / max(w_bg, h_bg)
        
        label_line = f"{EARLY_BLIGHT_INDEX} {x_center_norm:.6f} {y_center_norm:.6f} {radius_norm:.6f}\n"
        
        label_save_path = os.path.join(TARGET_LABELS_DIR, save_name + ".txt")
        with open(label_save_path, 'w') as f:
            f.write(label_line)

    print(f"--- Synthetic Data Generation Complete ---")
    print(f"Added {NUM_SYNTHETIC_IMAGES} new images to {TARGET_IMAGES_DIR}")
    print(f"Added {NUM_SYNTHETIC_IMAGES} new labels to {TARGET_LABELS_DIR}")

if __name__ == "__main__":
    generate_synthetic_data()

print("File 'scripts/synthgenerator.py' created.")