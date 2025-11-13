# scripts/build_clean_classifier_dataset.py
# This script builds a dataset of background-removed leaves.

import cv2
import numpy as np
import os
from tqdm import tqdm
import sys

from utils.segmenter import LeafSegmenter

# ------------------------------------------------------------------
# TODO: PLEASE UPDATE THIS
# What is the class index for "Early Blight" in your label files?
EARLY_BLIGHT_INDEX = 1
HEALTHY_INDEX = 7
# ------------------------------------------------------------------

# --- Configuration ---
# [FIX] Hardcode paths for Colab

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

SOURCE_DATA_DIR = os.path.join(PROJECT_ROOT, "Dataset/Annotated")
TARGET_DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")

# Settings
IOU_THRESHOLD = 0.01 # If a leaf overlaps a disease circle by even 1%.
MIN_MASK_AREA = 100 # Min pixel area to be considered a valid leaf.
BACKGROUND_COLOR = 'white' # 'black', 'white', or 'transparent'

# --- Helper Functions (Copied from previous script) ---

def parse_labels(label_path, img_shape):
    """Reads a label file and returns a list of disease 'bounding boxes'."""
    h, w = img_shape[:2]
    disease_boxes = []
    
    if not os.path.exists(label_path):
        return disease_boxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            radius_norm = float(parts[3])
            x_center = int(x_center_norm * w)
            y_center = int(y_center_norm * h)
            radius = int(radius_norm * max(w, h)) 
            xmin = max(0, x_center - radius)
            ymin = max(0, y_center - radius)
            xmax = min(w, x_center + radius)
            ymax = min(h, y_center + radius)
            
            disease_boxes.append({
                'class_id': class_id,
                'box': [xmin, ymin, xmax, ymax]
            })
    return disease_boxes


def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) for two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    if union == 0:
        return 0
    return intersection / union

# --- Main Processing Function ---

def process_dataset_split(split_name, segmenter):
    """
    Main function to process a dataset split (train, val, test).
    """
    print(f"\nProcessing split: {split_name}")
    
    # Define source paths
    img_dir = os.path.join(SOURCE_DATA_DIR, split_name, "images")
    label_dir = os.path.join(SOURCE_DATA_DIR, split_name, "labels")
    
    # Define target paths
    # [NEW] Using 'cls_clean' to denote this new dataset
    healthy_save_dir = os.path.join(TARGET_DATA_DIR, f"cls_clean_{split_name}", "healthy")
    disease_save_dir = os.path.join(TARGET_DATA_DIR, f"cls_clean_{split_name}", "early_blight")
    
    os.makedirs(healthy_save_dir, exist_ok=True)
    os.makedirs(disease_save_dir, exist_ok=True)
    
    if not os.path.exists(img_dir):
        print(f"Warning: Image directory {img_dir} does not exist. Skipping split.")
        return

    image_files = os.listdir(img_dir)
    
    for img_file in tqdm(image_files, desc=f"Processing {split_name} images"):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)
        
        # --- 1. Run Segmentation (Person A's tool) ---
        try:
            processed_seg_img, original_img_rgb, scale_factors = segmenter.preprocess_image(img_path)
        except FileNotFoundError:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        results = segmenter.run_yolov8_segmentation(processed_seg_img)
        leaf_masks, leaf_boxes = segmenter.extract_leaf_masks(results, scale_factors)
        
        if not leaf_boxes:
            continue # No leaves found in this image

        # --- 2. Load Disease Labels (Person B's data) ---
        all_disease_boxes = parse_labels(label_path, original_img_rgb.shape[:2])
        
        blight_circles = [
            db['box'] for db in all_disease_boxes 
            if db['class_id'] != HEALTHY_INDEX
        ]

        # --- 3. Match Leaves to Labels & Apply Mask ---
        for i, (leaf_mask, leaf_box) in enumerate(zip(leaf_masks, leaf_boxes)):
            
            # --- [NEW] This is the new logic ---
            # 1. Clean the mask
            cleaned_mask = segmenter.post_process_mask(leaf_mask, min_area=MIN_MASK_AREA)
            
            # 2. Apply mask to create the background-free crop
            clean_leaf_crop = segmenter.apply_mask_and_crop(
                original_img_rgb, 
                cleaned_mask, 
                leaf_box, 
                background_type=BACKGROUND_COLOR
            )
            # --- [END NEW] ---

            if clean_leaf_crop is None:
                continue
                
            # Now, check if this leaf is diseased
            is_diseased = False
            for circle_box in blight_circles:
                # We still check overlap using the leaf's *bounding box*
                if calculate_iou(leaf_box, circle_box) > IOU_THRESHOLD:
                    is_diseased = True
                    break
            
            # --- 4. Save to the new dataset ---
            # We must convert back to BGR for cv2.imwrite
            clean_leaf_bgr = cv2.cvtColor(clean_leaf_crop, cv2.COLOR_RGB2BGR)
            save_name = f"{os.path.splitext(img_file)[0]}_leaf_{i}.jpg"
            
            if is_diseased:
                cv2.imwrite(os.path.join(disease_save_dir, save_name), clean_leaf_bgr)
            else:
                cv2.imwrite(os.path.join(healthy_save_dir, save_name), clean_leaf_bgr)

# --- Main Execution ---
if __name__ == "__main__":
    
    print("Initializing Leaf Segmenter (may download YOLOv8n-seg)...")
    segmenter = LeafSegmenter(model_path="yolov8n-seg.pt")
    print("Segmenter initialized.")
    
    process_dataset_split("train", segmenter)
    process_dataset_split("valid", segmenter)
    process_dataset_split("test", segmenter)
    
    print("\n--- All splits processed! ---")
    print(f"New 'clean' classifier dataset is now populated in: {TARGET_DATA_DIR}/cls_clean_train")
