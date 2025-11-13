# Cell 5: Define Helper Functions & Configuration

import cv2
import torch
import torch.nn as nn
import timm
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt # Import matplotlib

# --- Path Correction (for Colab) ---
PROJECT_ROOT = "/content"
sys.path.insert(0, PROJECT_ROOT)
# --- End of Path Correction ---

# Import our two-stage models
from scripts.utils.segmenter import LeafSegmenter
from scripts.utils.augmentations import get_val_transforms

# --- Configuration ---
# Stage 1: Segmentation (yolov8n-seg.pt will be auto-downloaded)
SEG_MODEL_PATH = "yolov8n-seg.pt"

# Stage 2: Classification (Your model)
CLS_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CLS_MODEL_WEIGHTS = os.path.join(CLS_MODEL_DIR, "best_cls_weights_final.pt")
CLS_MODEL_NAME = "efficientnet_b0"
CLS_IMG_SIZE = 224 # Your classifier's input size

# IMPORTANT: Hardcoding classes as we don't have the data dir structure in colab
# Make sure this matches the order from your 'test_cls.py' output
CLS_CLASSES = ['early_blight', 'healthy']
NUM_CLASSES = len(CLS_CLASSES)

# --- Helper function to convert OpenCV image to Pyplot format ---
def cv2_to_pyplot(image):
    """Converts an OpenCV BGR image to RGB for Pyplot display."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- 1. Load Classification Model ---
def load_classifier(weights_path, device):
    print(f"Loading classifier from {weights_path}...")
    model = timm.create_model(CLS_MODEL_NAME, pretrained=False)
    num_features = model.get_classifier().in_features
    model.classifier = (nn.Linear(num_features, NUM_CLASSES))

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Classifier loaded successfully.")
    return model

# --- 2. Classifier Inference Function ---
def classify_crops(crops, model, device, transforms):
    if not crops:
        return []
    transformed_crops = []
    for crop in crops:
        # [FIX] Match the training bug fix
        transformed = transforms(image=crop)['image']
        transformed_crops.append(transformed)

    batch_size = 32
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(transformed_crops), batch_size):
            batch = torch.stack(transformed_crops[i:i+batch_size]).to(device)
            outputs = model(batch)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_preds_batch = torch.max(probs, 1)
            all_preds.extend(list(zip(top_preds_batch.cpu().numpy(), top_probs.cpu().numpy())))
    return all_preds

# --- 3. Main Image Processing Function ---
def process_image(image_path, output_path, seg_model, cls_model, cls_transforms, device):
    print(f"Processing image: {image_path}")

    # 2. Load and preprocess image for segmentation
    try:
        processed_seg_img, original_img_rgb, scale_factors = seg_model.preprocess_image(image_path)
    except FileNotFoundError as e:
        print(e)
        return

    # --- Display the preprocessed segmented image ---
    plt.figure(figsize=(8, 8))
    plt.imshow(processed_seg_img) # processed_seg_img is already RGB
    plt.title("Preprocessed Image for Segmentation (Letterboxed)")
    plt.axis('off')
    plt.show()
    # --- End of Display ---


    original_img_bgr = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR)

    # 3. Stage 1: Find Leaves (Segmentation)
    results = seg_model.run_yolov8_segmentation(processed_seg_img)
    leaf_masks, leaf_boxes = seg_model.extract_leaf_masks(results, scale_factors)

    if not leaf_boxes:
        print("No leaves detected.")
        cv2.imwrite(output_path, original_img_bgr)
        return

    # 4. Prepare crops for classifier
    crops_for_classification = []
    boxes_for_drawing = []

    for box in leaf_boxes:
        x1, y1, x2, y2 = box
        crop = original_img_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            crop_resized = cv2.resize(crop, (CLS_IMG_SIZE, CLS_IMG_SIZE), interpolation=cv2.INTER_AREA)
            crops_for_classification.append(crop_resized)
            boxes_for_drawing.append(box)

    # 5. Stage 2: Classify Leaves
    predictions = classify_crops(crops_for_classification, cls_model, device, cls_transforms)

    # 6. Draw results
    frame = original_img_bgr.copy()
    for (box, pred) in zip(boxes_for_drawing, predictions):
        class_idx, confidence = pred
        class_name = CLS_CLASSES[class_idx]
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if class_name == 'healthy' else (0, 0, 255) # Green / Red
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{class_name}: {confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imwrite(output_path, frame)
    print(f"Processing complete. Output image saved to: {output_path}")

# --- 4. Main Video Processing Function ---
def process_video(video_path, output_path, seg_model, cls_model, cls_transforms, device):
    print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_seg_img, scale_factors = seg_model.preprocess_image_from_frame(frame_rgb)

        results = seg_model.run_yolov8_segmentation(processed_seg_img)
        leaf_masks, leaf_boxes = seg_model.extract_leaf_masks(results, scale_factors)

        if not leaf_boxes:
            out.write(frame)
            continue

        crops_for_classification = []
        boxes_for_drawing = []
        for box in leaf_boxes:
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crop_resized = cv2.resize(crop, (CLS_IMG_SIZE, CLS_IMG_SIZE), interpolation=cv2.INTER_AREA)
                crops_for_classification.append(crop_resized)
                boxes_for_drawing.append(box)

        predictions = classify_crops(crops_for_classification, cls_model, device, cls_transforms)

        for (box, pred) in zip(boxes_for_drawing, predictions):
            class_idx, confidence = pred
            class_name = CLS_CLASSES[class_idx]

            x1, y1, x2, y2 = box
            color = (0, 255, 0) if class_name == 'healthy' else (0, 0, 255) # Green / Red

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{class_name}: {confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nProcessing complete. Output video saved to: {output_path}")