import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import binary_fill_holes
import torch

class LeafSegmenter:
    def __init__(self, model_path='yolov8n-seg.pt', target_size=(640, 640)):
        """
        Initialize the leaf segmentation pipeline
        """
        self.model = YOLO(model_path)
        self.target_size = target_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def preprocess_image_from_frame(self, original_image):
        """
        Preprocesses a raw OpenCV frame (H, W, C)
        """
        h_orig, w_orig = original_image.shape[:2]
        target_w, target_h = self.target_size

        scale = min(target_w / w_orig, target_h / h_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)

        resized_image = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create a letterboxed image
        processed_image = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        processed_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

        scale_factors = {
            'scale': scale,
            'x_offset': x_offset,
            'y_offset': y_offset,
            'new_w': new_w,
            'new_h': new_h,
            'orig_w': w_orig,
            'orig_h': h_orig
        }

        return processed_image, scale_factors

    def preprocess_image(self, image_path):
        """Load, resize, and letterbox the image from a path."""
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"Could not read image from {image_path}")
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        processed_image, scale_factors = self.preprocess_image_from_frame(original_image)
        return processed_image, original_image, scale_factors

    def run_yolov8_segmentation(self, processed_image):
        """Run YOLOv8 segmentation model."""
        results = self.model(processed_image, verbose=False, device=self.device)
        return results[0]

    def extract_leaf_masks(self, results, scale_factors):
        """Extract masks and bounding boxes from YOLOv8 results."""
        leaf_masks, leaf_boxes = [], []

        if results.masks is None:
            return leaf_masks, leaf_boxes

        orig_h, orig_w = scale_factors['orig_h'], scale_factors['orig_w']

        for i, mask in enumerate(results.masks.data):
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            box = results.boxes.xyxy[i].cpu().numpy()
            scale = scale_factors['scale']
            x_offset, y_offset = scale_factors['x_offset'], scale_factors['y_offset']

            box_orig = np.array([
                (box[0] - x_offset) / scale,
                (box[1] - y_offset) / scale,
                (box[2] - x_offset) / scale,
                (box[3] - y_offset) / scale
            ]).astype(int)

            box_orig[0] = max(0, box_orig[0])
            box_orig[1] = max(0, box_orig[1])
            box_orig[2] = min(orig_w, box_orig[2])
            box_orig[3] = min(orig_h, box_orig[3])

            leaf_masks.append(mask_resized)
            leaf_boxes.append(box_orig) # [x1, y1, x2, y2]

        return leaf_masks, leaf_boxes
    
    # --- [NEWLY ADDED] ---
    
    def post_process_mask(self, mask, min_area=500):
        """Clean, fill holes, and remove small components."""
        binary_mask = (mask > 0.5).astype(np.uint8)

        # Fill holes using SciPy
        filled = binary_fill_holes(binary_mask).astype(np.uint8) * 255

        # Remove small objects
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled)
        final_mask = np.zeros_like(filled)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                final_mask[labels == i] = 255

        return final_mask

    def apply_mask_and_crop(self, original_image, mask, box, background_type='white'):
        """Apply mask to the leaf and crop it."""
        x1, y1, x2, y2 = box
        
        # Ensure box coordinates are valid
        if x1 >= x2 or y1 >= y2:
            return None
            
        cropped_image = original_image[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]

        if cropped_image.size == 0 or cropped_mask.size == 0:
            return None
        
        # Ensure mask is 3-channel for bitwise operations
        mask_3d = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR)

        if background_type == 'transparent':
            output = np.zeros((*cropped_image.shape[:2], 4), dtype=np.uint8)
            output[..., :3] = cropped_image
            output[..., 3] = cropped_mask
        
        elif background_type == 'white':
            background = np.full_like(cropped_image, 255, dtype=np.uint8)
            output = cv2.bitwise_and(cropped_image, mask_3d)
            output += cv2.bitwise_and(background, cv2.bitwise_not(mask_3d))

        else: # Default to black background
            background = np.zeros_like(cropped_image, dtype=np.uint8)
            output = cv2.bitwise_and(cropped_image, mask_3d)
            output += cv2.bitwise_and(background, cv2.bitwise_not(mask_3d))
            
        return output

print("Utility scripts written to disk.")
