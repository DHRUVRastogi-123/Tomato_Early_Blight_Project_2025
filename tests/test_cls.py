# tests/test_cls.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import timm
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm

# --- Path Correction ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
# --- End of Path Correction ---

# Import our helper scripts
# We are now importing the *new* get_val_transforms
# which includes the padding and resizing logic.
from scripts.utils.augmentations import get_val_transforms 
from scripts.utils.metrics import save_classification_report, save_confusion_matrix

# --- Absolute File Paths ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "best_cls_weights_final.pt")
# --- End of Absolute File Paths ---

BATCH_SIZE = 32
MODEL_NAME = "efficientnet_b0"

# --- Albumentations Dataset Wrapper (Updated) ---
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # ImageFolder loads as PIL (RGB)
        image, label = self.dataset[idx]
        
        # --- [FIX] ---
        # Convert PIL (RGB) to Numpy (RGB)
        # This matches the new traincls.py
        image = np.array(image)
        # --- [END FIX] ---
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label

def test_model():
    print(f"--- Starting Model Evaluation on Test Set ---")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Test Data ---
    # --- [CHANGE] ---
    # Point to the new 'clean' dataset
    test_data_path = os.path.join(DATA_DIR, "cls_clean_test")
    # --- [END CHANGE] ---
    
    if not os.path.exists(test_data_path) or not os.listdir(test_data_path):
        print(f"Error: Test data not found or empty at {test_data_path}")
        print("Please make sure you have run 'build_clean_classifier_dataset.py' (Cell 6).")
        return

    # Load base dataset to get class names
    base_test_dataset = ImageFolder(root=test_data_path)
    class_names = base_test_dataset.classes
    print(f"Found {len(base_test_dataset)} test images.")
    print(f"Classes: {class_names}")

    # Create wrapped dataset with validation (test) transforms
    # This will use the new get_val_transforms (with padding)
    test_dataset = AlbumentationsDataset(
        dataset=base_test_dataset,
        transform=get_val_transforms()
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        # Set num_workers to 0 for local Windows stability
        num_workers=0 
    )

    # --- 2. Load Trained Model ---
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: Model weights not found at {MODEL_WEIGHTS}")
        print("Please make sure your new model is trained and saved.")
        return
        
    model = timm.create_model(MODEL_NAME, pretrained=False)
    num_features = model.get_classifier().in_features
    
    # --- [CHANGE] ---
    # Use set_classifier to match traincls.py
    model.classifier = (nn.Linear(num_features, len(class_names)))
    # --- [END CHANGE] ---
    
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model = model.to(device)
    model.eval()

    # --- 3. Run Inference ---
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating Test Set"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 4. Generate Reports ---
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    print("\nGenerating reports...")
    save_classification_report(all_labels, all_preds, class_names, REPORT_DIR)
    save_confusion_matrix(all_labels, all_preds, class_names, REPORT_DIR)
    
    print("\n--- Evaluation Complete ---")
    print(f"Reports saved in: {REPORT_DIR}")

if __name__ == "__main__":
    test_model()