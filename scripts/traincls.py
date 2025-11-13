# scripts/traincls.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm 
import cv2
import os
import numpy as np
from tqdm import tqdm
import sys

# --- [FIX] Path Correction for Colab ---
PROJECT_ROOT = "../"
sys.path.insert(0, PROJECT_ROOT)
# --- End of Fix ---

try:
    from utils.augmentations import get_train_transforms, get_val_transforms
except ImportError:
    print("Error: Could not import augmentations.py. Make sure 'scripts/utils/augmentations.py' exists (Cell 4).")
    sys.exit(1)


# --- Configuration ---
# [NEW] Point to the 'cls_clean' dataset folders
DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "cls_clean_train") 
VAL_DATA_PATH = os.path.join(DATA_DIR, "cls_clean_valid")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 2  # healthy, early_blight
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4

# --- Albumentations Dataset Wrapper ---
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # ImageFolder loads images as PIL (RGB) by default.
        image, label = self.dataset[idx]
        
        # Convert PIL (RGB) Image to numpy array (RGB)
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

# --- Main Training Function ---
def train():
    print(f"--- Starting Classifier Training ---")
    print(f"Model: {MODEL_NAME}, Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    if not os.path.exists(TRAIN_DATA_PATH) or not os.listdir(TRAIN_DATA_PATH):
        print(f"ERROR: Training data not found or empty at {TRAIN_DATA_PATH}")
        print("Please run the 'build_clean_classifier_dataset.py' script first (Cell 6).")
        return
        
    base_train_dataset = ImageFolder(root=TRAIN_DATA_PATH)
    base_val_dataset = ImageFolder(root=VAL_DATA_PATH)
    
    print(f"Found {len(base_train_dataset)} training images.")
    print(f"Found {len(base_val_dataset)} validation images.")
    print(f"Classes: {base_train_dataset.classes}")
    
    # --- [NEW] Class Balancing ---
    class_counts = np.bincount(base_train_dataset.targets)
    num_samples = len(base_train_dataset)
    
    if class_counts.size == 0:
        print(f"Error: No data loaded from {TRAIN_DATA_PATH}")
        return

    class_weights = num_samples / (len(class_counts) * class_counts)
    sample_weights = [class_weights[label] for label in base_train_dataset.targets]
    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(sample_weights_tensor, num_samples, replacement=True)
    print(f"Balancing classes. Weights: {class_weights}")
    # --- [END NEW CODE] ---

    train_dataset = AlbumentationsDataset(
        dataset=base_train_dataset,
        transform=get_train_transforms()
    )
    val_dataset = AlbumentationsDataset(
        dataset=base_val_dataset,
        transform=get_val_transforms()
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # Sampler handles shuffling
        num_workers=2,
        sampler=sampler
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # --- 2. Create Model ---
    model = timm.create_model(MODEL_NAME, pretrained=True)
    num_features = model.get_classifier().in_features
    model.classifier = (nn.Linear(num_features, NUM_CLASSES))
    
    model = model.to(device)

    # --- 3. Define Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. Training Loop ---
    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, "best_cls_weights.pt")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)
            
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_corrects.double() / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} with Val Acc: {best_val_acc:.4f}")

    print("--- Training Complete ---")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best model weights saved at: {best_model_path}")

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    train()

print("File 'scripts/traincls.py' created.")
