import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# We define our desired image size
IMG_SIZE = 224

def get_train_transforms():
    """
    Returns the augmentation pipeline for training.
    """
    return A.Compose([
        # We need to pad to a square before resizing
        # to maintain aspect ratio
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(
            min_height=IMG_SIZE,
            min_width=IMG_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            value=255 # Fills with black
        ),
        
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # Rotations will now look correct with the black background
        A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=255),
        
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),
        
        # Normalization for RGB images
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_val_transforms():
    """
    Returns the augmentation pipeline for validation.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(
            min_height=IMG_SIZE,
            min_width=IMG_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            value=0 # Fills with black
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
