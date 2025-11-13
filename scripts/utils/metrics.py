# scripts/utils/metrics.py

import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def save_classification_report(y_true, y_pred, target_names, save_path):
    """
    Saves the classification report (precision, recall, f1-score) to a text file.
    """
    report = classification_report(y_true, y_pred, target_names=target_names)
    
    with open(os.path.join(save_path, "classification_report.txt"), 'w') as f:
        f.write("--- Classification Report ---\n\n")
        f.write(report)
    
    print("\n--- Classification Report ---")
    print(report)
    print(f"Report saved to {save_path}")

def save_confusion_matrix(y_true, y_pred, target_names, save_path):
    """
    Generates and saves a heatmap of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=target_names, 
        yticklabels=target_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    matrix_path = os.path.join(save_path, "confusion_matrix.png")
    plt.savefig(matrix_path)
    print(f"Confusion matrix saved to {matrix_path}")
    plt.close()