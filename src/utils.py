import os
import random
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp

def set_seed(seed=42):
    """Sets seed for reproducibility across torch, numpy, and python random."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """
    Calculates Intersection over Union (Jaccard Index).
    
    Math:
        IoU = Intersection / Union
        IoU = TP / (TP + FP + FN)
        
    Args:
        pred_mask: Logits from model
        true_mask: Ground truth binary mask
    """
    # Apply sigmoid activation to get probabilities: σ(x) = 1 / (1 + e^(-x))
    pred_mask = torch.sigmoid(pred_mask)
    
    # Binarize predictions based on threshold
    pred_mask = (pred_mask > threshold).float()
    
    # Flatten to 1D vectors for set operations
    pred_mask = pred_mask.reshape(-1)
    true_mask = true_mask.reshape(-1)
    
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    
    # Add epsilon (1e-6) to prevent division by zero
    return (intersection + 1e-6) / (union + 1e-6)

class SegmentationLoss(nn.Module):
    """
    Compound Loss Function: Dice Loss + Binary Cross Entropy (BCE).
    
    Mathematical Formulation:
        L_total = L_Dice + L_BCE
        
    1. Dice Loss:
       L_Dice = 1 - (2 * |A ∩ B|) / (|A| + |B|)
       Optimizes for overlap, handles class imbalance well.
       
    2. BCE Loss:
       L_BCE = -1/N * Σ [y * log(p) + (1-y) * log(1-p)]
       Optimizes pixel-wise classification accuracy.
    """
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.bce = smp.losses.SoftBCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.dice(pred, target) + self.bce(pred, target)
