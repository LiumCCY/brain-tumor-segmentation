import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F

def dice_coefficient(pred_mask, true_mask,threshold):
    smooth = 0.00001
    pred_mask_binary = torch.where(pred_mask >= threshold, torch.ones_like(pred_mask), torch.zeros_like(pred_mask))
    true_mask_binary = torch.where(true_mask >= threshold, torch.ones_like(true_mask), torch.zeros_like(true_mask))
    pred_mask_binary  = pred_mask_binary .contiguous().view(-1)
    true_mask_binary = true_mask_binary.contiguous().view(-1)
    intersection = (pred_mask_binary  * true_mask_binary).sum()
    dice = (2.0 * intersection + smooth) / (pred_mask_binary .sum() + true_mask_binary.sum() + smooth)
    return dice

def f_score(y_true, y_pred, beta=1, eps=1e-7, threshold=None):
    
    if threshold is not None:
        y_pred = (y_pred > threshold).float()

    tp = torch.sum(y_true * y_pred)
    fp = torch.sum(y_pred) - tp
    fn = torch.sum(y_true) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + eps)
    return f_score


