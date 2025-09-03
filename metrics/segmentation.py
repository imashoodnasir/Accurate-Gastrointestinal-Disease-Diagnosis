import torch

def dice_score(pred, target, eps=1e-6):
    # pred: (B,1,H,W) logits or probs
    if pred.dtype.is_floating_point:
        pred = (pred > 0.5).float()
    inter = (pred*target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return (2*inter/denom).mean()

def iou_score(pred, target, eps=1e-6):
    if pred.dtype.is_floating_point:
        pred = (pred > 0.5).float()
    inter = (pred*target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter + eps
    return (inter/union).mean()
