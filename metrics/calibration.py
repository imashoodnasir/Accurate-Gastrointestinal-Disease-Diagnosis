import numpy as np
import torch

def ece_score(y_true, y_prob, n_bins=15):
    y_true = torch.as_tensor(y_true)
    conf, pred = y_prob.max(dim=1)
    bins = torch.linspace(0, 1, n_bins+1, device=y_prob.device)
    ece = torch.tensor(0.0, device=y_prob.device)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i+1])
        if m.any():
            acc = (pred[m] == y_true[m]).float().mean()
            avg_conf = conf[m].mean()
            ece += (m.float().mean()) * (avg_conf - acc).abs()
    return ece.item()

def brier_score(y_true, y_prob):
    y_true_oh = torch.nn.functional.one_hot(torch.as_tensor(y_true), num_classes=y_prob.shape[1]).float().to(y_prob.device)
    return ((y_prob - y_true_oh)**2).mean().item()
