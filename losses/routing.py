import torch, torch.nn as nn

class RoutingEntropyLoss(nn.Module):
    def forward(self, pi):
        # maximize entropy -> penalize -H
        eps = 1e-8
        H = -(pi * (pi.add(eps).log())).sum(dim=1).mean()
        return -H
