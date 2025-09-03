import torch, torch.nn as nn, torch.nn.functional as F

class Gating(nn.Module):
    def __init__(self, dim_in, experts=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_in//2), nn.GELU(),
            nn.Linear(dim_in//2, experts)
        )
        self.experts = experts
    def forward(self, z):
        logits = self.mlp(z)
        return torch.softmax(logits, dim=-1)

class Expert(nn.Module):
    def __init__(self, dim_in, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_in//2), nn.GELU(),
            nn.Linear(dim_in//2, num_classes)
        )
    def forward(self, z): return self.net(z)

class DynamicExpertRouting(nn.Module):
    # Eq. (18)-(22)
    def __init__(self, dim_in, num_classes, experts=3):
        super().__init__()
        self.gate = Gating(dim_in, experts)
        self.experts = nn.ModuleList([Expert(dim_in, num_classes) for _ in range(experts)])
    def forward(self, z):
        # z: (B, D)
        pi = self.gate(z)                              # (B,E)
        outs = torch.stack([e(z) for e in self.experts], dim=1)  # (B,E,C)
        probs = torch.softmax(outs, dim=-1)
        yhat = (pi.unsqueeze(-1) * probs).sum(dim=1)   # mixture
        return yhat, pi
