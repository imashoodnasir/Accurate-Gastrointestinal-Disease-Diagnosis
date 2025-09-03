import torch, torch.nn as nn, torch.nn.functional as F

class CASLoss(nn.Module):
    def __init__(self, tv_strength=0.05):
        super().__init__()
        self.tv = tv_strength

    def forward(self, A_star, mask):
        """
        A_star: (B,1,H,W) normalized (L1)
        mask:   (B,1,H,W) binary {0,1}
        """
        eps = 1e-6
        A = A_star / (A_star.sum(dim=(2,3), keepdim=True) + eps)
        M = mask / (mask.sum(dim=(2,3), keepdim=True) + eps)
        # KL(A || M)
        kl = (A * (torch.log(A + eps) - torch.log(M + eps))).sum(dim=(1,2,3)).mean()

        # sparsity + TV (Eq. 27)
        l1 = A_star.abs().mean()
        tv = (A_star[:,:,:-1,:]-A_star[:,:,1:,:]).abs().mean() + (A_star[:,:,:,:-1]-A_star[:,:,:,1:]).abs().mean()
        return kl + 0.01*l1 + self.tv*tv
