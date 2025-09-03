import torch, torch.nn as nn, torch.nn.functional as F

class FaithfulnessLoss(nn.Module):
    def __init__(self, keep_ratio=0.2):
        super().__init__()
        self.keep = keep_ratio

    @torch.no_grad()
    def _topk_mask(self, A_star, keep):
        B,_,H,W = A_star.shape
        flat = A_star.view(B,-1)
        k = (flat.shape[1]*keep)
        k = max(1, int(k))
        idx = flat.topk(k, dim=1).indices
        keep_mask = torch.zeros_like(flat)
        keep_mask.scatter_(1, idx, 1.0)
        return keep_mask.view(B,1,H,W)

    def forward(self, model, x, y_true, A_star):
        """
        model: callable returning dict with 'yhat'
        x: (B,3,H,W), y_true: (B,)
        A_star: (B,1,H,W)
        """
        B = x.size(0)
        # insertion: keep only top-k regions
        k_mask = self._topk_mask(A_star, self.keep)
        x_ins = x * k_mask
        y_ins = model(x_ins)["yhat"].gather(1, y_true.view(-1,1)).squeeze(1)

        # deletion: remove top-k regions
        x_del = x * (1 - k_mask)
        y_del = model(x_del)["yhat"].gather(1, y_true.view(-1,1)).squeeze(1)

        loss = (1 - y_ins).mean() + y_del.mean()  # Eq. (26)
        return loss
