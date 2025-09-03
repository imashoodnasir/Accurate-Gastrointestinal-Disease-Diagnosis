import torch, torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class GlobalAttention(nn.Module):
    # Eq. (5)-(8)
    def __init__(self, c_in, dk, num_classes):
        super().__init__()
        self.q = nn.Conv2d(c_in, dk, 1)
        self.k = nn.Conv2d(c_in, dk, 1)
        self.v = nn.Conv2d(c_in, dk, 1)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dk, num_classes))
    def forward(self, Fg):
        Q = self.q(Fg); K = self.k(Fg); V = self.v(Fg)                     # (B,dk,H,W)
        B, dk, H, W = Q.shape
        q = rearrange(Q, "b d h w -> b (h w) d")
        k = rearrange(K, "b d h w -> b d (h w)")
        v = rearrange(V, "b d h w -> b (h w) d")
        attn_logits = torch.bmm(q, k) / (dk**0.5)                          # (B,HW,HW)
        A1 = torch.softmax(attn_logits, dim=-1)
        z = torch.bmm(A1, v)                                               # (B,HW,dk)
        Z1 = rearrange(z, "b (h w) d -> b d h w", h=H, w=W)
        logits = self.head(Z1)
        A1_map = A1.mean(dim=1).reshape(B, 1, H, W)                        # average over queries -> coarse map
        return logits, Z1, A1_map

class ROIAttention(nn.Module):
    # Eq. (9)-(13) — lightweight cross-attn using conv for Q; reuse K,V from global
    def __init__(self, c_global, dk, num_classes, K_rois=4):
        super().__init__()
        self.K_rois = K_rois
        self.q = nn.Conv2d(c_global, dk, 1)
        self.cls = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dk, num_classes))

    def forward(self, Fg):
        B, C, H, W = Fg.shape
        # propose ROIs as top-K windows from global activation (cheap heuristic)
        heat = F.adaptive_avg_pool2d(Fg, (H, W)).sum(1, keepdim=True)     # (B,1,H,W)
        windows = []
        win = max(4, min(H,W)//4)
        stride = win//2
        for y in range(0, H-win+1, stride):
            for x in range(0, W-win+1, stride):
                windows.append((y,x,win))
        # score windows, pick top-K
        scores = []
        for (y,x,w) in windows:
            scores.append(heat[:,:,y:y+w, x:x+w].mean(dim=(2,3)))
        scores = torch.stack(scores, dim=-1)  # (B,1,Nw)
        topk = torch.topk(scores, k=min(self.K_rois, scores.shape[-1]), dim=-1).indices[0,0].tolist()

        A2_maps, Z2_list, logits_list = [], [], []
        K = Fg; V = Fg
        Q = self.q(Fg)
        dk = Q.shape[1]
        q = rearrange(Q, "b d h w -> b (h w) d")
        k = rearrange(K, "b d h w -> b d (h w)")
        v = rearrange(V, "b d h w -> b (h w) d")
        attn = torch.softmax(torch.bmm(q, k) / (dk**0.5), dim=-1)          # (B,HW,HW)
        z = torch.bmm(attn, v)                                             # (B,HW,d)
        Z = rearrange(z, "b (h w) d -> b d h w", h=H, w=W)

        for idx in topk:
            y,x,w = windows[idx]
            z_roi = Z[:,:,y:y+w, x:x+w]
            logits = self.cls(z_roi)
            A2_maps.append(attn.mean(1).reshape(B,1,H,W)[:,:,y:y+w,x:x+w]) # crop
            Z2_list.append(z_roi)
            logits_list.append(logits)

        if len(Z2_list)==0:
            # fallback: full map
            logits = self.cls(Z)
            return logits, [Z], [attn.mean(1).reshape(B,1,H,W)]
        return torch.stack(logits_list, dim=1), Z2_list, A2_maps

class PatchAttention(nn.Module):
    # Eq. (14)-(17) — neighborhood weighting
    def __init__(self, c_in, patch=32, num_classes=10):
        super().__init__()
        self.patch = patch
        self.enc = nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, padding=1), nn.GELU(),
            nn.Conv2d(c_in, c_in, 3, padding=1), nn.GELU()
        )
        self.cls = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c_in, num_classes))

    def forward(self, Fmid):
        B,C,H,W = Fmid.shape
        p = self.patch
        A3_maps, Z3_list, logits_list = [], [], []
        for y in range(0, H, p):
            for x in range(0, W, p):
                ph = Fmid[:,:,y:min(y+p,H), x:min(x+p,W)]
                h = self.enc(ph)
                # simple local affinity (avg of neighborhood)
                nbr = h.mean(dim=(2,3), keepdim=True)                      # (B,C,1,1)
                a = torch.sigmoid(nbr)                                     # Eq. (15) style
                z = a * h                                                  # Eq. (16)
                logit = self.cls(z)
                A3_maps.append(a.expand_as(h))
                Z3_list.append(z)
                logits_list.append(logit)
        if len(Z3_list)==0:
            # fallback: global
            h = self.enc(Fmid)
            a = torch.sigmoid(h.mean(dim=(2,3), keepdim=True))
            z = a * h
            return self.cls(z), [z], [a]
        return torch.stack(logits_list, dim=1), Z3_list, A3_maps

def fuse_attention(A1, A2_list, A3_list, beta1=0.5, beta2=0.3, beta3=0.2, out_size=None):
    B = A1.shape[0]
    if out_size is None:
        out_size = A1.shape[-2:]
    A = beta1*F.interpolate(A1, size=out_size, mode="bilinear", align_corners=False)
    if A2_list:
        A2_sum = sum([F.interpolate(a, size=out_size, mode="bilinear", align_corners=False) for a in A2_list])
        A = A + beta2*A2_sum
    if A3_list:
        A3_sum = sum([F.interpolate(a.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True), size=out_size, mode="bilinear", align_corners=False) for a in A3_list])
        A = A + beta3*A3_sum
    A = torch.relu(A)
    A = A / (A.sum(dim=(2,3), keepdim=True) + 1e-6)
    return A
