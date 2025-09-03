import torch, torch.nn as nn, torch.nn.functional as F
from .backbone import HierarchicalBackbone
from .attention_stages import GlobalAttention, ROIAttention, PatchAttention, fuse_attention
from .der import DynamicExpertRouting

class HMADER(nn.Module):
    def __init__(self, num_classes, dk=128, experts=3, base=64):
        super().__init__()
        self.backbone = HierarchicalBackbone(in_ch=3, base=base)
        c_global = base*4
        self.stage1 = GlobalAttention(c_in=c_global, dk=dk, num_classes=num_classes)
        self.stage2 = ROIAttention(c_global=c_global, dk=dk, num_classes=num_classes, K_rois=4)
        self.stage3 = PatchAttention(c_in=base*2, patch=32, num_classes=num_classes)
        # DER head
        fusion_dim = dk + dk + (base*2)  # concat pooled descriptors from stages
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.der = DynamicExpertRouting(dim_in=fusion_dim, num_classes=num_classes, experts=experts)

    def forward(self, x):
        Fg, Fmid, Flow = self.backbone(x)         # Fg=global (B,4C,H/8,W/8), Fmid=(B,2C,...)
        # Stage 1
        logits1, Z1, A1 = self.stage1(Fg)
        # Stage 2
        logits2, Z2_list, A2_list = self.stage2(Fg)
        # Stage 3
        logits3, Z3_list, A3_list = self.stage3(Fmid)

        # fuse attention (Eq. 30)
        A_star = fuse_attention(A1, A2_list, A3_list, out_size=x.shape[-2:])

        # build fused representation Z (Eq. 18) -> pooled tensors
        z1 = self.pool(Z1).flatten(1)
        z2 = torch.stack([self.pool(z).flatten(1) for z in Z2_list], dim=1).mean(1) if Z2_list else z1
        z3 = torch.stack([self.pool(z).flatten(1) for z in Z3_list], dim=1).mean(1) if Z3_list else z1
        z = torch.cat([z1, z2, z3], dim=1)

        yhat, pi = self.der(z)

        return {
            "logits_s1": logits1,
            "logits_s2": logits2,    # (B,K,C) or (B,C)
            "logits_s3": logits3,    # (B,M,C) or (B,C)
            "yhat": yhat,            # (B,C)
            "pi": pi,                # (B,E)
            "A_star": A_star         # (B,1,H,W), L1-normalized
        }
