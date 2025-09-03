import torch, torch.nn as nn
from einops import rearrange

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, c, k=3, d=1):
        super().__init__()
        p = d * (k-1)//2
        self.dw = nn.Conv2d(c, c, k, padding=p, dilation=d, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.GELU()
    def forward(self, x): 
        x = self.dw(x); x = self.pw(x); x = self.bn(x); return self.act(x)

class DilatedResidualBlock(nn.Module):
    def __init__(self, c, d_rates=(1,2,3)):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(c, k=3, d=d) for d in d_rates])
        self.proj = nn.Conv2d(c, c, 1)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.GELU()
    def forward(self, x):
        out = sum([m(x) for m in self.convs]) / len(self.convs)
        out = self.proj(out)
        out = self.bn(out + x)
        return self.act(out)

class TinyTransformerBlock(nn.Module):
    def __init__(self, c, heads=4, mlp_ratio=4.0, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, dropout=attn_drop, batch_first=True)
        self.drop1 = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(c)
        self.mlp = nn.Sequential(
            nn.Linear(c, int(c*mlp_ratio)), nn.GELU(),
            nn.Linear(int(c*mlp_ratio), c)
        )
    def forward(self, x):  # x: (B,C,H,W)
        B,C,H,W = x.shape
        z = rearrange(x, "b c h w -> b (h w) c")
        z = self.norm1(z)
        z2, _ = self.attn(z, z, z)
        z = z + self.drop1(z2)
        z = z + self.mlp(self.norm2(z))
        return rearrange(z, "b (h w) c -> b c h w", h=H, w=W)

class HierarchicalBackbone(nn.Module):
    """
    Produces F(G), F(R) (pooled ROIs), F(P) (patch tokens).
    """
    def
