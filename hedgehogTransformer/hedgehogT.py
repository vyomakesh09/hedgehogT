import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class HedgehogAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim_head, dim_head),
            nn.ReLU(),
            nn.Linear(dim_head, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        spiky_k = self.mlp(k).squeeze(-1)
        spiky_weights = F.softmax(spiky_k, dim=-1).unsqueeze(1).unsqueeze(-1)
        weighted_dots = dots * spiky_weights
        weights = F.softmax(weighted_dots, dim=-1)

        out = torch.matmul(weights, v)
        out = out.squeeze(1)  # Adjusting dimensions to remove extra singleton dimension
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class HedgehogTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                HedgehogAttention(dim, heads=heads, dim_head=dim_head),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.ReLU(),
                    nn.Linear(mlp_dim, dim),
                ),
                nn.LayerNorm(dim)
            ]))

    def forward(self, x):
        for attn, norm1, ff, norm2 in self.layers:
            x = attn(x) + x
            x = norm1(x)
            x = ff(x) + x
            x = norm2(x)
        return x

