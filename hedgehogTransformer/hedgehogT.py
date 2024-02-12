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
        
        # Squeeze the unnecessary singleton dimension if present
        # Check if there's a singleton dimension at index 2 and remove it
        #if x.shape[2] == 1:
         #   x = x.squeeze(2)  # This adjusts x from [8, 511, 1, 512] to [8, 511, 512]
        
        if x.dim() > 3:  # Adjust for the edge case where there's a singleton dim
            x = x.squeeze(2)

        print(f'x.shape in before rearrange the forward attention class {x.shape}')
        b, n, _ = x.shape  # Now this should correctly unpack the dimensions
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h=h) for t in qkv]

        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        print(f'out.shape after tourch matmul and rearrage {out.shape}')
        # out = self.to_out(out)
        # out = out.reshape(b, n, -1)
        print(f'x.shape after the rearrage in the forward attention class {x.shape}')

        # print(out.shape, x.shape)
        if out.shape != x.shape:
            raise ValueError('output shape of attention mechanism\
                             does not match input shape,\
                             cannot perform residual connection')                           
        if out.shape == x.shape:
            x = out + x
        else:
            raise ValueError('Shape mismatch in residual connection')
        return x

class HedgehogTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
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
        x = self.embedding(x)
        print(x.shape)

        #if x.shape[2] == 1:
         #    x = x.squeeze(2)
        

        for attn, norm1, ff, norm2 in self.layers:
            # print(attn(x).shape, x.shape)
            # x_squeeezed = x.squeeze(2)
            print(f'this is in the forward class of HT {attn(x).shape, x.shape}')
            x = attn(x) + x
            x = norm1(x)
            x = ff(x) + x
            x = norm2(x)
        return x
