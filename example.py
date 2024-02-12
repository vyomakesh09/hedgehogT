# Example for testing the softmax mimicry feature map
import torch
from hedgehogTransformer.hedgehogT import HedgehogTransformer

if __name__ == "__main__":
    dim = 512
    depth = 6
    heads = 8
    dim_head = 64
    mlp_dim = 2048
    vocab_size = 10000

    model = HedgehogTransformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, vocab_size=vocab_size)
    x = torch.randint(0, vocab_size, (1, 1024))  # Dummy input
    output = model(x)
    print(f"Output shape: {output.shape}")
