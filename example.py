# Example for testing the softmax mimicry feature map
import torch
from hedgehog.hedgehogT import HedgehogTransformer

if __name__ == "__main__":
    # Model configuration
    dim = 512  # Dimension of the model
    depth = 6  # Number of layers in the model
    heads = 8  # Number of attention heads
    dim_head = 64  # Dimension of each attention head
    mlp_dim = 2048  # Dimension of the MLP layer

    # Initialize the model
    model = HedgehogTransformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim)

    # Dummy input tensor
    x = torch.randn(1, 1024, dim)  # Batch size of 1, sequence length of 1024, feature dimension of dim

    # Forward pass
    output = model(x)

    # Print output shape
    print(f"Output shape: {output.shape}")