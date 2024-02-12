import torch
from hedgehogTransformer import HedgehogTransformer


def test_hedgehog_transformer():
    # Model configuration
    dim = 512  # Dimension of the model
    depth = 6  # Number of layers in the model
    heads = 8  # Number of attention heads
    dim_head = 64  # Dimension of each attention head
    mlp_dim = 2048  # Dimension of the MLP layer

    # Initialize the model
    model = HedgehogTransformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim)
    model.eval()  # Set the model to evaluation mode

    # Generate a batch of dummy input data
    batch_size = 1  # Number of samples in the batch
    seq_length = 1024  # Length of the input sequence
    x = torch.randn(batch_size, seq_length, dim)  # Randomly generated input data

    # Pass the input data through the model
    with torch.no_grad():  # Ensure gradients are not computed in this operation
        output = model(x)

    # Print output shape to verify the model's forward pass
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    test_hedgehog_transformer()
