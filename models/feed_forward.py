"""
FFN part of the transformer.
layers:
1. linear(emb_dim, 4*embedding_dim)
2. GeLU (4*embedding_dim, 4*embedding_dim)
3. linear (4*embedding_dim, embedding_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.config import *

class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 4*embedding_dim), # ([32, 256, 3072])
            nn.GELU(),
            nn.Linear(4*embedding_dim, embedding_dim) # # ([32, 256, 768])
        )

    def forward(self, input_tensor):
        return(self.layers(input_tensor))


if __name__ == "__main__":
    # Test FeedForward
    batch_size = 32
    seq_len = 256
    embedding_dim = 768

    # Create dummy input
    input_tensor = torch.randn(batch_size, seq_len, embedding_dim)
    print(f"Input shape: {input_tensor.shape}")

    # Create FFN
    ffn = FeedForward(embedding_dim)

    # Forward pass
    output = ffn(input_tensor)
    print(f"Output shape: {output.shape}")

    # Check output shape matches input
    assert output.shape == input_tensor.shape, "Output shape mismatch!"
    print("✓ Test passed! FFN working correctly.")