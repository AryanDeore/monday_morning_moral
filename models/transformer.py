"""
The flow is:

  input ([32, 256, 768])
    ↓
  [LayerNorm]
    ↓
  [MultiHeadAttention]
    ↓
  [Dropout]
    ↓
  [Add to input] ← Residual connection
    ↓
  [LayerNorm]
    ↓
  [FeedForward]
    ↓
  [Dropout]
    ↓
  [Add to previous] ← Residual connection
    ↓
  output
"""
import torch
import torch.nn as nn
from models import multi_head_attention, feed_forward

class Transformer(nn.Module):
    def __init__(self, context_length, embedding_dim, num_heads, dropout_rate):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm([embedding_dim])
        self.mha = multi_head_attention.MultiHeadAttention(num_heads, embedding_dim, context_length, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm([embedding_dim])
        self.feed_forward = feed_forward.FeedForward(embedding_dim)


    def forward(self, input_tokens):
        shortcut = input_tokens
        input_tokens = self.layer_norm1(input_tokens)
        input_tokens = self.mha(input_tokens)
        input_tokens = self.dropout(input_tokens)
        input_tokens = input_tokens + shortcut

        # second set: define shortcut -> layernorm2 -> feed forward -> dropout-> add shourtcut
        shortcut = input_tokens
        input_tokens = self.layer_norm2(input_tokens)
        input_tokens = self.feed_forward(input_tokens)
        input_tokens = self.dropout(input_tokens)
        input_tokens = input_tokens + shortcut

        return input_tokens


if __name__ == "__main__":
    # Test Transformer Block
    batch_size = 32
    seq_len = 256
    embedding_dim = 768
    context_length = 1024
    num_heads = 12
    dropout_rate = 0.1

    # Create dummy input
    input_tokens = torch.randn(batch_size, seq_len, embedding_dim)
    print(f"Input shape: {input_tokens.shape}")

    # Create Transformer Block
    transformer_block = Transformer(context_length, embedding_dim, num_heads, dropout_rate)

    # Forward pass
    output = transformer_block(input_tokens)
    print(f"Output shape: {output.shape}")

    # Check output shape matches input
    assert output.shape == input_tokens.shape, "Output shape mismatch!"
    print("✓ Test passed! Transformer Block working correctly.")
