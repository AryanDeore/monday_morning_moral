"""
1. create Q, K, V
2. calculate attention score
3. mask the attention score
4. normalise the attention scores to find attention weights
5. context_vec = attention_weights @ v
token embeddings are the input
input: [batch, seq_len, embedding_dim]

simulate token_embedding = torch.rand(32, 256, 768)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, num_heads, embedding_dim, context_length, dropout_rate):
        super().__init__()
        self.q = nn.Linear(embedding_dim, embedding_dim//num_heads)
        self.k = nn.Linear(embedding_dim, embedding_dim//num_heads)
        self.v = nn.Linear(embedding_dim, embedding_dim//num_heads)
        self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length))) # storing the mask in buffer because it is a fixed shape and value
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, token_embedding):
        q = self.q(token_embedding) # ([32, 256, 64])
        k = self.k(token_embedding) 
        v = self.v(token_embedding)

        k_transposed = torch.transpose(k, 2, 1) # ([32, 64, 256])
        # [32, 256, 64] x [32, 64, 256] = [32, 256, 256]
        scores = q@k_transposed # ([32, 256, 256])
        scores = scores/(q.shape[-1]**0.5) # ([32, 256, 256])

        _, T, _ = scores.shape
        mask = self.mask[:T, :T] # ([256, 256])
        scores = scores.masked_fill(mask == 0, float('-inf')) # ([32, 256, 256])

        attention_weights = F.softmax(scores, dim = -1) # ([32, 256, 256])

        context_vec = attention_weights @ v  # [32, 256, 256] @ [32, 256, 64] = [32, 256, 64]

        context_vec = self.dropout(context_vec)

        return context_vec


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embedding_dim, context_length, dropout_rate):
        super().__init__()
        assert embedding_dim % num_heads == 0, \
            f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"

        self.heads = nn.ModuleList([ Head(num_heads, embedding_dim, context_length, dropout_rate) for _ in range(num_heads)])
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, token_embedding):
        head_outputs = [head(token_embedding) for head in self.heads] # is a list of 12 tensors
        concatenated = torch.cat(head_outputs, dim=-1)  # [32, 256, 768]
        output = self.out_proj(concatenated)

        return output


if __name__ == "__main__":
    # Create dummy input
    batch_size = 32
    seq_len = 256
    embedding_dim = 768
    context_length = 1024
    num_heads = 12
    dropout_rate = 0.1

    # Create dummy token embedding
    token_embedding = torch.randn(batch_size, seq_len, embedding_dim)
    print(f"Input shape: {token_embedding.shape}")

    # Create MHA module
    mha = MultiHeadAttention(num_heads, embedding_dim, context_length, dropout_rate)

    # Forward pass
    output = mha(token_embedding)
    print(f"Output shape: {output.shape}")

    # Check if output shape is correct
    assert output.shape == token_embedding.shape, "Output shape mismatch!"
    print("✓ Test passed! Output shape is correct.")