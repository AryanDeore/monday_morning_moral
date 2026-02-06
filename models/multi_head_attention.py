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
from utils.config import *

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


class MultiHeadAttention():
    pass


