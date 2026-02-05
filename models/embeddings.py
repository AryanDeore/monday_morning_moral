"""
accepts input_idw and creates embeddings with positon encoded
# simulate input_ids = torch.tensor(torch.randint(low=0, high=50256, size=(32, 256)))
input_ids = [50256, 2163, ...] 32 X 256
"""

from random import randint
import torch
import torch.nn as nn
# from utils.config import vocab_size, embedding_dim, context_length

class Embeddings(nn.Module):
    """
    Generate position encoded embeddings for a token ID
    """

    def __init__(self, vocab_size, context_length, embedding_dim, dropout_rate):
        """
        Declare the layers for embedding.
        """
        super().__init__()
        # Layers
        # input is text input: [32, 256]| its size: [50257, 768]| output: [32, 256, 768]
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # input is positional index: [256]| its size: [256, 768]| output: [256, 768]
        self.pos_embedding = nn.Embedding(context_length, embedding_dim)

        self.dropout = nn.Dropout(dropout_rate) # use a smaller dropout like 5% for input layers.

    def forward(self, input_ids):
        """
        Takes token IDs and returns combined embeddings
        """
        tok_embeds = self.token_embedding(input_ids)

        pos = torch.arange((input_ids.shape[1]))
        pos_embeds = self.pos_embedding(pos)

        combined = tok_embeds + pos_embeds

        final_embedding = self.dropout(combined)

        return final_embedding
