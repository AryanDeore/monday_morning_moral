"""
accepts input_ids and creates embeddings with positon encoded
# simulate input_ids = torch.tensor(torch.randint(low=0, high=50256, size=(32, 256)))
input_ids = [50256, 2163, ...] 32 X 256
output: embedding: [32, 256, 768]
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

        pos = torch.arange((input_ids.shape[1]), device=input_ids.device)
        pos_embeds = self.pos_embedding(pos)

        combined = tok_embeds + pos_embeds

        final_embedding = self.dropout(combined)

        return final_embedding


if __name__ == "__main__":
    # Test Embeddings
    batch_size = 32
    seq_len = 256
    vocab_size = 50257
    embedding_dim = 768
    context_length = 1024
    dropout_rate = 0.1

    # Create dummy token IDs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input token IDs shape: {input_ids.shape}")

    # Create Embeddings module
    embeddings = Embeddings(vocab_size, context_length, embedding_dim, dropout_rate)

    # Forward pass
    output = embeddings(input_ids)
    print(f"Embeddings output shape: {output.shape}")

    # Check output shape
    assert output.shape == (batch_size, seq_len, embedding_dim), "Output shape mismatch!"
    print("✓ Test passed! Embeddings working correctly.")
