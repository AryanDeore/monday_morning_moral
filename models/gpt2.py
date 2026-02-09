"""
The model flow is:
  Token indices [batch, seq_len]
    ↓
  Token embeddings + positional embeddings
    ↓
  Dropout
    ↓
  12 Transformer blocks (stacked)
    ↓
  Final Layer Norm
    ↓
  Linear output layer → logits [batch, seq_len, vocab_size]
"""
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from models import embeddings, transformer

class GPT2(nn.Module, PyTorchModelHubMixin):
    _model_name = "gpt2"

    def __init__(self, dropout_rate, vocab_size, context_length, embedding_dim, num_layers, num_heads):
        super().__init__()
        # Store init parameters as instance attributes for HF Hub mixin serialization
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embeddings = embeddings.Embeddings(vocab_size, context_length, embedding_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([ transformer.Transformer(context_length, embedding_dim, num_heads, dropout_rate) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_id):
        x = self.embeddings(token_id) # [batch, seq_len, embedding_dim]
        
        x = self.dropout(x)           # [batch, seq_len, embedding_dim]

        for block in self.blocks:
            x = block(x)

        x = self.layer_norm(x)

        output = self.output_layer(x)

        return output


if __name__ == "__main__":
    batch_size = 32
    seq_len = 256
    vocab_size = 50257
    embedding_dim = 768
    context_length = 1024
    num_layers = 12
    num_heads = 12
    dropout_rate = 0.1

    # Create dummy token IDs
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input token IDs shape: {token_ids.shape}")

    # Create GPT-2 model
    model = GPT2(dropout_rate, vocab_size, context_length, embedding_dim, num_layers, num_heads)

    # Forward pass
    output = model(token_ids)
    print(f"Output logits shape: {output.shape}")

    # Check output shape
    assert output.shape == (batch_size, seq_len, vocab_size), "Output shape mismatch!"
    print("✓ Test passed! GPT-2 model working correctly.")