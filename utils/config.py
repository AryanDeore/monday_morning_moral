"""
Model and training configuration
Based on GPT-2 (124M parameters)
"""

# Model Architecture
vocab_size = 50257  # GPT-2 tokenizer vocabulary size
context_length = 256  # Maximum sequence length. 1024 in production.
embedding_dim = 768  # Token embedding dimension
num_heads = 12  # Number of attention heads
num_layers = 12  # Number of transformer blocks
dropout_rate = 0.1  # Dropout probability
qkv_bias = False  # Use bias in QKV projections

# Dataset and DataLoader
stride = context_length
batch_size = 32

# Training
learning_rate = 0.0003
num_epochs = 6