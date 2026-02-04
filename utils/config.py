"""
Model and training configuration
Based on GPT-2 (124M parameters)
"""

# Model Architecture
VOCAB_SIZE = 50257  # GPT-2 tokenizer vocabulary size
CONTEXT_LENGTH = 256  # Maximum sequence length. 1024 in production.
EMBEDDING_DIM = 768  # Token embedding dimension
NUM_HEADS = 12  # Number of attention heads
NUM_LAYERS = 12  # Number of transformer blocks
DROPOUT_RATE = 0.1  # Dropout probability
QKV_BIAS = False  # Use bias in QKV projections

# Dataset and DataLoader
STRIDE = CONTEXT_LENGTH
BATCH_SIZE = 32

# Training