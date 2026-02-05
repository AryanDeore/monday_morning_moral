"""
Test embeddings layer
Tests shape of token and positional embeddings
"""
import sys
sys.path.insert(0, '.')

import torch
from models.embeddings import Embeddings
from utils.config import vocab_size, embedding_dim, context_length, dropout_rate


def test_embeddings_shape():
    """Test that embeddings output has correct shape"""
    print(f"\n{'='*50}")
    print("Testing Embeddings Layer - Shape")
    print(f"{'='*50}")

    # Create embeddings layer
    embeddings = Embeddings(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate
    )
    embeddings.eval()  # Disable dropout for testing

    # Test cases: (batch_size, seq_len)
    test_cases = [
        (32, 256),   # Normal case
        (1, 256),    # Single sequence
        (32, 10),    # Shorter sequence
        (64, 128),   # Larger batch, different seq_len
    ]

    print(f"\nTesting with different input shapes:")
    for batch_size, seq_len in test_cases:
        # Create random input_ids
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        output = embeddings(input_ids)

        # Expected shape
        expected_shape = torch.Size([batch_size, seq_len, embedding_dim])

        # Assert
        assert output.shape == expected_shape, \
            f"Shape mismatch: got {output.shape}, expected {expected_shape}"

        print(f"  ✓ Input {input_ids.shape} → Output {output.shape}")

    print(f"\n✓ All shape tests passed!")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("EMBEDDINGS LAYER TEST")
    print("="*50)

    test_embeddings_shape()

    print("\n" + "="*50)
    print("ALL EMBEDDINGS TESTS PASSED!")
    print("="*50)
