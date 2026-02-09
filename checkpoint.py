"""
Checkpoint utilities for saving and loading models.
"""

import os
import torch
from models import gpt2
from utils.config import *


def save_checkpoint(model, epoch, checkpoint_dir="checkpoints"):
    """
    Save model checkpoint.

    Args:
        model: GPT-2 model
        epoch: Current epoch number
        checkpoint_dir: Directory to save checkpoint
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    filepath = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
    torch.save(model.state_dict(), filepath)
    print(f"Epoch {epoch + 1}: Checkpoint saved at {filepath}")


def load_model(filepath, device):
    """
    Load model from checkpoint.

    Args:
        filepath: Path to model file
        device: Device to load on

    Returns:
        GPT-2 model in eval mode
    """
    model = gpt2.GPT2(
        dropout_rate=dropout_rate,
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )

    model.load_state_dict(torch.load(filepath, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Loaded model from: {filepath}\n")
    return model
