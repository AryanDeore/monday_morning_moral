"""
Checkpoint utilities for saving and loading models.
"""

import os
import torch
from models import gpt2
from utils.config import *


def save_checkpoint(model, epoch, config_name="gpt2-125m", checkpoint_dir="checkpoints"):
    """
    Save model checkpoint with metadata.

    Args:
        model: GPT-2 model
        epoch: Current epoch number
        config_name: Name of config (e.g., "gpt2-125m", "gpt2-30m"). Used to create subdirectory.
        checkpoint_dir: Base directory to save checkpoints
    """
    subdir = os.path.join(checkpoint_dir, config_name)
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    filepath = os.path.join(subdir, f"model_epoch_{epoch + 1}.pt")

    # Save checkpoint with metadata
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch + 1,
        "config": {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
        }
    }

    torch.save(checkpoint, filepath)
    print(f"Epoch {epoch + 1}: Checkpoint saved at {filepath}")


def load_model(filepath, device):
    """
    Load model from checkpoint. Handles both old format (bare state_dict) and new format (with metadata).

    Args:
        filepath: Path to model file
        device: Device to load on

    Returns:
        GPT-2 model in eval mode
    """
    checkpoint = torch.load(filepath, map_location=device)

    # Handle both old format (bare state_dict) and new format (dict with keys)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # New format: checkpoint contains model_state_dict, epoch, config
        state_dict = checkpoint["model_state_dict"]
        cfg = checkpoint["config"]
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        # Old format: checkpoint is just the state_dict
        state_dict = checkpoint
        cfg = {
            "dropout_rate": dropout_rate,
            "vocab_size": vocab_size,
            "context_length": context_length,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "num_heads": num_heads
        }
        print("Loaded legacy checkpoint (no metadata). Using config from utils.config")

    # Create model with loaded or default config
    model = gpt2.GPT2(
        dropout_rate=cfg["dropout_rate"],
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        embedding_dim=cfg["embedding_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"]
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Loaded model from: {filepath}\n")
    return model
