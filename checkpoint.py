"""
Checkpoint utilities for saving and loading models.
"""

import os
import torch
from models import gpt2
from utils.config import *


def save_checkpoint(model, epoch, optimizer=None, config_name="gpt2-125m", checkpoint_dir="checkpoints"):
    """
    Save model checkpoint with metadata and optimizer state.

    Args:
        model: GPT-2 model
        epoch: Current epoch number
        optimizer: Optimizer (optional, for resuming training)
        config_name: Name of config (e.g., "gpt2-125m", "gpt2-30m"). Used to create subdirectory.
        checkpoint_dir: Base directory to save checkpoints
    """
    subdir = os.path.join(checkpoint_dir, config_name)
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    filepath = os.path.join(subdir, f"model_epoch_{epoch + 1}.pt")

    # Save checkpoint with metadata (extract config from model instance attributes)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch + 1,
        "config": {
            "vocab_size": model.vocab_size,
            "context_length": model.context_length,
            "embedding_dim": model.embedding_dim,
            "num_heads": model.num_heads,
            "num_layers": model.num_layers,
            "dropout_rate": model.dropout_rate,
        }
    }

    # Save optimizer state if provided
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

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
        cfg = {}
        print("Loaded legacy checkpoint (no metadata)")

    # Extract true config from model weights (overrides any wrong config values)
    pos_emb_shape = state_dict["embeddings.pos_embedding.weight"].shape
    true_context_length = pos_emb_shape[0]
    true_embedding_dim = pos_emb_shape[1]
    true_vocab_size = state_dict["output_layer.weight"].shape[0]

    # Count transformer blocks by finding the highest block index
    block_keys = [k for k in state_dict if k.startswith("blocks.")]
    true_num_layers = max(int(k.split(".")[1]) for k in block_keys) + 1

    # Count attention heads from the first block
    head_keys = [k for k in state_dict if k.startswith("blocks.0.mha.heads.") and k.endswith(".q.weight")]
    true_num_heads = len(head_keys)

    # Use dropout from config dict if available, otherwise default
    true_dropout_rate = cfg.get("dropout_rate", 0.1)

    print(f"Model config from weights: context_length={true_context_length}, embedding_dim={true_embedding_dim}, num_layers={true_num_layers}, num_heads={true_num_heads}")

    # Create model with true config extracted from weights
    model = gpt2.GPT2(
        dropout_rate=true_dropout_rate,
        vocab_size=true_vocab_size,
        context_length=true_context_length,
        embedding_dim=true_embedding_dim,
        num_layers=true_num_layers,
        num_heads=true_num_heads
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Loaded model from: {filepath}\n")
    return model


def load_checkpoint(filepath, device):
    """
    Load checkpoint for resuming training. Returns model, optimizer state, and epoch.

    Args:
        filepath: Path to checkpoint file
        device: Device to load on

    Returns:
        Tuple of (model, optimizer_state_dict, epoch) or (model, None, epoch) if optimizer state not available
    """
    checkpoint = torch.load(filepath, map_location=device)

    # Extract model state
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        cfg = checkpoint["config"]
        epoch = checkpoint.get("epoch", 1)
    else:
        state_dict = checkpoint
        cfg = {}
        epoch = 1
        print("Loaded legacy checkpoint (no metadata)")

    # Extract true config from model weights
    pos_emb_shape = state_dict["embeddings.pos_embedding.weight"].shape
    true_context_length = pos_emb_shape[0]
    true_embedding_dim = pos_emb_shape[1]
    true_vocab_size = state_dict["output_layer.weight"].shape[0]

    block_keys = [k for k in state_dict if k.startswith("blocks.")]
    true_num_layers = max(int(k.split(".")[1]) for k in block_keys) + 1

    head_keys = [k for k in state_dict if k.startswith("blocks.0.mha.heads.") and k.endswith(".q.weight")]
    true_num_heads = len(head_keys)

    true_dropout_rate = cfg.get("dropout_rate", 0.1)

    # Create model with true config extracted from weights
    model = gpt2.GPT2(
        dropout_rate=true_dropout_rate,
        vocab_size=true_vocab_size,
        context_length=true_context_length,
        embedding_dim=true_embedding_dim,
        num_layers=true_num_layers,
        num_heads=true_num_heads
    )

    model.load_state_dict(state_dict)
    model = model.to(device)

    # Extract optimizer state if available
    optimizer_state_dict = checkpoint.get("optimizer_state_dict", None)

    print(f"Loaded checkpoint from epoch {epoch}")
    return model, optimizer_state_dict, epoch
