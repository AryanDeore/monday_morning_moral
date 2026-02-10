"""
Training loop:
  Part 1: Setup & Initialization

  - Import necessary libraries (torch, nn, optimizers, dataloaders)
  - Import your GPT-2 model and config values
  - Create the GPT-2 model instance using config values
  - Set up the optimizer (AdamW) with the model parameters
  - Define the loss function (CrossEntropyLoss for language modeling)
  - Determine the device (CPU or GPU)
  - Move the model to the device

  Part 2: Data Loading

  - Create training dataloader using create_dataloader('train')
  - Create validation/test dataloader using create_dataloader('test')
  - These return batches of (input_tokens, target_tokens)

  Part 3: Main Training Loop

  For each epoch:
  - Training Phase:
    - Set model to training mode
    - For each batch from training dataloader:
        - Forward pass: feed input tokens through model → get logits
      - Compute loss: compare logits with target tokens
      - Backward pass: compute gradients
      - Optimizer step: update weights
      - Log progress (print loss every N batches)
  - Validation Phase:
    - Set model to evaluation mode (turns off dropout)
    - For each batch from test dataloader:
        - Forward pass (no gradients)
      - Compute validation loss
      - Log validation loss
  - Checkpointing:
    - Save model weights every N epochs
    - Save final model at the end

  Part 4: Main Block

  - Call the train() function when script is run directly
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import time
import json
import math
import os
import argparse
from accelerate import Accelerator
from models import gpt2
from data.dataloader import create_dataloader
from checkpoint import save_checkpoint
from utils.config import get_config

# Default config values (will be overridden by --model-size argument)
vocab_size = 50257
context_length = 1024
embedding_dim = 768
num_heads = 12
num_layers = 12
dropout_rate = 0.1
batch_size = 32
learning_rate = 0.0003
num_epochs = 6

def train(num_epochs, max_batches=None, max_tokens=None, config_name="gpt2-125m"):
    """
    Train GPT-2 model with Accelerate for distributed training.

    Args:
        num_epochs: Number of training epochs
        max_batches: Limit batches per epoch (for quick testing)
        max_tokens: Limit tokens in dataset (for quick testing)
        config_name: Name of config for checkpoint directory (e.g., "gpt2-125m", "gpt2-30m")
    """
    # Initialize Accelerate for distributed training with bf16 mixed precision
    accelerator = Accelerator(mixed_precision="bf16")

    model = gpt2.GPT2(
        dropout_rate=dropout_rate,
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )

    # Calculate model size
    param_count = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        fused=True
    )

    loss_fn = nn.CrossEntropyLoss()

    # Create dataloaders
    train_dataloader = create_dataloader(split='train', max_tokens=max_tokens, context_length=context_length, batch_size=batch_size)
    test_dataloader = create_dataloader(split='test', max_tokens=max_tokens // 10 if max_tokens else None, context_length=context_length, batch_size=batch_size)

    # Prepare model, optimizer, and dataloaders for distributed training
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    if accelerator.is_main_process:
        print(f"Using device: {accelerator.device}")
        print(f"Number of GPUs: {accelerator.num_processes}")
        print(f"Dataloaders created successfully\n")

    # Debug: Check actual batch size
    for input_ids, targets in train_dataloader:
        if accelerator.is_main_process:
            print(f"DEBUG - Input batch shape: {input_ids.shape}")
            print(f"DEBUG - Per-GPU batch size: {input_ids.shape[0]}")
            print(f"DEBUG - Sequence length: {input_ids.shape[1]}")
        break

    # Track losses for plotting
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        batch_count = 0
        total_loss = 0

        for input_ids, targets in train_dataloader:
            if max_batches and batch_count >= max_batches:
                break

            # Data is already on the correct device via accelerator.prepare()

            # Forward pass
            logits = model(input_ids)  # [batch, seq_len, vocab_size]
            logits = torch.reshape(logits, (-1, vocab_size))
            targets = torch.reshape(targets, (-1,))
            loss = loss_fn(logits, targets)

            # Backward pass with Accelerate
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            batch_count += 1

        # Validation and logging
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for val_input_ids, val_targets in test_dataloader:
                # Data is already on the correct device via accelerator.prepare()

                val_logits = model(val_input_ids)
                val_logits = torch.reshape(val_logits, (-1, vocab_size))
                val_targets = torch.reshape(val_targets, (-1,))
                val_loss += loss_fn(val_logits, val_targets).item()
                val_batches += 1

        avg_train_loss = total_loss / batch_count if batch_count > 0 else 0.0
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0

        # Append to tracking lists
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Only log and save on main process
        if accelerator.is_main_process:
            if batch_count == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}: WARNING - No training batches processed")
            else:
                print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

            # Save checkpoint every epoch (unwrap model from Accelerate)
            unwrapped_model = accelerator.unwrap_model(model)
            save_checkpoint(unwrapped_model, epoch, config_name=config_name)

    if accelerator.is_main_process:
        print(f"\nTraining complete!")
        print(f"Final Train Loss: {train_losses[-1]:.4f}")
        print(f"Final Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses, param_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-2 model")
    parser.add_argument(
        "--model-size",
        choices=["125m", "30m"],
        default="125m",
        help="Model size to train (default: 125m)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Limit tokens in dataset (for quick testing)"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit batches per epoch (for quick testing)"
    )
    args = parser.parse_args()

    # Load config for selected model size
    config = get_config(args.model_size)

    # Update module-level variables with selected config
    vocab_size = config["vocab_size"]
    context_length = config["context_length"]
    embedding_dim = config["embedding_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    dropout_rate = config["dropout_rate"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]

    config_name = f"gpt2-{args.model_size}"

    print(f"Training GPT-2 {args.model_size.upper()} model")
    print(f"Config: vocab_size={vocab_size}, context_length={context_length}, embedding_dim={embedding_dim}")
    print(f"Training config: epochs={num_epochs}, max_tokens={args.max_tokens}\n")

    start_time = time.time()
    train_losses, val_losses, param_count = train(
        num_epochs=num_epochs,
        max_batches=args.max_batches,
        max_tokens=args.max_tokens,
        config_name=config_name
    )
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Save training metrics
    metrics = {
        "param_count": param_count,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_perplexities": [math.exp(loss) for loss in val_losses],
        "training_time_seconds": elapsed_time,
        "hardware": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "num_gpu_processes": None  # Will be set by Accelerate in distributed training
        },
        "config": {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout_rate": dropout_rate,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs
        }
    }

    metrics_path = os.path.join("checkpoints", config_name, "metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Clean up distributed training resources
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    # Save losses for later plotting
    # print(f"\nTrain Losses: {[f'{l:.4f}' for l in train_losses]}")
    # print(f"Val Losses: {[f'{l:.4f}' for l in val_losses]}")