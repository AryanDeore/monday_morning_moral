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
from accelerate import Accelerator
from models import gpt2
from data.dataloader import create_dataloader
from checkpoint import save_checkpoint
from utils.config import *

def train(num_epochs, max_batches=None, max_tokens=None):
    """
    Train GPT-2 model with Accelerate for distributed training.

    Args:
        num_epochs: Number of training epochs
        max_batches: Limit batches per epoch (for quick testing)
        max_tokens: Limit tokens in dataset (for quick testing)
    """
    # Initialize Accelerate for distributed training
    accelerator = Accelerator()

    model = gpt2.GPT2(
        dropout_rate=dropout_rate,
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
        # ,fused=True
    )

    loss_fn = nn.CrossEntropyLoss()

    # Create dataloaders
    train_dataloader = create_dataloader(split='train', max_tokens=max_tokens)
    test_dataloader = create_dataloader(split='test', max_tokens=max_tokens // 10 if max_tokens else None)

    # Prepare model, optimizer, and dataloaders for distributed training
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    if accelerator.is_main_process:
        print(f"Using device: {accelerator.device}")
        print(f"Number of GPUs: {accelerator.num_processes}")
        print(f"Dataloaders created successfully\n")

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

        avg_train_loss = total_loss / batch_count
        avg_val_loss = val_loss / val_batches

        # Append to tracking lists
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Only log and save on main process
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

            # Save checkpoint every epoch (unwrap model from Accelerate)
            unwrapped_model = accelerator.unwrap_model(model)
            save_checkpoint(unwrapped_model, epoch)

    if accelerator.is_main_process:
        print(f"\nTraining complete!")
        print(f"Final Train Loss: {train_losses[-1]:.4f}")
        print(f"Final Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses


if __name__ == "__main__":
    # Testing configuration (quick iteration)
    num_epochs = 5
    max_batches = 5  # No batch limit
    max_tokens = 300000  # Use only 100k tokens for fast testing

    # For full training, use:
    # num_epochs = 6
    # max_batches = None
    # max_tokens = None  # Use all 500M tokens

    print(f"Training config: epochs={num_epochs}, max_tokens={max_tokens}\n")

    start_time = time.time()
    train_losses, val_losses = train(num_epochs=num_epochs, max_batches=max_batches, max_tokens=max_tokens)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Clean up distributed training resources
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    # Save losses for later plotting
    # print(f"\nTrain Losses: {[f'{l:.4f}' for l in train_losses]}")
    # print(f"Val Losses: {[f'{l:.4f}' for l in val_losses]}")