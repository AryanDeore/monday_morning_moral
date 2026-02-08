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
from models import gpt2
from data.dataloader import create_dataloader
from utils.config import *

def train(num_epochs, max_batches=None):
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
    )

    loss_fn = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    model = model.to(device)
    print(f"Using device: {device}")
    
    train_dataloader = create_dataloader(split='train')
    test_dataloader = create_dataloader(split='test')
    print(f"Dataloaders created successfully")

    for epoch in range(num_epochs):
        model.train()
        batch_count = 0
        total_loss = 0

        for input_ids, targets in train_dataloader:
            if max_batches and batch_count >= max_batches:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(input_ids)  # [batch, seq_len, vocab_size]
            logits = torch.reshape(logits, (-1, vocab_size))
            targets = torch.reshape(targets, (-1,))
            loss = loss_fn(logits, targets)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            total_loss += loss.item()
            batch_count += 1

            if batch_count % 10 == 0:
                avg_loss = total_loss / batch_count
                print(f"Epoch {epoch + 1}, Batch {batch_count}, Loss: {loss.item():.4f}")

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            val_batches = 0

            with torch.no_grad():
                for val_input_ids, val_targets in test_dataloader:
                    val_input_ids = val_input_ids.to(device)
                    val_targets = val_targets.to(device)

                    val_logits = model(val_input_ids)
                    val_logits = torch.reshape(val_logits, (-1, vocab_size))
                    val_targets = torch.reshape(val_targets, (-1,))
                    val_loss += loss_fn(val_logits, val_targets).item()
                    val_batches += 1

            avg_train_loss = total_loss / batch_count
            avg_val_loss = val_loss / val_batches
            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")


if __name__ == "__main__":

    num_epochs = 1
    max_batches = 100
    
    train(num_epochs=num_epochs, max_batches=max_batches)