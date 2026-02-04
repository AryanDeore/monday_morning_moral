# DataLoader creation utilities

import torch
from torch.utils.data import DataLoader
from data.dataset import TokenDataset
from utils.config import CONTEXT_LENGTH, STRIDE, BATCH_SIZE


def create_dataloader(split='train'):
    """
    Create a DataLoader for training or testing.

    Args:
        split: 'train' or 'test'

    Returns:
        DataLoader with batched token sequences
    """
    # Load pre-tokenized tokens
    token_tensor = torch.load(f"data/{split}_tokens.pt")

    # Create dataset (creates sliding windows + input/target pairs)
    dataset = TokenDataset(
        token_tensor=token_tensor,
        context_length=CONTEXT_LENGTH,
        stride=STRIDE
    )

    # Wrap in DataLoader (creates batches)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(split == 'train'),  # Shuffle only training data
        drop_last=True  # Drop incomplete batches
    )

    return dataloader
