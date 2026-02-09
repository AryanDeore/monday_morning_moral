# DataLoader creation utilities

import torch
from torch.utils.data import DataLoader
from data.dataset import TokenDataset
from utils.config import batch_size


def create_dataloader(split='train', max_tokens=None, context_length=512, stride=None):
    """
    Create a DataLoader for training or testing.

    Args:
        split: 'train' or 'test'
        max_tokens: Optional limit on number of tokens to load (for testing)
        context_length: Context window size
        stride: Stride for sliding window (default: context_length)

    Returns:
        DataLoader with batched token sequences
    """
    if stride is None:
        stride = context_length

    # Load pre-tokenized tokens
    token_tensor = torch.load(f"data/{split}_tokens.pt")

    # Limit tokens if specified (for quick testing)
    if max_tokens:
        token_tensor = token_tensor[:max_tokens]

    # Create dataset (creates sliding windows + input/target pairs)
    dataset = TokenDataset(
        token_tensor=token_tensor,
        context_length=context_length,
        stride=stride
    )

    # Wrap in DataLoader (creates batches)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),  # Shuffle only training data
        drop_last=True  # Drop incomplete batches
    )

    return dataloader


if __name__ == "__main__":
    print("Checking DataLoader sizes...\n")

    # Full dataloader
    train_dl_full = create_dataloader(split='train')
    test_dl_full = create_dataloader(split='test')

    print(f"Full Dataset:")
    print(f"  Train batches: {len(train_dl_full)}")
    print(f"  Test batches: {len(test_dl_full)}\n")

    # Small dataloader for testing
    train_dl_small = create_dataloader(split='train', max_tokens=100000)
    test_dl_small = create_dataloader(split='test', max_tokens=10000)

    print(f"Small Dataset (100k train, 10k test tokens):")
    print(f"  Train batches: {len(train_dl_small)}")
    print(f"  Test batches: {len(test_dl_small)}\n")

    # Get first batch to see shape
    for input_ids, targets in train_dl_small:
        print(f"Sample batch shape: input {input_ids.shape}, targets {targets.shape}")
        break
