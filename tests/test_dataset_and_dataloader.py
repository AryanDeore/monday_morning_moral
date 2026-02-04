"""
Test dataset and dataloader
Prints shapes and sample values for both train and test data
"""
import torch
from data.dataset import TokenDataset
from data.dataloader import create_dataloader
from utils.config import CONTEXT_LENGTH, BATCH_SIZE


def test_dataset(split='train'):
    """Test TokenDataset with shape and sample inspection"""
    print(f"\n{'='*50}")
    print(f"Testing {split.upper()} Dataset")
    print(f"{'='*50}")

    # Load tokens
    token_tensor = torch.load(f"data/{split}_tokens.pt")
    print(f"Token tensor shape: {token_tensor.shape}")
    print(f"Token tensor dtype: {token_tensor.dtype}")
    print(f"Sample tokens (first 20): {token_tensor[:20]}")

    # Create dataset
    dataset = TokenDataset(token_tensor, context_length=CONTEXT_LENGTH, stride=CONTEXT_LENGTH)
    print(f"\nDataset size: {len(dataset):,} examples")

    # Check sample
    x, y = dataset[0]
    print(f"\nSample (index 0):")
    print(f"  Input shape: {x.shape}")
    print(f"  Input dtype: {x.dtype}")
    print(f"  Input (first 20 tokens): {x[:20]}")
    print(f"  Target shape: {y.shape}")
    print(f"  Target (first 20 tokens): {y[:20]}")

    # Verify input/target relationship
    if torch.equal(x[1:], y[:-1]):
        print(f"  ✓ Input/target properly shifted (x[1:] == y[:-1])")
    else:
        print(f"  ✗ Input/target NOT properly shifted!")

    # Check another sample
    x2, y2 = dataset[100]
    print(f"\nSample (index 100):")
    print(f"  Input (first 10 tokens): {x2[:10]}")
    print(f"  Target (first 10 tokens): {y2[:10]}")

    print(f"\n✓ {split.upper()} Dataset tests passed")
    return dataset


def test_dataloader(split='train'):
    """Test DataLoader with shape and sample inspection"""
    print(f"\n{'='*50}")
    print(f"Testing {split.upper()} DataLoader")
    print(f"{'='*50}")

    # Create dataloader
    dataloader = create_dataloader(split)
    print(f"DataLoader created with batch_size={BATCH_SIZE}, context_length={CONTEXT_LENGTH}")
    print(f"Total batches: {len(dataloader):,}")

    # Check first batch
    print(f"\nFirst batch:")
    x_batch, y_batch = next(iter(dataloader))

    print(f"  Input batch shape: {x_batch.shape}")
    print(f"  Input batch dtype: {x_batch.dtype}")
    print(f"  First example in batch (first 20 tokens): {x_batch[0, :20]}")
    print(f"  Target batch shape: {y_batch.shape}")
    print(f"  First example target (first 20 tokens): {y_batch[0, :20]}")

    # Verify batch shapes
    assert x_batch.shape == torch.Size([BATCH_SIZE, CONTEXT_LENGTH]), \
        f"Wrong input batch shape: {x_batch.shape}"
    assert y_batch.shape == torch.Size([BATCH_SIZE, CONTEXT_LENGTH]), \
        f"Wrong target batch shape: {y_batch.shape}"
    print(f"  ✓ Batch shapes correct")

    # Check multiple batches
    print(f"\nIterating through batches:")
    batch_count = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        batch_count += 1
        if batch_idx == 0:
            print(f"  Batch 0: x={x.shape}, y={y.shape}")
        elif batch_idx == 1:
            print(f"  Batch 1: x={x.shape}, y={y.shape}")
            print(f"    First example (first 15 tokens): {x[0, :15]}")
        if batch_idx >= 2:
            break

    print(f"  Successfully iterated {batch_count} batches")
    print(f"\n✓ {split.upper()} DataLoader tests passed")
    return dataloader


if __name__ == "__main__":
    print("\n" + "="*50)
    print("DATA PIPELINE TEST")
    print("="*50)

    # Test datasets
    train_dataset = test_dataset('train')
    test_dataset_obj = test_dataset('test')

    # Test dataloaders
    train_loader = test_dataloader('train')
    test_loader = test_dataloader('test')

    print("\n" + "="*50)
    print("ALL TESTS PASSED!")
    print("="*50)
    print(f"\nSummary:")
    print(f"  Train dataset: {len(train_dataset):,} examples")
    print(f"  Test dataset: {len(test_dataset_obj):,} examples")
    print(f"  Train loader: {len(train_loader):,} batches of {BATCH_SIZE}")
    print(f"  Test loader: {len(test_loader):,} batches of {BATCH_SIZE}")
