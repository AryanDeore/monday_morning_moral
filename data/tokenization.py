from datasets import load_dataset
import tiktoken
import torch
import os


def tokenize_batch(batch):
    """Join and tokenize stories in a batch."""
    enc = tiktoken.get_encoding("gpt2")
    batch_text = "".join(batch["text"])
    token_ids = enc.encode(batch_text, allowed_special={"<|endoftext|>"})

    return {"token_ids": [token_ids]}


# Check if tokens already exist
if os.path.exists("data/train_tokens.pt") and os.path.exists("data/test_tokens.pt"):
    print("Loading pre-tokenized tokens...")
    all_train_tokens = torch.load("data/train_tokens.pt").tolist()
    all_test_tokens = torch.load("data/test_tokens.pt").tolist()
    print(f"✓ Loaded: data/train_tokens.pt ({len(all_train_tokens):,} tokens)")
    print(f"✓ Loaded: data/test_tokens.pt ({len(all_test_tokens):,} tokens)")
else:
    # Load and tokenize
    dataset = load_dataset("fhswf/TinyStoriesV2_cleaned")
    train_df = dataset["train"]

    print("Tokenizing train split...")
    # Break dataset into batches. One batch contains 1000 stories.
    train_tokenized = train_df.map(
        tokenize_batch,
        batched=True,
        batch_size=1000,
        remove_columns=["text"]
    )

    # Tokenize test split
    test_df = dataset["test"]
    print("Tokenizing test split...")
    test_tokenized = test_df.map(
        tokenize_batch,
        batched=True,
        batch_size=1000,
        remove_columns=["text"]
    )

    print(f"\nTrain: {len(train_tokenized)} batches")
    print(f"Test: {len(test_tokenized)} batches")

    # Combine all batches into single token sequences
    print("\nCombining all batches into single sequences...")
    all_train_tokens = []
    for batch in train_tokenized:
        all_train_tokens.extend(batch['token_ids'])

    all_test_tokens = []
    for batch in test_tokenized:
        all_test_tokens.extend(batch['token_ids'])

    print(f"\nTotal train tokens: {len(all_train_tokens):,}")
    print(f"Total test tokens: {len(all_test_tokens):,}")
    print(f"Total tokens: {len(all_train_tokens) + len(all_test_tokens):,}")

    # Save as PyTorch tensors
    print("\nSaving tokens to disk...")
    torch.save(torch.tensor(all_train_tokens), "data/train_tokens.pt")
    torch.save(torch.tensor(all_test_tokens), "data/test_tokens.pt")

    print(f"✓ Saved: data/train_tokens.pt ({len(all_train_tokens):,} tokens)")
    print(f"✓ Saved: data/test_tokens.pt ({len(all_test_tokens):,} tokens)")
    print(f"\nTokenization complete!")

