"""
Text generation with trained GPT-2 model.

Supports:
- Temperature scaling (control randomness)
- Top-k sampling (sample from top k likely tokens)
- EOS token stopping condition
- Throughput measurement (tokens/sec)
"""

import torch
import tiktoken
import time
from checkpoint import load_model
from utils.config import *


def text_to_token_ids(text, tokenizer):
    """Convert text to token IDs with batch dimension."""
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """Convert token IDs back to text."""
    flat = token_ids.squeeze(0)  # Remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=1.0,
    top_k=None,
    eos_id=None
):
    """
    Generate tokens autoregressively.

    Args:
        model: GPT-2 model in eval mode
        idx: Starting token indices [batch_size, seq_len]
        max_new_tokens: Number of tokens to generate
        context_size: Maximum context length for model
        temperature: Sampling temperature (>1 = more random, <1 = more greedy, 0 = greedy)
        top_k: If not None, only sample from top k tokens
        eos_id: End-of-sequence token ID. Stop if generated. If None, generate full length.

    Returns:
        Generated token IDs [batch_size, seq_len + max_new_tokens]
    """

    for _ in range(max_new_tokens):
        # Crop to context size (sliding window)
        idx_cond = idx[:, -context_size:]

        # Forward pass
        with torch.no_grad():
            logits = model(idx_cond)  # [batch, seq_len, vocab_size]

        # Get only last position logits
        logits = logits[:, -1, :]  # [batch, vocab_size]

        # Top-k filtering
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )

        # Temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Numerical stability: subtract max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Get probabilities
            probs = torch.softmax(logits, dim=-1)

            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # [batch, 1]
        else:
            # Greedy: take argmax
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # [batch, 1]

        # Check for EOS token
        if eos_id is not None and idx_next.item() == eos_id:
            break

        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


if __name__ == "__main__":
    # Setup
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    tokenizer = tiktoken.get_encoding("gpt2")

    # Load model checkpoint
    checkpoint_path = "checkpoints/model_epoch_5.pt"  # Update path if needed
    try:
        model = load_model(checkpoint_path, device)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Train the model first using: python train.py")
        exit(1)

    # Generation parameters
    prompt = "Once upon a time"
    max_new_tokens = 50
    temperature = 1.0  # 0.0 = greedy, 1.0 = neutral, >1.0 = more random
    top_k = 50  # Only sample from top 50 tokens, None to disable
    eos_id = None  # Set to token ID to stop on end-of-text, e.g., 50256

    print(f"Generation config:")
    print(f"  Prompt: '{prompt}'")
    print(f"  Max tokens: {max_new_tokens}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-k: {top_k}\n")

    # Convert prompt to tokens
    input_ids = text_to_token_ids(prompt, tokenizer).to(device)
    num_input_tokens = input_ids.shape[1]

    # Generate
    start_time = time.time()
    output_ids = generate(
        model=model,
        idx=input_ids,
        max_new_tokens=max_new_tokens,
        context_size=context_length,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id
    )
    end_time = time.time()

    elapsed_time = end_time - start_time
    tokens_per_sec = max_new_tokens / elapsed_time if elapsed_time > 0 else 0

    # Decode and display
    output_text = token_ids_to_text(output_ids, tokenizer)

    print(f"Generated text:")
    print(f"  {output_text}\n")

    print(f"Performance:")
    print(f"  Generated {max_new_tokens} tokens in {elapsed_time:.2f} seconds")
    print(f"  Throughput: {tokens_per_sec:.2f} tokens/sec")
