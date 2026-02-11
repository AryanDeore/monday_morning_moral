"""
Upload trained GPT-2 models to Hugging Face Hub.

Usage:
    python upload_to_hf.py

Requires: huggingface-cli login (run beforehand)
"""

import torch
from checkpoint import load_model
from huggingface_hub import HfApi, upload_file
import tempfile
import os


def upload_model(checkpoint_path, repo_id, model_card_path):
    """Load a checkpoint and push model + model card to HF Hub."""
    device = "cpu"  # Upload from CPU
    model = load_model(checkpoint_path, device)

    print(f"\nPushing model to {repo_id}...")
    model.push_to_hub(repo_id)
    print(f"Model weights + config.json pushed to {repo_id}")

    # Upload model card
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    print(f"Model card uploaded to {repo_id}")


if __name__ == "__main__":
    # 30M model
    upload_model(
        checkpoint_path="checkpoints/gpt2-30m/model_epoch_6.pt",
        repo_id="0rn0/gpt2-30m-tinystories",
        model_card_path="model_cards/gpt2-30m-README.md",
    )

    # 125M model
    upload_model(
        checkpoint_path="checkpoints/gpt2-125m/model_epoch_3.pt",
        repo_id="0rn0/gpt2-125m-tinystories",
        model_card_path="model_cards/gpt2-125m-README.md",
    )

    print("\nDone! Both models uploaded to Hugging Face.")
