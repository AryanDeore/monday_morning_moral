<div align="center">

# TinyStories GPT

[![Demo](https://img.shields.io/badge/Live-%23FE4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://tinytales.aryandeore.ai/)
[![HuggingFace](https://img.shields.io/badge/huggingface-%23FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/0rn0) 


</div>

A GPT-2 language model built from scratch in PyTorch, pre-trained on bedtime stories. The model learns to generate children's stories with simple vocabulary and narrative structure. Two variants were trained and are available on HuggingFace: a [30M parameter model](https://huggingface.co/0rn0/gpt2-30m-tinystories) and a [125M parameter model](https://huggingface.co/0rn0/gpt2-125m-tinystories).



Below is a sample output from the **125M model** given the prompt *"Once upon a time"*:

```
Once upon a time, there was a little boy named Tim. Tim liked to play outside in the dirt.
One day, he found a big pipe in his yard. It was very old and dirty. Tim wanted to see what
was inside the pipe.

Tim tried to pull the pipe, but it was stuck. He pulled harder and harder until the pipe came
out of the ground. Inside the pipe, Tim found a small cat. The cat was scared and hungry. Tim
wanted to help the cat, but he didn't know how.

Then, something unexpected happened. The cat started to grow bigger and bigger. The big cat was
now as big as Tim! Tim was surprised, but he still wanted to help the cat. He took the cat
inside and gave it some food. The big cat became small again. Tim was happy that the dirty pipe
was now a magic pipe. He and the big cat became best friends and played together every day.
```

And from the **30M model**:

```
Once upon a time, there was a little girl named Lily. She had a big, blue box. One day, she
wanted to play with her toys in the warm sun. But the sun was too hot. Lily was sad.

Lily's mom saw her sad face. She asked, "Why are you sad, Lily?" Lily said, "I want to play,
but it is too hot." Her mom smiled and said, "Don't worry, we can use a cool towel to make it
feel better."

Lily and her mom put the cool towel on the table. They sat down and watched the sun rise.
The sun made the towel feel soft and warm. Lily was happy. She was not sad anymore. Her mom
said, "See, you can do anything when you help others."
```

## Table of Contents
- [Training Data](#training-data)
- [Model Configurations](#model-configurations)
- [Code Structure](#code-structure)
- [Step by Step Explanation](#step-by-step-explanation)
  - [Tokenization](#1-tokenization)
  - [Data Batching](#2-data-batching)
  - [Embeddings](#3-embeddings)
  - [Multi-Head Causal Self-Attention](#4-multi-head-causal-self-attention)
  - [Feed-Forward Network](#5-feed-forward-network)
  - [Transformer Block](#6-transformer-block)
  - [The GPT-2 Model](#7-the-gpt-2-model)
  - [Training](#8-training)
  - [Text Generation](#9-text-generation)
- [Training Results](#training-results)
- [What's Next](#whats-next)

## Training Data

The training data comes from [TinyStoriesV2_cleaned](https://huggingface.co/datasets/fhswf/TinyStoriesV2_cleaned) on HuggingFace. This dataset contains short bedtime stories written with vocabulary that young children can understand. Each story is delimited by a special `<|endoftext|>` token.

| Split | Examples | Tokens |
|-------|----------|--------|
| Train | 2.6M | 526.7M |
| Test | 68K | 13.6M |
| **Total** | **2.67M** | **540.4M** |


## Model Configurations

Two model variants were trained:

| Config | 30M | 125M |
|--------|-----|------|
| Vocab size | 50,257 | 50,257 |
| Context length | 512 | 512 |
| Embedding dim | 384 | 768 |
| Attention heads | 6 | 12 |
| Transformer layers | 6 | 12 |
| Dropout | 0.1 | 0.1 |
| Batch size | 32 | 64 |
| Learning rate | 5e-4 | 3e-4 |
| Epochs | 6 | 2 |

Both models are available on HuggingFace:
- [0rn0/gpt2-30m-tinystories](https://huggingface.co/0rn0/gpt2-30m-tinystories)
- [0rn0/gpt2-125m-tinystories](https://huggingface.co/0rn0/gpt2-125m-tinystories)

## Code Structure

```
monday_morning_moral/
├── models/
│   ├── gpt2.py                  # Full GPT-2 model (stacks all components)
│   ├── transformer.py           # Single transformer block
│   ├── multi_head_attention.py  # Multi-head causal self-attention
│   ├── embeddings.py            # Token + positional embeddings
│   └── feed_forward.py          # Feed-forward network (FFN)
├── data/
│   ├── tokenization.py          # Tokenize raw text with TikToken
│   ├── dataset.py               # Sliding window dataset (input/target pairs)
│   └── dataloader.py            # PyTorch DataLoader creation
├── utils/
│   └── config.py                # Model configs (30M and 125M)
├── train.py                     # Training loop with DDP via Accelerate
├── generate.py                  # Text generation (temperature + top-k sampling)
├── checkpoint.py                # Save/load model checkpoints
└── upload_to_hf.py              # Upload trained models to HuggingFace Hub
```

## Step by Step Explanation

### 1. Tokenization

Neural networks operate on numbers, not words. Tokenization converts raw text into a sequence of integer token IDs. I used OpenAI's [TikToken](https://github.com/openai/tiktoken) library with the GPT-2 encoding, which has a vocabulary of 50,257 tokens. Since the dataset is large (2.6M stories), tokenization is done in batches of 1,000 stories using HuggingFace's `.map()`, and the resulting token tensors are saved to disk so we don't have to re-tokenize every time.

```python
def tokenize_batch(batch):
    """Join and tokenize stories in a batch."""
    enc = tiktoken.get_encoding("gpt2")
    batch_text = "".join(batch["text"])
    token_ids = enc.encode(batch_text, allowed_special={"<|endoftext|>"})
    return {"token_ids": [token_ids]}

train_tokenized = train_df.map(
    tokenize_batch,
    batched=True,
    batch_size=1000,
    remove_columns=["text"]
)
```

### 2. Data Batching

The tokenized data is one long sequence of ~526M tokens. To feed it to the model, we slice it into fixed-length windows using a sliding window approach where the stride equals the context length (no overlap). For each window, the target is the input shifted by one position - the model learns to predict the next token at every position.

```python
class TokenDataset(Dataset):
    def __init__(self, token_tensor, context_length, stride):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_tensor) - context_length, stride):
            input_window = token_tensor[i:i+context_length]
            target_window = token_tensor[i+1:i+context_length+1]
            self.input_ids.append(input_window)
            self.target_ids.append(target_window)
```

These windows are then wrapped in a PyTorch `DataLoader` that handles batching and shuffling. Training data is shuffled; test data is not.

### 3. Embeddings

Each token ID is converted into a dense vector (embedding) that captures its semantic meaning. GPT-2 uses two embedding layers: one for the token itself and one for its position in the sequence. The position embedding lets the model understand word order - without it, "the cat sat on the mat" and "the mat sat on the cat" would look identical. 
>The two embeddings are added together element-wise, and dropout is applied for regularization.

```python
class Embeddings(nn.Module):
    def __init__(self, vocab_size, context_length, embedding_dim, dropout_rate):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        tok_embeds = self.token_embedding(input_ids)          # [batch, seq_len, 768]
        pos = torch.arange((input_ids.shape[1]), device=input_ids.device)
        pos_embeds = self.pos_embedding(pos)                  # [seq_len, 768]
        combined = tok_embeds + pos_embeds                    # broadcast add
        return self.dropout(combined)
```

### 4. Multi-Head Causal Self-Attention

Attention is the mechanism that lets each token look at other tokens in the sequence to build context. Each attention head computes three projections from the input: a **query** (what am I looking for?), a **key** (what do I contain?), and a **value** (what information do I carry?). The attention score between two tokens is the dot product of the query and key, scaled by the square root of the head dimension for numerical stability. A **causal mask** (lower triangular matrix) ensures tokens can only attend to previous positions - the model can't cheat by looking at future tokens during training.
> pattern: 

```python
class Head(nn.Module):
    def __init__(self, num_heads, embedding_dim, context_length, dropout_rate):
        super().__init__()
        self.q = nn.Linear(embedding_dim, embedding_dim // num_heads)
        self.k = nn.Linear(embedding_dim, embedding_dim // num_heads)
        self.v = nn.Linear(embedding_dim, embedding_dim // num_heads)
        self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, token_embedding):
        q = self.q(token_embedding)
        k = self.k(token_embedding)
        v = self.v(token_embedding)

        scores = (q @ k.transpose(2, 1)) / (q.shape[-1] ** 0.5)   # scaled dot-product
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))  # causal mask
        attention_weights = F.softmax(scores, dim=-1)
        context_vec = self.dropout(attention_weights @ v)
        return context_vec
```

Multiple heads run in parallel, each attending to different aspects of the input. Their outputs are concatenated and passed through a linear projection layer.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embedding_dim, context_length, dropout_rate):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(num_heads, embedding_dim, context_length, dropout_rate)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, token_embedding):
        concatenated = torch.cat([head(token_embedding) for head in self.heads], dim=-1)
        return self.out_proj(concatenated)
```

### 5. Feed-Forward Network

The feed-forward network processes each token independently after attention has mixed information between tokens. It consists of two linear layers with a GELU activation in between. The first layer expands the dimension by 4x (768 to 3,072 for the 125M model), allowing the network to learn in a higher-dimensional space, and the second projects it back down.
> Stacked layers: linear(emb_dim, 4*embedding_dim) -> GELU() -> linear(4*embedding_dim, embedding_dim)

```python
class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

    def forward(self, input_tensor):
        return self.layers(input_tensor)
```

### 6. Transformer Block

A transformer block combines attention and feed-forward processing with two key additions: **layer normalization** (normalizes activations to stabilize training) and **residual connections** (add the input back to the output so gradients flow easily through deep networks). 
> The pattern is: normalize, attend, add residual, normalize, feed-forward, add residual.

```python
class Transformer(nn.Module):
    def __init__(self, context_length, embedding_dim, num_heads, dropout_rate):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm([embedding_dim])
        self.mha = MultiHeadAttention(num_heads, embedding_dim, context_length, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm([embedding_dim])
        self.feed_forward = FeedForward(embedding_dim)

    def forward(self, input_tokens):
        shortcut = input_tokens
        input_tokens = self.layer_norm1(input_tokens)
        input_tokens = self.mha(input_tokens)
        input_tokens = self.dropout(input_tokens)
        input_tokens = input_tokens + shortcut          # residual connection

        shortcut = input_tokens
        input_tokens = self.layer_norm2(input_tokens)
        input_tokens = self.feed_forward(input_tokens)
        input_tokens = self.dropout(input_tokens)
        input_tokens = input_tokens + shortcut          # residual connection
        return input_tokens
```

### 7. The GPT-2 Model

The full model stacks everything together: embeddings feed into N transformer blocks, followed by a final layer norm and a linear output layer that maps back to vocabulary size. Each position in the output is a probability distribution over all 50,257 tokens, representing what the model thinks the next token should be.
> Layers: embeddings -> dropout -> 12 Transformer blocks -> layer_norm -> Linear Output_layer

```python
class GPT2(nn.Module):
    def __init__(self, dropout_rate, vocab_size, context_length, embedding_dim, num_layers, num_heads):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, context_length, embedding_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([
            Transformer(context_length, embedding_dim, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_id):
        x = self.embeddings(token_id)    # [batch, seq_len, embedding_dim]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)
        return self.output_layer(x)      # [batch, seq_len, vocab_size]
```

### 8. Training

Training uses cross-entropy loss between the model's predicted next-token distribution and the actual next token. The loss is calculated by flattening the logits and targets across the batch and sequence dimensions. Perplexity (e^loss) is tracked as a more interpretable metric - it roughly represents "how many tokens the model is choosing between" at each step.

Distributed training is handled by HuggingFace's [Accelerate](https://huggingface.co/docs/accelerate) library, which wraps the model, optimizer, and dataloaders for **DDP (Distributed Data Parallel)** across multiple GPUs. Training uses **bf16 mixed precision** and a **fused AdamW optimizer** for performance on 4x H100 GPUs. Checkpoints are saved after every epoch.

```python
accelerator = Accelerator(mixed_precision="bf16")
model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, test_dataloader
)

for epoch in range(num_epochs):
    model.train()
    for input_ids, targets in train_dataloader:
        logits = model(input_ids)
        logits = torch.reshape(logits, (-1, vocab_size))
        targets = torch.reshape(targets, (-1,))
        loss = loss_fn(logits, targets)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

<img width="769" height="636" alt="gpu_100" src="https://github.com/user-attachments/assets/8c2096b1-7828-43b5-8258-a787c4ff1b9b" />

### 9. Text Generation

At inference time, the model generates text one token at a time. Given a prompt, it predicts the next token, appends it, and repeats. Two sampling techniques add creativity:

- **Temperature scaling**: Divides logits by a temperature value before softmax. Temperature > 1 makes the distribution flatter (more random), < 1 makes it sharper (more deterministic), and 0 is pure greedy (always pick the highest probability token).
- **Top-k sampling**: Only considers the k most likely tokens at each step, setting all other probabilities to zero. This prevents the model from picking very unlikely tokens while still allowing diversity.

A sliding window crops the input to the model's context length, allowing generation beyond the context window.

```python
def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]                  # sliding window
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]                          # last position only

        if top_k is not None:                              # top-k filtering
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, float("-inf"), logits)

        if temperature > 0.0:                              # temperature scaling
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next.item() == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

## Training Results

**30M model** (30M params, 6 epochs on NVIDIA H100 80GB, ~50 minutes):

| Epoch | Train Loss | Val Loss | Val Perplexity |
|-------|-----------|----------|----------------|
| 1 | 2.1402 | 1.5468 | 4.696 |
| 2 | 1.5407 | 1.4061 | 4.080 |
| 3 | 1.4463 | 1.3487 | 3.852 |
| 4 | 1.3987 | 1.3134 | 3.719 |
| 5 | 1.3675 | 1.2878 | 3.625 |
| 6 | 1.3458 | 1.2721 | 3.568 |

## What's Next

The pre-trained models generate coherent stories but can't follow instructions like "write a sad story about a dog." In a [separate repo](https://github.com/0rn0/monday-morning-moral-sft), I instruction fine-tuned these models using supervised fine-tuning (SFT) on the TinyStoriesInstruct dataset, enabling prompts like:

```
Write a story about: a brave knight on an adventure
With: a sad ending
```

## References

- [Training a base model from scratch on RTX 3090](https://www.gilesthomas.com/2025/12/llm-from-scratch-28-training-a-base-model-from-scratch) - Giles Thomas
- [DDP training a base model in the cloud](https://www.gilesthomas.com/2026/01/llm-from-scratch-29-ddp-training-a-base-model-in-the-cloud) - Giles Thomas
- [PyTorch Performance Tips for Faster LLM Training](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/10_llm-training-speed) - Sebastian Raschka
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) - Lilian Weng
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar
- [Attention Mechanism Explained](https://www.youtube.com/watch?v=4vye3iUrR-g)
- [Let's Build GPT from Scratch](https://www.youtube.com/watch?v=bQ5BoolX9Ag)
- [LLMs from Scratch - Training](https://www.youtube.com/watch?v=YSAkgEarBGE&list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11&index=4)
- [Day 7: 21 Days of Building a Small Language Model](https://www.reddit.com/r/LocalLLaMA/comments/1pn0cik/day_7_21_days_of_building_a_small_language_model/)

---

[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/aryandeore) [![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:aryandeore.work@gmail.com)
