# monday_morning_moral
Pretrained and Instruction Fine Tuned LLM to generate bed time stories with moral

## Data:
TinyStoriesV2_cleaned Dataset Summary from huggingface
Dataset Structure:
  - Train: 2.6M examples
  - Test: 68K examples
  - Format: Single column text with bedtime stories
  - 99.98% of stories have |endoftext| delimiter

### After tokenization
  - Train tokens: 526.7 million (0.5 Billion)
  - Test tokens: 13.6 million
  - Total: 540.4 million tokens

## Epochs Calculation:
**Chinchilla suggests:** Train on 20 tokens per parameter
- For 163M parameters GPT-2 model: 163,009,536 × 20 = **3.26 billion tokens needed**
**We have:** **0.5 billion training tokens** in our dataset
**So we need to:** Train for **~6-7 epochs** (3.26B ÷ 526.7M = 6.19 epochs)
- This gives us Chinchilla-optimal training with a single-epoch equivalent of diverse data

## Process:
1. **Tokenization**: 
	1. read data from hugging face.
	2. since data is too large, we have to batch it using map() over huggingface dataset. 
	3. convert raw text data to tokens using the `TickToken`. Save the train_tokens and test_tokens as pytorch_tensors since re-tokenization takes time.

2. Embedding


  INPUT: [32, 256]

          ┌─ TOKEN EMBEDDING ─→ [32, 256, 768]
   Input ─┤                               ├─→ ADD ─→ [32, 256, 768]
          └─ POS EMBEDDING ──→ [256, 768] ┘

  They're parallel, not sequential:

  Then they're added together (PyTorch broadcasts):
  [32, 256, 768] + [256, 768] = [32, 256, 768]

  You have:
  Layer1(input) → output1 ──--┐
                              ├→ ADD → final_output
  Layer2(positions) → output2 ┘

# Learnings
- So it should be: nn.Embedding(CONTEXT_LENGTH, EMBEDDING_DIM)
  Why? nn.Embedding(num_embeddings, embedding_dim) where:
  - num_embeddings = how many things to look up (256 positions)
  - embedding_dim = what dimension each outputs (768 dims)