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
  - Train tokens: 526.7 million
  - Test tokens: 13.6 million
  - Total: 540.4 million tokens


## Process:
1. **Tokenization**: 
	1. read data from hugging face.
	2. since data is too large, we have to batch it using map() over huggingface dataset. 
	3. convert raw text data to tokens using the `TickToken`. Save the train_tokens and test_tokens as pytorch_tensors since re-tokenization takes time.