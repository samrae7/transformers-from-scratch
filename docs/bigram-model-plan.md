# Bigram Model Plan

## Purpose

Build a character-level bigram language model on Tiny Shakespeare before adding
attention or transformer blocks.

This milestone should prove that the basic language-modeling pipeline works:

- Reading text data.
- Tokenizing characters.
- Creating next-token prediction batches.
- Computing cross-entropy loss.
- Training with gradient descent.
- Sampling generated text.

The model will be simple, but it gives a debugging baseline for later
transformer components.

## Learning Goals

By the end of this milestone, be able to explain:

- Why language modeling is next-token prediction.
- What `x` and `y` represent in a training batch.
- Why logits have shape `[batch, time, vocab_size]`.
- How cross-entropy compares logits against target token ids.
- Why sampling introduces variation even when the model is deterministic.
- What it means for loss to decrease during training.

## Non-Goals

Do not add:

- Attention.
- Positional embeddings.
- Transformer blocks.
- Multi-head attention.
- HuggingFace or high-level trainer APIs.
- Beam search.
- Complex experiment tracking.

## Files

Suggested initial files:

- `input.txt`: Tiny Shakespeare raw text.
- `bigram.py`: main implementation and training script.
- `tests/test_bigram.py`: small invariant tests, if using `pytest`.

Keeping everything in one script is fine at this stage. Split into modules only
when the file becomes hard to reason about.

## Step 1: Inspect the Dataset

Tasks:

- Read `input.txt`.
- Print dataset length.
- Print the first few hundred characters.
- Count unique characters.

Questions to answer:

- What is one training example at character level?
- What is the vocabulary size?
- Are newline characters part of the vocabulary?

Checkpoint:

- You can explain what the raw text becomes before it enters the model.

## Step 2: Build the Character Tokenizer

Tasks:

- Create sorted unique characters.
- Build `stoi`: character to integer id.
- Build `itos`: integer id to character.
- Implement `encode(text)`.
- Implement `decode(ids)`.

Tests:

- `decode(encode(sample_text)) == sample_text`.
- Every encoded id is between `0` and `vocab_size - 1`.
- `len(stoi) == len(itos) == vocab_size`.

Questions to answer:

- Why does the model need integer token ids instead of raw characters?
- What information is lost with character-level tokenization?
- What becomes easier with character-level tokenization?

## Step 3: Encode and Split the Data

Tasks:

- Encode the entire dataset into a tensor of token ids.
- Split into train and validation sets, for example 90% / 10%.

Tests:

- Encoded data is one-dimensional.
- Train and validation sets do not overlap.
- The split preserves original token order.

Questions to answer:

- Why split by position instead of randomly shuffling individual characters?
- What does validation loss tell us that training loss does not?

## Step 4: Implement Batching

Tasks:

- Choose small initial values for `batch_size` and `block_size`.
- Implement `get_batch(split)`.
- Return:
  - `x` with shape `[batch_size, block_size]`.
  - `y` with shape `[batch_size, block_size]`.
- Make `y` the next-token targets for `x`.

Tests:

- `x.shape == y.shape == (batch_size, block_size)`.
- For each row, `y[:, :-1] == x[:, 1:]`.
- `x` and `y` contain integer token ids.

Questions to answer:

- For a single row, what token is each position trying to predict?
- Why does the model receive multiple positions per sequence instead of one
  character at a time?

## Step 5: Implement the Bigram Model

Model idea:

- Use a single embedding table with shape `[vocab_size, vocab_size]`.
- Each input token id indexes one row.
- That row is interpreted as logits for the next token.

Tasks:

- Create a small `torch.nn.Module`.
- Add `nn.Embedding(vocab_size, vocab_size)`.
- Implement `forward(idx, targets=None)`.
- Return logits and optional loss.

Expected shapes:

- Input `idx`: `[batch, time]`.
- Logits: `[batch, time, vocab_size]`.
- Targets: `[batch, time]`.
- Loss: scalar.

Tests:

- Forward pass returns logits with expected shape.
- Loss is scalar when targets are provided.
- Loss is omitted or `None` when targets are not provided.

Questions to answer:

- Why can an embedding table act as a bigram model?
- What does one row of the embedding table mean?
- Why does cross-entropy need logits reshaped before loss computation?

Reference:

- [Bigram: Logits, Softmax, and Cross-Entropy](bigram-loss-and-softmax.md)

## Step 6: Train the Model

Tasks:

- Choose an optimizer, such as AdamW.
- Sample batches from the training split.
- Compute loss.
- Zero gradients.
- Backpropagate.
- Step the optimizer.
- Log training loss periodically.
- Optionally estimate train and validation loss every fixed number of steps.

Tests and sanity checks:

- Loss should generally decrease over time.
- A tiny dataset should be overfittable.
- Gradients should be non-`None` for model parameters after `backward()`.

Questions to answer:

- What does it mean if loss stays flat?
- What does it mean if loss becomes `nan`?
- Why might validation loss differ from training loss?

## Step 7: Generate Text

Tasks:

- Start from a single token, often newline or zero.
- Run the model to get logits.
- Use only the logits for the final time step.
- Convert logits to probabilities with softmax.
- Sample the next token.
- Append it to the context.
- Repeat for a chosen number of tokens.
- Decode the generated ids.

Tests:

- Generated output has the requested length plus the initial context length.
- Generated ids are valid token ids.
- Decoding generated ids returns a string.

Questions to answer:

- Why use only the final time step's logits during generation?
- Why is generated output different across runs?
- What does the bigram model fail to capture?

Concepts to revisit next:

- Why cross-entropy loss can be interpreted through `exp(loss)`.
- Why `exp(2.47) ~= 11.8` means the model is roughly as uncertain as choosing
  among about 12 plausible next characters on average.
- How `torch.multinomial` samples from a probability distribution.
- How multinomial sampling differs from always taking `argmax`.
- Why sampling can produce more varied text while `argmax` can get stuck in
  repetitive loops.

## Step 8: Write Observations

Add a short note after the milestone covering:

- Final training loss.
- Final validation loss.
- Example generated text.
- What worked as expected.
- What was confusing.
- What failure modes appeared.

This note will be useful when comparing the bigram baseline to attention-based
models later.

## Suggested Milestone Definition

The bigram milestone is complete when:

- Tokenizer round-trip works.
- Batching produces correct shifted targets.
- The model forward pass returns `[batch, time, vocab_size]` logits.
- Cross-entropy loss computes without shape errors.
- Training loss decreases from its initial value.
- Generation produces valid decoded text.
- You can explain why the model is limited.

## Stretch Experiments

Only do these after the main milestone works:

- Compare different `block_size` values. The bigram model should not benefit
  much from longer context.
- Try overfitting on a tiny text sample.
- Sample with different temperatures.
- Inspect the learned next-character probabilities for a few characters.
- Compare generated text before and after training.
