# Fixed-Context MLP Language Model Plan

## Purpose

This chapter comes after the bigram model and before attention.

The goal is to build a language model that can use more than one previous
character when predicting the next character, without using recurrence or
attention yet.

The core upgrade is:

```text
bigram:
P(next char | current char)

fixed-context MLP:
P(next char | previous N chars)
```

This model should make the limitation of the bigram model concrete, while also
showing why fixed-context models are still awkward compared with attention.

## Historical Reference

This chapter is closely related to:

- [A Neural Probabilistic Language Model, Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

The paper's core setup is:

```text
previous N words
-> word embeddings
-> concatenate embeddings
-> neural network
-> predict next word
```

This chapter uses the same basic idea at character level:

```text
previous N characters
-> character embeddings
-> flatten embeddings
-> MLP
-> predict next character
```

This is an important step between count-based n-gram models and later sequence
models, because it introduces learned distributed token representations and a
neural network over a fixed context window.

## Core Innovation

The important idea is that each character still gets its own embedding, but the
model receives several embedded characters together.

For a context size of 4:

```text
input characters:
"Firs"

target:
"t"
```

The tokenizer turns the context into token ids:

```text
[stoi["F"], stoi["i"], stoi["r"], stoi["s"]]
```

The embedding table maps each token id independently:

```text
F -> [f1, f2, ..., f_n]
i -> [i1, i2, ..., i_n]
r -> [r1, r2, ..., r_n]
s -> [s1, s2, ..., s_n]
```

So after embedding, one example has shape:

```text
[context_size, n_embd]
```

For example:

```text
[4, 16]
```

With a batch, the shape is:

```text
[batch, context_size, n_embd]
```

The key reason to flatten is that a dense layer only mixes values across its
input feature dimension. If we apply a linear layer directly to:

```text
[batch, context_size, n_embd]
```

then PyTorch applies the same linear transformation to each token position
independently:

```text
[batch, context_size, n_embd]
-> Linear(n_embd, hidden_dim)
-> [batch, context_size, hidden_dim]
```

No token position has interacted with any other token position yet.

Flattening changes the feature dimension so the first dense layer can see all
positions in the fixed context at once:

```text
[batch, context_size, n_embd]
-> [batch, context_size * n_embd]
```

For example:

```text
[32, 4, 16]
-> [32, 64]
```

That flattened vector contains all previous `N` character embeddings side by
side:

```text
[F embedding, i embedding, r embedding, s embedding]
```

Now each hidden unit in the MLP can learn from coordinates belonging to any
position in the context:

```text
feature from position 1
+ feature from position 2
+ feature from position 4
-> useful signal for predicting the next character
```

So the MLP combines those embeddings to predict the single next character.

This is the key distinction:

```text
embedding layer:
maps each token independently

flatten + MLP:
combines the previous N embedded tokens into one prediction
```

So the model sees more sequence context not because one embedding represents
multiple characters, but because the input contains multiple embeddings at once.

## What This Model Can Learn

Compared with the bigram model, the MLP can distinguish contexts that end in the
same character.

For example, the bigram model sees all of these as just:

```text
r -> next character
```

But the MLP can see:

```text
Fir -> s
fur -> t
ear -> ?
pro -> c
```

That means it can learn short local patterns such as:

- Common prefixes.
- Common suffixes.
- Short word fragments.
- Local punctuation patterns.
- Some short phrase structure.

It still cannot naturally handle arbitrary-length context. The context size is
fixed, and every position has a fixed slot in the flattened input.

## Learning Goals

By the end of this chapter, be able to explain:

- The difference between `vocab_size`, `context_size`, and `n_embd`.
- Why `n_embd` is the dimension of each embedding vector, not the number of
  embedded tokens.
- Why embedding multiple characters produces shape
  `[batch, context_size, n_embd]`.
- Why flattening gives the MLP access to all previous `N` embedded characters.
- Why this is still a fixed-context model.
- Why attention is a more flexible way to use context.

## Step 1: Reuse the Bigram Pipeline

Reuse:

- Reading `input.txt`.
- Character-level vocabulary.
- `stoi` and `itos`.
- `encode` and `decode`.
- Train and validation split.
- Cross-entropy loss.
- Multinomial sampling during generation.

Questions to answer:

- Which pieces are general language-modeling infrastructure?
- Which pieces were specific to the bigram model?

## Step 2: Create Fixed-Context Batches

Each training example should contain:

```text
x: previous N token ids
y: one next token id
```

For example, with `context_size = 4`:

```text
text: "First"

x: "Firs"
y: "t"
```

Expected batch shapes:

```text
x: [batch, context_size]
y: [batch]
```

This differs from the bigram batching, where:

```text
x: [batch, block_size]
y: [batch, block_size]
```

Questions to answer:

- Why does this MLP predict one next token per row?
- How could we later extend this idea to predict at every position?
- What information does one row contain that a bigram row did not?

## Step 3: Add Token Embeddings

Create an embedding table:

```text
nn.Embedding(vocab_size, n_embd)
```

Expected shape path:

```text
x: [batch, context_size]
embeddings: [batch, context_size, n_embd]
```

Questions to answer:

- Why is the embedding table shape `[vocab_size, n_embd]`?
- Why can `n_embd` be smaller than `vocab_size` here?
- What does one row of the embedding table represent?

## Step 4: Flatten the Context

Flatten:

```text
[batch, context_size, n_embd]
-> [batch, context_size * n_embd]
```

This creates one vector per example containing the whole fixed context.

Questions to answer:

- What information is preserved by flattening?
- What sequence structure is made implicit by flattening?
- Why does changing `context_size` change the input size of the first linear
  layer?

## Step 5: Build the MLP

Suggested first model:

```text
embedding
-> flatten
-> linear
-> tanh or relu
-> linear
-> logits
```

Expected final shape:

```text
logits: [batch, vocab_size]
targets: [batch]
loss: scalar
```

Questions to answer:

- What does the hidden layer allow that a single linear layer would not?
- Why are logits still over the vocabulary?
- Why does cross-entropy now receive `[batch, vocab_size]` logits instead of
  `[batch * time, vocab_size]` logits?

## Step 6: Train and Generate

Training should look similar to the bigram model:

- Sample a batch.
- Compute logits.
- Compute cross-entropy.
- Backpropagate.
- Step the optimizer.
- Track train and validation loss.

Generation needs a current context of length `context_size`.

At each step:

- Take the most recent `context_size` token ids.
- Predict logits for the next token.
- Softmax to probabilities.
- Sample with `torch.multinomial`.
- Append the sampled token to the generated sequence.

Questions to answer:

- Why does generation need an initial context now?
- What should happen if the prompt is shorter than `context_size`?
- Why does the model still only generate one token at a time?

## Step 7: Compare Against Bigram

Record:

- Final training loss.
- Final validation loss.
- Generated samples.
- Overfit behavior on a tiny text.
- Differences from bigram output.

Questions to answer:

- Does the model produce more word-like fragments?
- Does loss improve over the bigram baseline?
- Which errors remain because the model still has fixed context?
- What would attention solve that this MLP does not?

## Suggested Initial Hyperparameters

Keep the first version small:

```text
context_size = 8
n_embd = 16
hidden_dim = 64
batch_size = 32
learning_rate = 1e-3
```

These are not magic values. They are small enough to debug and large enough to
show behavior beyond the bigram model.

## Milestone Complete When

- Fixed-context batching works.
- Embeddings have shape `[batch, context_size, n_embd]`.
- Flattened inputs have shape `[batch, context_size * n_embd]`.
- Logits have shape `[batch, vocab_size]`.
- Cross-entropy loss computes without reshaping confusion.
- Training loss decreases.
- Generation uses the rolling most recent `context_size` tokens.
- You can explain why this model uses more context than a bigram model, and why
  it is still limited.
