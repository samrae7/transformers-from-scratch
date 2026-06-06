# MLPs, Linear Layers, and Matrix Operations

## Why Does the MLP Output `vocab_size` Values?

We deliberately shape the final layer of the MLP to output `vocab_size`
values:

```python
nn.Linear(hidden_dim, vocab_size)
```

The model is predicting the next token. Every token in the vocabulary is a
possible answer, so the model needs to produce one score, called a logit, for
each possible token.

Therefore:

```text
hidden representation: [batch, hidden_dim]
output logits:         [batch, vocab_size]
```

Cross-entropy compares these `vocab_size` logits with the index of the correct
next token.

## Layer Shapes Are Chosen Deliberately

Each layer must accept the shape produced by the previous operation:

```text
token IDs:
[batch, context_size]

embedding lookup:
[batch, context_size, n_embd]

flatten:
[batch, context_size * n_embd]

hidden linear layer:
[batch, hidden_dim]

output linear layer:
[batch, vocab_size]
```

For:

```python
nn.Linear(in_features, out_features)
```

- `in_features` must match the last dimension of its input.
- `out_features` determines the last dimension of its output.

The embedding dimension and output dimension have different purposes:

- `n_embd` controls how many learned features represent each input token.
- `vocab_size` is the number of possible next-token predictions.

## Linear Layers Are Matrix Operations

A linear layer approximately computes:

```text
output = input @ weight.T + bias
```

For example:

```text
input:  [batch, context_size * n_embd]
weight: [hidden_dim, context_size * n_embd]
output: [batch, hidden_dim]
```

The inner dimensions must match for the matrix multiplication to work.

An embedding lookup is implemented as selecting rows from an embedding matrix:

```text
embedding table: [vocab_size, n_embd]
token ID 7:      select row 7
```

Conceptually, this is equivalent to multiplying a one-hot token vector by the
embedding matrix, but direct row lookup is more efficient.

## Key Idea

Neural-network layer shapes are architectural choices, not accidental results.
We choose intermediate dimensions to build useful representations, and we
choose the final dimension to match the prediction task. For next-token
prediction, that means one final logit for every token in the vocabulary.
