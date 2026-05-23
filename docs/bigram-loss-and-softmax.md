# Bigram: Logits, Softmax, and Cross-Entropy

In the bigram model, the embedding table acts like a lookup table:

```text
current token id -> row of logits for the next token
```

If `vocab_size = 65`, then each row has 65 logits: one raw score for each
possible next character.

## Logits

Logits are raw, unnormalized scores. They are not probabilities yet, so they do
not need to be positive and they do not need to sum to 1.

```text
embedding row = logits over next-token choices
```

## Softmax

Softmax converts logits into probabilities:

```text
p_i = exp(logit_i) / sum(exp(logits))
```

This produces positive values that sum to 1. The exponentiation matters because
model logits can be any real number, including negative values.

## Cross-Entropy

Cross-entropy measures how much probability the model assigned to the correct
target token.

```text
high probability on correct token -> low loss
low probability on correct token -> high loss
```

For one token position, suppose:

```text
z = logits for all possible next tokens
y = correct next-token id
```

Step 1: convert logits to probabilities with softmax.

```text
p_i = exp(z_i) / sum_j(exp(z_j))
```

Step 2: take the probability assigned to the correct target token.

```text
p_y
```

Step 3: take the negative log of that probability.

```text
loss = -log(p_y)
```

So the full expression for one position is:

```text
loss = -log(exp(z_y) / sum_j(exp(z_j)))
```

For a batch, PyTorch computes this for every token position and returns the
mean loss by default:

```text
loss = mean(-log(p_y))
```

In PyTorch, `F.cross_entropy(logits, targets)` expects raw logits. It applies a
stable softmax/log operation internally, so do not softmax the logits first.

For the bigram model:

```text
logits:  [batch, block, vocab_size]
targets: [batch, block]
```

Before calling cross-entropy, flatten batch and block into one dimension:

```text
logits:  [batch * block, vocab_size]
targets: [batch * block]
```

This treats every token position in the batch as one next-token classification
example.
