# Single-Head Causal Self-Attention Plan

## Purpose

This section comes after the RNN and LSTM bridge. The goal is to understand
what attention changes: instead of compressing all history into a recurrent
state, each token can directly choose which earlier token representations to
read.

The progression is:

```text
vanilla RNN:
history compressed into h_t

LSTM:
history carried in gated (h_t, c_t)

causal self-attention:
position t directly weights representations at positions 0 through t
```

This milestone is intentionally limited to one manual causal attention head.
Do not use `nn.MultiheadAttention` or PyTorch's scaled-dot-product-attention
operator for the learning implementation.

## Reading

See the [attention reading list](reading.md). Read the first three sources
before implementation and use the PyTorch documentation only as a reference
after implementing the mechanism manually.

## Core Innovation

An attention head projects every input representation into a query, key, and
value vector:

```text
q_t = W_q x_t
k_t = W_k x_t
v_t = W_v x_t
```

For each query position `t`, query-key dot products score how relevant every
earlier key position is. After scaling and causal masking, softmax turns those
scores into weights. The output at position `t` is a weighted sum of value
vectors.

```text
attention weights = softmax((QK^T / sqrt(head_size)) + causal_mask)
output = attention weights @ V
```

## Shape Path

Use batch-first tensors throughout:

```text
idx:                [B, T]
token embeddings:   [B, T, E]
position embeddings:[T, E]
x:                  [B, T, E]
q, k, v:            [B, T, H]
scores:             [B, T, T]
weights:            [B, T, T]
head output:        [B, T, H]
logits:             [B, T, V]
```

One row `weights[b, t, :]` means: for example `b`, how position `t` divides
its attention across the allowed input positions.

## Implementation Steps

### Step 1: Set Up Token and Positional Embeddings

Given token ids `[B, T]`, produce a representation `x` shaped `[B, T, E]`.

Test:

- Token embedding shape.
- Position embedding shape.
- Combined representation shape.
- Changing a position changes its representation even when token ids match.

### Step 2: Create One Query, Key, and Value Projection

Use three learned linear layers to map `[B, T, E]` to `[B, T, H]`.

Test each projection shape before computing scores.

### Step 3: Compute and Scale Attention Scores

Compute:

```text
scores = Q @ K.transpose(-2, -1)
```

Verify its shape is `[B, T, T]`. Divide by `sqrt(head_size)` before softmax.

Explain why one score row compares one query with every key position.

### Step 4: Apply the Causal Mask

Before softmax, make future positions impossible to attend to.

Test:

- Position `t` cannot attend to positions greater than `t`.
- After softmax, allowed weights sum to one.
- Masked future positions have zero attention weight.

Then deliberately remove the mask in a controlled test to see why predicting
the next token would leak future information during training.

### Step 5: Compute the Weighted Value Sum

Multiply the attention weights by `V`:

```text
output = weights @ V
```

Test the output shape `[B, T, H]`. Explain that attention weights select and
mix value vectors; they are not output-token probabilities.

### Step 6: Build a Small Causal Attention Language Model

Project the head output to vocabulary logits `[B, T, V]`, calculate
cross-entropy over all positions, and verify one backward pass. Add basic
generation only after the forward and loss paths are understood.

## Questions Before Moving On

- Why do Q, K, and V use separate learned projections of the same input `x`?
- Why does `QK^T` have one score for every query-key position pair?
- Why is scaling by `sqrt(head_size)` necessary before softmax?
- Why must causal masking happen before softmax?
- How does attention avoid the fixed-width recurrent-state bottleneck?
- Why can training attention over a sequence be parallel while autoregressive
  generation remains sequential?

## Milestone Complete When

- A single causal attention head has shape and masking tests.
- The model produces `[B, T, V]` logits and supports a backward pass.
- You can explain one row of the attention-weight matrix.
- You can explain why causal masking prevents target leakage.
- You can contrast direct attention over earlier positions with RNN/LSTM state
  compression.

Stop before multi-head attention, residual connections, layer normalization,
or transformer blocks. Those are the following milestones.
