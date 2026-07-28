# Single-Head Causal Self-Attention Reading List

Read these in order before and alongside the single-head attention milestone.
The aim is to understand the shape flow and causal constraint before writing
the implementation.

## 1. Dive into Deep Learning: Queries, Keys, and Values

[Read Section 11.1](https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html).

This introduces the vocabulary behind attention. Before moving on, be able to
say what a query asks for, what a key is matched against, and what a value is
combined after matching.

## 2. Dive into Deep Learning: Attention Pooling and Scoring Functions

[Read Section 11.2](https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/attention-pooling.html)
and [Section 11.3](https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html).

These sections explain attention as a weighted combination of values and
introduce dot-product scoring. This is the direct conceptual lead-in to the
scaled dot-product attention head you will implement.

## 3. Dive into Deep Learning: Self-Attention and Positional Encoding

[Read the chapter](https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html).

Focus on:

- Why self-attention uses the same sequence as queries, keys, and values.
- Why self-attention can process sequence positions in parallel during
  training, unlike an RNN or LSTM.
- Why position information must be added when the architecture has no
  recurrent ordering mechanism.

Before continuing, be able to explain why an input `x` with shape `[B, T, E]`
can produce `q`, `k`, and `v` with shape `[B, T, H]`.

## 4. The Illustrated GPT-2: Masked Self-Attention

[Read the self-attention and masked self-attention sections](https://jalammar.github.io/illustrated-gpt2/).

This is the most directly relevant visual explanation for the decoder-only
model in this project. Focus on the idea that each position computes a query,
compares it with earlier keys, and uses the resulting weights to combine value
vectors.

Do not worry about multi-head attention, residual connections, or the full
GPT-2 architecture yet.

## 5. Attention Is All You Need: The Original Equations

[Read the paper](https://arxiv.org/abs/1706.03762), focusing on Sections 3.2.1
(Scaled Dot-Product Attention), 3.2.2 (Multi-Head Attention), and 3.5
(Positional Encoding).

The paper describes an encoder-decoder Transformer, while this project will
first build decoder-only causal self-attention. Extract the general attention
equation, but do not copy its whole architecture:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

Questions to answer:

- Why does `Q @ K.transpose(-2, -1)` have shape `[B, T, T]`?
- What does one row of that matrix mean?
- Why can large dot products make softmax difficult to train?
- Why must future positions be made impossible before softmax in a language
  model?

## Later: Bahdanau and Multi-Head Attention

[Section 11.4, Bahdanau Attention](https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html)
is useful historical context for encoder-decoder RNNs, but it is not needed to
implement the decoder-only causal head in this milestone.

[Section 11.5, Multi-Head Attention](https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/multi-head-attention.html)
should be read after the single-head implementation works. It is the next
architecture step, not a prerequisite for understanding one head.

## 6. PyTorch Scaled Dot-Product Attention Documentation

[Read the documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)
after implementing the attention calculation manually.

Use it as a reference for tensor conventions and masking semantics, not as a
replacement for the learning implementation. In particular, notice the
`is_causal` option and the documented boolean-mask convention.

## Implementation Readiness Check

You are ready to implement one causal attention head when you can trace this
without looking it up:

```text
x:       [B, T, E]
q, k, v: [B, T, H]
scores:  [B, T, T]
weights: [B, T, T]
output:  [B, T, H]
```

The key contrast with an LSTM is that position `t` can directly weight earlier
positions rather than relying on all history being compressed into a fixed-size
recurrent state.
