# Transformers From Scratch Learning Roadmap

Project started: May 17, 2026.

## Expected Timeline

Available study time:

```text
3 to 4 hours per week
```

The project is expected to require roughly 72 to 84 hours, giving an estimated
duration of:

```text
18 to 28 weeks
```

This is a directional estimate rather than a deadline. The project prioritizes
mechanistic understanding, debugging practice, and controlled experiments over
finishing each milestone quickly. The RNN and LSTM bridge may move the duration
toward the upper end of the range.

## Purpose

This roadmap keeps the project focused on learning sequence modeling from first
principles. The goal is not to reach a transformer as quickly as possible. The
goal is to understand what each modeling step adds, what failure modes appear,
and why the transformer architecture is shaped the way it is.

Each milestone should leave behind:

- A small working implementation.
- Focused tests for tensor shapes and core invariants.
- A short note with observed loss, generated samples, confusing points, and
  open questions.
- At least one controlled experiment or debugging observation.

## Milestone 1: Bigram Language Model

Reference: [Bigram Model Plan](bigram-model-plan.md)

Core idea:

```text
P(next token | current token)
```

What this teaches:

- Character-level tokenization.
- Next-token prediction.
- Logits, softmax, cross-entropy, and sampling.
- Why generated text can vary even when the model is deterministic internally.
- How a learned lookup table can act as a tiny language model.

Limitations to notice:

- No memory beyond one token.
- Greedy generation can collapse into repetition.
- Output can learn local character statistics without learning words or syntax.

Completion signal:

- Training loss decreases.
- Generation produces valid decoded text.
- The limitations of a bigram model are explainable without hand-waving.

## Milestone 2: Fixed-Context MLP Language Model

Reference: [Fixed-Context MLP Language Model Plan](mlp-language-model/plan.md)

Core idea:

```text
P(next token | previous N tokens)
```

What this adds:

- A context window longer than one token.
- Token embeddings with a smaller embedding dimension than vocab size.
- Flattening multiple position embeddings into one vector.
- A nonlinear hidden layer between inputs and output logits.

Suggested shape path:

```text
idx: [batch, context]
token embeddings: [batch, context, n_embd]
flattened: [batch, context * n_embd]
logits: [batch, vocab_size]
```

What this teaches:

- Why embeddings are useful beyond direct vocab-to-vocab lookup.
- What an MLP can and cannot learn from fixed slots.
- Why position matters.
- Why fixed-context models do not scale naturally to flexible sequence lengths.

Comparison to bigram:

- Does validation loss improve?
- Does generated text show longer local structure?
- Does increasing context size help?
- What happens when context size is too small or too large?

Completion signal:

- Training and validation loss are recorded.
- A generated sample is recorded.
- The fixed-context limitation can be explained.

## Milestone 3: RNN and LSTM Bridge

Reference: [RNN and LSTM Language Model Plan](rnn-language-model/plan.md)

This milestone provides the historical and conceptual bridge between
fixed-context models and attention. Keep it tightly scoped: the goal is to
understand what transformers replaced, not to build a highly tuned recurrent
language model.

Core idea:

```text
h_t = f(x_t, h_{t-1})
P(x_{t+1} | h_t)
```

What this teaches:

- Sequence memory through hidden state.
- Why recurrent models process tokens one step at a time.
- How history becomes compressed into an evolving hidden state.
- How backpropagation through time connects repeated recurrent steps.
- Why gradients through many time steps can vanish or explode.
- Why LSTMs introduce a cell state and learned gates.
- What direct token-to-token access in attention changes.

### Step 1: Manual Vanilla RNN

Implement one small recurrent cell manually:

```text
h_t = tanh(W_x x_t + W_h h_{t-1} + b)
logits_t = W_o h_t + b_o
```

Include:

- Character embeddings.
- Explicit hidden-state initialization.
- A short sequence unrolled in a Python loop.
- Next-character logits at each time step.
- Cross-entropy over the sequence.
- Basic autoregressive generation.

Inspect:

- Hidden-state shapes across time.
- Why each step depends on the previous step.
- Gradient norms over a short and a longer unrolled sequence.

### Step 2: Manual LSTM Cell

Implement or closely derive one small LSTM cell without using `nn.LSTM`.

Understand:

- The hidden state `h_t`.
- The cell state `c_t`.
- The forget, input, candidate, and output computations.
- Why the cell-state update has an additive path:

```text
c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
```

The LSTM does not need to become a polished or heavily trained model. One
working cell, shape tests, and a short sequence pass are enough.

Comparison to MLP:

- RNNs can carry state across sequence length.
- MLPs consume a fixed context window.
- RNNs and LSTMs still process sequence positions serially.
- Historical information must pass through recurrent state at every step.

Completion signal:

- The vanilla recurrent update can be explained from first principles.
- Backpropagation through time can be described at a high level.
- The purpose of each LSTM gate and the cell state can be explained.
- The limitations of recurrent state motivate direct attention over prior
  positions.
- Stop before architecture tuning or a full GRU/LSTM comparison project.

## Milestone 4: Single-Head Causal Self-Attention

Core idea:

```text
Each token computes which previous tokens to read from.
```

What this adds:

- Token embeddings plus positional embeddings.
- Query, key, and value projections.
- Attention scores from `q @ k.T`.
- Scaling by `sqrt(head_size)`.
- Causal masking so tokens cannot see the future.
- Attention-weighted sums of value vectors.

Suggested shape path:

```text
idx: [batch, time]
x: [batch, time, n_embd]
q, k, v: [batch, time, head_size]
attention scores: [batch, time, time]
attention weights: [batch, time, time]
output: [batch, time, head_size]
logits: [batch, time, vocab_size]
```

What this teaches:

- How a token chooses information from earlier tokens.
- Why masking is required for language modeling.
- Why attention is different from fixed-window MLPs and recurrent state.
- Why scaling attention scores matters for softmax behavior and gradients.

Key experiments:

- Inspect attention weights for a short sequence.
- Remove the causal mask and observe why that leaks target information.
- Compare loss and generated samples to the MLP model.

## Milestone 5: Multi-Head Attention

Core idea:

```text
Multiple attention heads can read different relationships in parallel.
```

What this adds:

- Several independent attention heads.
- Concatenation of head outputs.
- Output projection back to `n_embd`.

What this teaches:

- Why one attention pattern may not be enough.
- How heads split representational capacity.
- Why the model dimension and head size need to line up.

Key checks:

- Each head output has expected shape.
- Concatenated output returns to the model dimension.
- The implementation still preserves causal masking.

## Milestone 6: Transformer Block

Core idea:

```text
self-attention + feed-forward network + residual paths + normalization
```

What this adds:

- Residual connections.
- Layer normalization.
- Position-wise feed-forward network.
- Dropout if useful for experimentation.

What this teaches:

- Why residual paths help gradient flow.
- Why normalization stabilizes training.
- Why attention and MLP sublayers play different roles.
- Why transformer blocks can be stacked.

Key experiments:

- Train with and without residual connections on a small model.
- Inspect gradient norms.
- Compare shallow and deeper versions.

## Milestone 7: Small Decoder-Only Transformer

Core idea:

```text
Stack transformer blocks and train a GPT-style character language model.
```

What this adds:

- Multiple transformer blocks.
- Final layer norm.
- Generation with cropped context.
- Basic hyperparameter tuning.
- Train and validation loss estimation.

What this teaches:

- How the full architecture fits together.
- How context length, embedding size, number of heads, and number of layers
  affect training.
- How to debug unstable or flat training.

Completion signal:

- The model trains stably.
- Validation loss is tracked separately from training loss.
- Generated samples are compared against bigram and MLP baselines.
- The full forward pass can be explained from token ids to logits.

## Milestone 8: Controlled Experiments and Debugging

The final phase is about building research-engineering habits.

Suggested experiments:

- Overfit a tiny dataset.
- Vary context length.
- Vary embedding dimension.
- Vary number of heads.
- Vary number of layers.
- Compare `argmax`, multinomial sampling, and temperature.
- Inspect gradient norms.
- Track train vs validation loss.

Questions to answer:

- What changes improve loss but not generated text?
- What changes make training unstable?
- What does a healthy training curve look like?
- What failure mode did each debugging tool reveal?

## Milestone 9: Documentation and Retrospective

Create a short final write-up covering:

- What each model learned that the previous model could not.
- Where training failed or behaved unexpectedly.
- Which concepts are now intuitive.
- Which concepts still need work.
- What paper or architecture would be a reasonable next implementation target.

The project is successful when the architecture is no longer just code that
runs, but a system whose parts, shapes, gradients, and failure modes can be
reasoned about directly.
