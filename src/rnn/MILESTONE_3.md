## Milestone 3: RNN Gradients, Exploding Gradients, and Clipping

This chapter should make gradient behavior in recurrent networks concrete
before moving on to LSTMs.

The goal is not to add more architecture yet. The goal is to inspect what
happens when gradients flow backward through an unrolled recurrent chain:

```text
h_0 -> h_1 -> h_2 -> ... -> h_T
```

During backpropagation, later losses send gradient signal backward through many
repeated hidden-to-hidden transformations. This is backpropagation through time
(BPTT).

## Core Questions

- What is a gradient for one parameter tensor?
- What is a gradient norm?
- How does the total model gradient norm behave during RNN training?
- How does sequence length affect gradient norms?
- What does an exploding gradient look like in the training logs?
- What does gradient clipping do?
- What does gradient clipping not solve?

## Concepts To Understand

### Gradient

A gradient is the tensor of partial derivatives for a parameter:

```text
parameter: [rows, cols]
gradient:  [rows, cols]
```

Each entry says how the loss would change if that parameter changed slightly.

### Gradient Norm

A gradient norm is one scalar summary of the size of a gradient tensor.

For an L2 norm:

```text
norm = sqrt(sum(gradient_entry^2))
```

For the whole model, the total gradient norm summarizes the combined size of
all parameter gradients.

### Exploding Gradients

An RNN repeatedly applies the hidden-to-hidden transformation:

```text
h_t = tanh(W_x x_t + W_h h_(t-1) + b)
```

During backpropagation, gradients repeatedly flow through this recurrent path.
Across many steps, those repeated effects can grow very large.

Symptoms may include:

- Loss spikes.
- `nan` loss.
- Very large gradient norms.
- Training becoming unstable when sequence length increases.

### Gradient Clipping

Gradient clipping limits the gradient norm before the optimizer step:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

It rescales gradients when their total norm is too large.

Important distinction:

```text
gradient clipping helps with exploding gradients
gradient clipping does not fix vanishing gradients
```

## Implementation Plan

### Step 1: Add Gradient Norm Measurement

Add a helper that computes the total gradient norm after `loss.backward()` and
before `optimizer.step()`.

Record:

```text
iteration
loss
total_gradient_norm
```

Optionally also inspect individual parameter norms:

```text
embeddings.weight
input_to_hidden.weight
hidden_to_hidden.weight
hidden_to_output.weight
```

### Step 2: Compare Sequence Lengths

Train or run short diagnostic passes with:

```text
sequence_length = 8
sequence_length = 32
sequence_length = 64
sequence_length = 128
```

Keep everything else fixed:

```text
embed_dim
hidden_dim
batch_size
learning_rate
optimizer
number of iterations
```

Record whether gradient norms change as the unroll length increases.

### Step 3: Run Without Clipping

Temporarily remove or disable:

```python
clip_grad_norm_
```

Observe:

- Does training remain stable?
- Do gradient norms spike?
- Does longer sequence length make instability more likely?

Do not assume clipping is necessary. Look for evidence.

### Step 4: Add Clipping Back

Add clipping after `loss.backward()` and before `optimizer.step()`:

```python
loss.backward()
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

Note: `clip_grad_norm_` returns the norm before clipping.

Compare:

- Loss stability.
- Gradient norm logs.
- Generated samples.

### Step 5: Write a Short Observation

Record:

```text
sequence_length tested
whether clipping was enabled
typical gradient norm
largest observed gradient norm
whether loss was stable
what changed after clipping
```

## Experiments

Suggested first experiment:

```text
sequence_length = 32
gradient clipping off
max_iters = 500 to 1000
```

Then:

```text
sequence_length = 128
gradient clipping off
max_iters = 500 to 1000
```

Then repeat with:

```text
gradient_clip_max_norm = 1.0
```

Do not tune many variables at once.

## Where LSTM Fits

Do not include the LSTM implementation in this chapter.

This chapter is about observing the vanilla RNN's gradient behavior and
understanding why clipping exists.

The next chapter should introduce the LSTM cell:

```text
c_t = f_t * c_(t-1) + i_t * g_t
h_t = o_t * tanh(c_t)
```

The LSTM is related because it was designed partly to improve long-range memory
and gradient flow, but it deserves its own focused implementation chapter.

## Completion Signal

- Gradient norms can be measured and logged.
- Sequence lengths have been compared while keeping other settings fixed.
- Training has been observed with and without clipping.
- Gradient clipping can be explained as norm rescaling for exploding gradients.
- The difference between exploding and vanishing gradients is clear.
- The motivation for LSTM gated memory is sharper than before.
