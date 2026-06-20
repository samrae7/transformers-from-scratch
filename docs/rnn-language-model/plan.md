# RNN and LSTM Language Model Plan

## Purpose

This chapter comes after the fixed-context MLP and before attention.

The goal is to understand how recurrent neural networks process sequences,
what the hidden state represents, how gradients flow through time, and why
LSTMs were introduced.

The model progression is:

```text
bigram:
P(next token | current token)

fixed-context MLP:
P(next token | previous N tokens)

RNN:
P(next token | learned summary of previous tokens)
```

The RNN removes the MLP's architectural dependence on a fixed number of input
positions. It does this by updating a fixed-size hidden state at every time
step:

```text
h_t = tanh(W_x x_t + W_h h_(t-1) + b_h)
```

The LSTM section then examines how gated memory addresses some weaknesses of
this simple recurrent update.

## Reading

Read these sections in order:

1. [Dive into Deep Learning: Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html)
2. [Dive into Deep Learning: RNN Implementation from Scratch](https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html)
3. [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
4. [Dive into Deep Learning: Backpropagation Through Time](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html)
5. [Dive into Deep Learning: Long Short-Term Memory](https://d2l.ai/chapter_recurrent-modern/lstm.html)

Use the readings to understand equations and shapes. Implement the model
independently rather than copying the reference implementation.

## Core Innovation

The MLP receives all context tokens explicitly:

```text
previous N tokens
-> embeddings
-> flatten
-> hidden representation
```

The RNN receives one token at a time:

```text
current token + previous hidden state
-> new hidden state
-> next-token logits
```

For a sequence such as `"First"`:

```text
F + h_0 -> h_1 -> predict i
i + h_1 -> h_2 -> predict r
r + h_2 -> h_3 -> predict s
s + h_3 -> h_4 -> predict t
```

The same weights are reused at every time step. Sequence length changes the
number of recurrent steps, not the shapes of the parameter matrices.

## Hidden State Intuition

Both the MLP and RNN create intermediate hidden representations, but they use
them differently:

```text
MLP hidden representation:
summary of one explicitly supplied fixed context

RNN hidden state:
running summary passed from one time step to the next
```

The RNN hidden state is not a perfect record of the past. It is a fixed-size
vector that the model learns to update in ways useful for prediction.

## Learning Goals

By the end of this chapter, be able to explain:

- How a Markov assumption relates to the bigram and fixed-context MLP models.
- Why an RNN can process different sequence lengths with the same parameters.
- What `x_t`, `h_(t-1)`, `h_t`, and `logits_t` represent.
- Why the same recurrent weights are reused at every time step.
- Why RNN computation is sequential.
- How unrolling through time creates a deeper computation graph.
- Why gradients can vanish or explode across many recurrent steps.
- How the LSTM cell state and gates change the recurrent update.
- What limitations remain that motivate attention.

## Part 1: Manual Vanilla RNN

### Step 1: Create Sequence Batches

Return to the next-token batch structure used by the bigram model:

```text
x: [batch, time]
y: [batch, time]
```

For one example:

```text
x: "Firs"
y: "irst"
```

Every position has a next-token target.

Questions to answer:

- Why does the RNN predict at every time step?
- How does this differ from the MLP's one target per context window?
- Why must each row of `y` be shifted one token ahead of `x`?

### Step 2: Embed the Input Sequence

Use a character embedding table:

```text
nn.Embedding(vocab_size, n_embd)
```

Expected shape:

```text
idx: [batch, time]
x:   [batch, time, n_embd]
```

Do not flatten the time and embedding dimensions. The RNN must process the time
dimension in order.

Questions to answer:

- Why did the MLP flatten time while the RNN must preserve it?
- What does `x[:, t, :]` represent?

### Step 3: Initialize the Hidden State

Create an initial hidden state:

```text
h_0: [batch, hidden_dim]
```

Start with zeros.

Questions to answer:

- Why is there one hidden-state vector per batch example?
- Why is `hidden_dim` independent of sequence length?
- What information can `h_0` contain when it is initialized to zero?

### Step 4: Implement One Recurrent Step

Implement the recurrent update manually:

```text
h_t = tanh(W_x x_t + W_h h_(t-1) + b_h)
```

One possible layer arrangement is:

```text
x_t:
[batch, n_embd]

input projection:
Linear(n_embd, hidden_dim)

h_(t-1):
[batch, hidden_dim]

hidden projection:
Linear(hidden_dim, hidden_dim)

h_t:
[batch, hidden_dim]
```

Then produce next-token logits:

```text
logits_t = W_o h_t + b_o

[batch, hidden_dim]
-> Linear(hidden_dim, vocab_size)
-> [batch, vocab_size]
```

Initially test one step in isolation.

Questions to answer:

- Why must the hidden-to-hidden projection be square?
- Why are the input and previous-hidden projections added?
- Why is `tanh` applied after combining them?
- Which parameters are reused at the next time step?

### Step 5: Unroll Across Time

Loop over the time dimension:

```text
for t in range(time):
    x_t = embeddings[:, t, :]
    h_t = recurrent_step(x_t, h_(t-1))
    logits_t = output(h_t)
```

Collect logits from every time step.

Expected shape path:

```text
idx:        [batch, time]
embeddings: [batch, time, n_embd]
h_t:        [batch, hidden_dim]
logits_t:   [batch, vocab_size]
logits:     [batch, time, vocab_size]
```

Questions to answer:

- Why can time steps not be computed independently?
- What is shared across time, and what changes at each step?
- What does the final hidden state summarize?

### Step 6: Compute Sequence Loss

Cross-entropy needs:

```text
logits:  [batch * time, vocab_size]
targets: [batch * time]
```

Flatten only for the loss:

```text
[batch, time, vocab_size]
-> [batch * time, vocab_size]
```

Questions to answer:

- Why is flattening appropriate here but not before recurrence?
- How does each time step contribute to the total loss?
- How does this compare with the bigram loss computation?

### Step 7: Overfit a Tiny Sequence

Before training on Tiny Shakespeare, repeatedly train on one short sequence.

Completion signal:

- Loss falls substantially.
- The model can reproduce the short sequence.
- Every recurrent parameter receives a finite gradient.

If this fails, inspect:

- Input and target shifting.
- Hidden-state shape.
- Whether `h_t` is passed to the next step.
- Whether all parameters belong to the optimizer.
- Gradient values for each recurrent matrix.

### Step 8: Train the Character RNN

Train on Tiny Shakespeare with the existing tokenizer and train/validation
split.

Start small:

```text
sequence_length = 32
n_embd = 16
hidden_dim = 64
batch_size = 32
learning_rate = 1e-3
```

These values are starting points, not targets for tuning.

Track:

- Recent training loss.
- Validation loss.
- Gradient norm.
- Generated samples.

### Step 9: Generate Autoregressively

During generation:

1. Initialize the hidden state.
2. Process the prompt one token at a time.
3. Use the final prompt hidden state to predict the next token.
4. Sample from the vocabulary distribution.
5. Feed the sampled token and previous hidden state into the next step.

Unlike the MLP, generation does not require padding to a fixed context width.

Questions to answer:

- Why should the prompt update hidden state before sampling?
- Why can the RNN continue beyond the training sequence length?
- Why does generation remain sequential?
- What information is lost if hidden state is reset at every generated token?

## Part 2: Backpropagation Through Time

An unrolled RNN is one computation graph with repeated use of the same
parameters:

```text
h_0 -> h_1 -> h_2 -> h_3 -> ... -> h_T
```

The loss at later positions can send gradients backward through many hidden
state updates. This is backpropagation through time.

### Gradient Experiment

Run the same model on multiple sequence lengths, for example:

```text
8, 32, 128
```

After `loss.backward()`, record gradient norms for:

```text
input projection weights
hidden-to-hidden weights
output projection weights
embedding weights
```

Change one variable at a time and compare:

- Sequence length.
- `tanh` versus no activation as a diagnostic only.
- Different hidden-to-hidden initializations if needed.
- Training with and without gradient clipping.

Questions to answer:

- Why does the gradient repeatedly multiply through recurrent transformations?
- What would make those products shrink?
- What would make them grow?
- What symptom would an exploding gradient produce during training?

Do not add gradient clipping until exploding gradients are actually observed.
Treat clipping as a deliberate response to evidence.

## Part 3: Manual LSTM Cell

### Why Add an LSTM?

A vanilla RNN repeatedly transforms its hidden state through `tanh`. This can
make long-range information and gradients difficult to preserve.

An LSTM introduces:

```text
h_t: exposed hidden state
c_t: cell state or memory path
```

The cell state is updated additively:

```text
c_t = f_t * c_(t-1) + i_t * g_t
h_t = o_t * tanh(c_t)
```

### Step 10: Understand the Four Computations

For the current input and previous hidden state, compute:

```text
forget gate:
f_t = sigmoid(...)

input gate:
i_t = sigmoid(...)

candidate memory:
g_t = tanh(...)

output gate:
o_t = sigmoid(...)
```

Interpret them as:

- `f_t`: how much previous cell state to retain.
- `i_t`: how much candidate memory to write.
- `g_t`: proposed new memory content.
- `o_t`: how much cell-state information to expose as hidden state.

Questions to answer:

- Why do the gates use sigmoid?
- Why does candidate memory use `tanh`?
- Why are `f_t`, `i_t`, `g_t`, `o_t`, `c_t`, and `h_t` the same width?
- Which operation creates the more direct gradient path through time?

### Step 11: Implement One LSTM Cell

Implement a single time-step function without using `nn.LSTM`.

Inputs:

```text
x_t:     [batch, n_embd]
h_(t-1): [batch, hidden_dim]
c_(t-1): [batch, hidden_dim]
```

Outputs:

```text
h_t: [batch, hidden_dim]
c_t: [batch, hidden_dim]
```

Test:

- Every gate shape.
- New hidden-state shape.
- New cell-state shape.
- Values from sigmoid gates lie between 0 and 1.
- Gradients reach the cell parameters.

### Step 12: Unroll a Short LSTM Sequence

Run the cell over a short sequence and produce next-token logits.

This does not need to become a fully tuned language model. A working sequence
pass, scalar loss, successful backward pass, and basic generation are enough.

Compare:

```text
vanilla RNN:
one recurrent hidden state

LSTM:
hidden state + gated cell state
```

## Comparison With Earlier Models

Record the essential architectural differences:

```text
Bigram:
one-token context

Fixed-context MLP:
explicit fixed window, combined by flattening

Vanilla RNN:
history compressed into recurrent hidden state

LSTM:
history carried through gated hidden and cell states
```

Questions to answer:

- What can the RNN represent that the fixed-context MLP cannot?
- What makes RNN training and inference sequential?
- Why can distant information still be difficult to retain?
- How does the LSTM create a better memory path?
- Why might attention be preferable to compressing all history into one state?

## Milestone Complete When

- Sequence batches produce `[batch, time]` inputs and targets.
- A manual vanilla recurrent step has correct tensor shapes.
- The RNN returns `[batch, time, vocab_size]` logits.
- Cross-entropy computes over all time positions.
- The RNN can overfit a tiny sequence.
- Training and generation work on character data.
- Gradient norms have been compared across sequence lengths.
- One manual LSTM cell passes shape and gradient checks.
- The roles of hidden state, cell state, and each LSTM gate are explainable.
- The recurrent bottleneck and sequential computation motivate attention.

Stop here. Do not expand this milestone into bidirectional RNNs, stacked LSTMs,
GRUs, sequence-to-sequence systems, or extensive hyperparameter tuning.
