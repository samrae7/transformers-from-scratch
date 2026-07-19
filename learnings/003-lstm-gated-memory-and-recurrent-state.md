# LSTM: Gated Memory and Recurrent State

## Purpose

This chapter introduces the questions to answer before implementing an LSTM
language model. An LSTM is a recurrent neural network cell designed to give a
model more control over what information it retains across a sequence.

The aim is to understand the cell before treating it as a PyTorch component.

## Starting Point: The Vanilla RNN

A vanilla RNN carries one hidden state forward through a sequence:

```text
h_t = tanh(W_x x_t + W_h h_(t-1) + b)
```

Questions to revisit:

- What information must the single hidden state represent?
- Why can repeatedly updating one vector make old information difficult to
  preserve?
- How do long sequences affect gradients flowing through the recurrent path?

## The LSTM State

An LSTM carries two states at every timestep:

```text
h_t: hidden state
c_t: cell state
```

At timestep `t`, the cell receives:

```text
x_t, h_(t-1), c_(t-1)
```

and produces:

```text
h_t, c_t
```

Questions to answer:

- What different jobs do `h_t` and `c_t` perform?
- Why is the cell state a useful path for carrying information over time?
- How does the model expose only part of its memory through the hidden state?

## Gates

The LSTM uses learned gates to control memory. The main components to study
are:

```text
forget gate
input gate
candidate memory
output gate
```

For each component, identify:

- Its inputs.
- Its output shape.
- Whether it uses sigmoid or tanh, and why.
- How it affects the cell state or hidden state.

## Pointwise Memory Control

LSTM gates act pointwise on vectors. A gate is not one global decision; each
dimension of the cell state can be kept, changed, or exposed differently.

Questions to answer:

- How does pointwise multiplication differ from a linear layer?
- What does a sigmoid output near `0` or near `1` mean for one memory
  dimension?
- Why is per-dimension control useful instead of one shared keep/forget
  decision?

## Cell-State Update

The central LSTM update has this broad form:

```text
new memory = retained old memory + selected new information
```

Study questions:

- Which gate controls retained memory?
- Which gate controls new information written to memory?
- Why might an additive memory update preserve gradients differently from the
  repeatedly transformed hidden state in a vanilla RNN?

## Implementation Checks for Later

When implementing the cell, verify:

- The input, hidden state, and cell state have compatible batch dimensions.
- Each gate has one value per hidden dimension.
- The cell returns both `h_t` and `c_t`.
- A sequence forward pass carries both states from one timestep to the next.
- Training still predicts one next-token distribution per sequence timestep.

## Hands-On Implementation Path

The implementation is part of the learning, not an optional follow-up. Work
through the manual LSTM steps in the
[RNN and LSTM Language Model Plan](../docs/rnn-language-model/plan.md):

1. Implement one LSTM timestep, taking `x_t`, `h_(t-1)`, and `c_(t-1)`.
2. Test gate, hidden-state, and cell-state shapes before unrolling time.
3. Run the cell across a short sequence while carrying both states forward.
4. Produce next-token logits at each position and verify a backward pass.
5. Compare the state flow with the vanilla RNN before deciding whether a full
   training and generation pass adds further insight.

The goal is to implement the mechanics yourself, using the questions in this
chapter and the existing plan as guides rather than copying a completed cell.

## Key Questions Before Moving On

- Can I explain why an LSTM is still a type of RNN?
- Can I distinguish hidden state from cell state?
- Can I explain what each gate controls without memorizing equations?
- Can I relate the gated memory path to vanishing-gradient and long-context
  problems in a vanilla RNN?
