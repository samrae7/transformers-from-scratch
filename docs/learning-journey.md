# Learning Journey Overview

This is the high-level map of the project: what has been covered, where the
current work is, and what remains.

## Status Map

```text
[1] Bigram Language Model
    P(next token | current token)

    Learned:
    - character tokenization
    - logits over vocabulary
    - cross-entropy
    - probabilistic sampling
    - one-token context limitation

    Status: Done


        |
        v


[2] Fixed-Context MLP
    P(next token | previous N tokens)

    Learned:
    - token embeddings
    - flattening fixed context windows
    - hidden layer + tanh
    - vocabulary-sized logits
    - fixed context length
    - rolling-context generation

    Status: Done


        |
        v


[3] Vanilla RNN
    h_t = tanh(W_x x_t + W_h h_(t-1) + b)

    Learned:
    - sequence batches: [batch, time]
    - hidden state as recurrent sequence summary
    - loop over time inside forward()
    - logits at every time step: [batch, time, vocab_size]
    - sequence loss by flattening batch and time
    - stateful generation by carrying hidden state forward
    - gradient clipping introduced

    Status: Done


        |
        v


[4] RNN Gradients
    Backpropagation Through Time

    Learned:
    - gradient norms
    - exploding gradients
    - clipping as norm rescaling
    - sequence length comparisons
    - training with and without clipping

    Status: Done


        |
        v


[5] LSTM
    gated recurrent memory

    Learned:
    - hidden state h_t
    - cell state c_t
    - forget gate
    - input gate
    - candidate memory
    - output gate
    - why the additive cell-state path helps memory and gradient flow

    Status: Done


        |
        v


[6] Single-Head Causal Self-Attention
    each token directly reads from previous tokens

    Currently covering:
    - positional embeddings
    - queries, keys, and values
    - attention scores: q @ k.T
    - scaling by sqrt(head_size)
    - causal masking
    - attention-weighted value sums

    Status: Current


        |
        v


[7] Multi-Head Attention
    multiple attention patterns in parallel

    Still to cover:
    - independent heads
    - concatenation
    - output projection
    - model dimension vs head dimension

    Status: Not started


        |
        v


[8] Transformer Block
    attention + feed-forward + residual + normalization

    Still to cover:
    - residual connections
    - layer normalization
    - position-wise feed-forward networks
    - stacking blocks
    - training stability

    Status: Not started


        |
        v


[9] Decoder-Only Transformer
    small GPT-style character model

    Still to cover:
    - stacked transformer blocks
    - final layer norm
    - cropped-context generation
    - train/validation curves
    - comparison against bigram, MLP, RNN, and LSTM baselines

    Status: Not started


        |
        v


[10] Controlled Experiments and Debugging
     research-engineering practice

     Still to cover:
     - overfit tiny datasets
     - vary context length
     - vary embedding size
     - vary heads and layers
     - inspect gradient norms
     - compare sampling strategies and temperatures
     - write up failure modes

     Status: Ongoing, then final phase
```

## Current Position

The project is currently here:

```text
Bigram -> Fixed-Context MLP -> Vanilla RNN -> RNN gradients -> LSTM
    -> Single-Head Causal Self-Attention
```

The current chapter is about implementing single-head causal self-attention and
understanding how it lets each token directly read from earlier token
representations instead of compressing history into recurrent state.

## Why This Order

Each model changes how sequence context is represented:

```text
Bigram:
uses only the current token

Fixed-context MLP:
uses an explicit fixed window

Vanilla RNN:
compresses history into a recurrent hidden state

LSTM:
adds gated cell memory to improve long-range retention

Attention:
lets tokens directly read from previous token representations

Transformer:
stacks attention, feed-forward networks, residual paths, and normalization
```

The point of the journey is not just to make a model work. The point is to
understand what each architecture adds, what failure modes it introduces, and
why the transformer architecture replaced earlier sequence-modeling patterns.
