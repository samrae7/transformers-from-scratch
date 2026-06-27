## Milestone 2: Stateful Vanilla RNN Generation

Updated generation so the model processes the prompt once, keeps the final
hidden state, and then generates one token at a time while carrying the hidden
state forward.

This avoids recomputing the entire generated prefix on every step and better
matches the intended RNN mechanism:

```text
prompt -> final hidden state
new token + previous hidden -> next hidden -> next-token logits
```

Configuration:

```text
sequence_length = 32
embed_dim = 32
hidden_dim = 128
batch_size = 32
learning_rate = 1e-3
max_iters = 20,000
gradient_clip_max_norm = 1.0
temperature = 0.8
```

Results:

```text
recent training loss ~= 1.58
validation loss ~= 1.75
```

Observation:

The generated text shows much stronger Shakespeare-like structure than the
earlier RNN runs. It produces speaker labels, line breaks, punctuation, and
some plausible phrase fragments. It still invents names and words, and it does
not maintain coherent meaning over long spans.

Key learning:

The hidden state acts as a compressed running summary of the sequence. During
generation, carrying the hidden state forward lets the model condition on the
prompt and generated history without replaying the whole sequence each time.
This also makes clear why RNNs are sequential: each new prediction depends on
the hidden state produced by the previous step.

Next:

Inspect gradient norms across different sequence lengths, then move to a manual
LSTM cell to understand how gated memory changes the recurrent update.
