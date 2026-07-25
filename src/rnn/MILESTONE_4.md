## Milestone 4: Manual LSTM Language Model

Implemented a manual LSTM cell, unrolled it across character sequences, and
trained it as a character-level language model.

The LSTM carries two states across timesteps:

```text
x_t, h_(t-1), c_(t-1) -> h_t, c_t
```

During generation, the prompt is processed once to obtain the final
`(h_t, c_t)` state. Each sampled character is then processed with that carried
state to produce logits for the next character.

Configuration:

```text
sequence_length = 32
embed_dim = 32
hidden_dim = 128
batch_size = 32
learning_rate = 1e-3
max_iters = 20,000
sampling temperature = 1.0
```

Results:

The rolling 100-batch training loss stabilised around `1.46` near the end of
training:

```text
iter 19200: loss_mean=1.4597, loss_max=1.5827
iter 19600: loss_mean=1.4593, loss_max=1.6166
iter 19900: loss_mean=1.4614, loss_max=1.5735
```

This is a stable run. The modest variation between windows is expected because
each window contains randomly sampled training sequences. A cross-entropy of
about `1.46` corresponds to perplexity of roughly `exp(1.46) = 4.3`.

Observation:

The generated text has recognisable Shakespeare-like structure: speaker
labels, line breaks, punctuation, dialogue-like turns, and mostly word-like
character sequences. It also contains invented words, blended speaker names
such as `ROMEONARE`, repeated fragments, and semantic drift over longer spans.

This is expected for a small character-level model. It learns strong local
patterns without reliably preserving detailed meaning across a long
continuation.

Comparison with the stateful vanilla RNN:

The RNN milestone used the same sequence length, embedding width, hidden
width, batch size, learning rate, and iteration count. Its final reported
training loss was about `1.58`, compared with the LSTM's `1.46`.

```text
stateful RNN training loss: about 1.58  -> perplexity about 4.85
LSTM training loss:         about 1.46  -> perplexity about 4.31
```

This is a reduction of about `0.12` nats in training cross-entropy, or about
an 11% reduction in perplexity. It suggests that the LSTM's gated memory made
next-character prediction easier in this run.

This is a useful but not perfectly controlled comparison: the RNN used
gradient clipping with maximum norm `1.0`, while the LSTM baseline did not.
The RNN also sampled with temperature `0.8` and the LSTM sample used `1.0`;
temperature affects the appearance of generated text but not training loss.
The RNN recorded validation loss around `1.75`; the LSTM validation loss was
not measured in this run, so generalisation cannot yet be compared directly.

Key learning:

The hidden state `h_t` is projected to next-character logits, while the cell
state `c_t` is internal gated memory. Both states must be carried during LSTM
generation. They preserve the prompt's influence as fixed-width vectors, but
that compression still limits how precisely the model can retain long-range
context.

Next:

If a stricter comparison is useful, evaluate the LSTM on the validation split
and generate both models from the same prompt, temperature, and length. Then
compare formatting, repetition, and longer-range coherence.
