import torch
import random
from torch import nn

from language_modeling.data import encode_data, get_corpus_text, get_splits
from language_modeling.tokenizer import CharTokenizer


def get_batch(
    split: torch.Tensor, batch_size: int = 5, sequence_length: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Output tuple:
    x: input sequence. Shape: batch_size, sequence_length
    y: target sequence (same as x but shifted to the left): Shape: batch_size, sequence_length
    """
    x = []
    y = []
    greatest_start_idx = len(split) - sequence_length - 1
    for _ in range(batch_size):
        start_idx = random.randint(0, greatest_start_idx)
        end_idx = start_idx + sequence_length
        x_row = split[start_idx:end_idx]
        y_row = split[start_idx + 1 : end_idx + 1]
        x.append(x_row)
        y.append(y_row)
    x = torch.stack(x)
    y = torch.stack(y)
    return x, y


if __name__ == "__main__":
    text = get_corpus_text()
    tokenizer = CharTokenizer(text)
    data = encode_data(text, tokenizer=tokenizer)
    splits = get_splits(data)
    train_data = splits["train"]


class RNN(nn.Module):
    def __init__(
        self, tokenizer: CharTokenizer, embed_dim: int = 10, hidden_dim: int = 32
    ):
        super().__init__()
        vocab_size = tokenizer.vocab_size
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        self.hidden_dim = hidden_dim
        self.input_to_hidden = nn.Linear(embed_dim, hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        # later for generation can pass hidden as a param to forward
        device = input_tokens.device
        batch_size, sequence_length = input_tokens.shape
        hidden = self._init_hidden(batch_size, device)
        embs = self.embeddings(input_tokens)
        hidden_states = []
        for t in range(sequence_length):
            x_t = embs[:, t, :]
            new_h = torch.tanh(
                self.input_to_hidden(x_t) + self.hidden_to_hidden(hidden)
            )
            hidden = new_h
            hidden_states.append(new_h)
        hidden_states = torch.stack(hidden_states, dim=1)
        logits = self.hidden_to_output(hidden_states)
        return logits

    def _init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros((batch_size, self.hidden_dim), device=device)
