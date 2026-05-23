from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
import math

file = Path("input.txt")
text = file.read_text()

chars = sorted(set(text))
vocab_size = np.unique(chars).size

stoi = {char: i for i, char in enumerate(chars)}

itos = {i: char for char, i in stoi.items()}


def encode(text: str) -> list[int]:
    return [stoi[ch] for ch in text]


def decode(ids: list[int]) -> str:
    return "".join([itos[i] for i in ids])


# embedding layers and cross-entropy targets expect integer class indices as long tensors.
data = torch.tensor(encode(text), dtype=torch.long)

data_length = data.size()[0]
split_idx = math.floor(data_length * 90 / 100)
test_data = data[:split_idx]
validation_data = data[split_idx:]

batch_size = 5
block_size = 10


def get_batch(split: torch.Tensor):
    """Generate a batch of data of inputs x and targets y"""
    x_rows = []
    y_rows = []
    for i in range(batch_size):
        greatest_start = len(split) - block_size - 1
        start = random.randint(0, greatest_start)
        x_row = split[start : start + block_size]
        y_row = split[start + 1 : start + block_size + 1]
        x_rows.append(x_row)
        y_rows.append(y_row)
    x_row = torch.stack(x_rows)
    y_row = torch.stack(y_rows)
    return (x_row, y_row)


class BigramModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.embeddings(idx)
        logits_for_loss = logits.flatten(0, 1)
        if targets is None:
            return (None, logits)
        targets_for_loss = targets.flatten()
        loss = F.cross_entropy(logits_for_loss, targets_for_loss)
        return loss, logits
