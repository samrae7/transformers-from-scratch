from pathlib import Path
import random
from typing import TypedDict
import numpy as np
import torch
import torch.nn.functional as F
import math

file = Path("input.txt")
text = file.read_text()
print(f"length of dataset in characters: {len(text)}")
print(f"dataset sample:\n{text[:1000]}")

chars = sorted(set(text))
vocab_size = np.unique(chars).size

stoi = {char: i for i, char in enumerate(chars)}

itos = {i: char for char, i in stoi.items()}


def encode(text: str) -> list[int]:
    return [stoi[ch] for ch in text]


def decode(ids: list[int]) -> str:
    return "".join([itos[i] for i in ids])


def get_corpus_text():
    file = Path("input.txt")
    return file.read_text()


def get_data():
    text = get_corpus_text()
    return torch.tensor(encode(text), dtype=torch.long)


class DataSplits(TypedDict):
    train: torch.Tensor
    validation: torch.Tensor


def get_splits(data: torch.Tensor) -> DataSplits:
    data_length = data.size()[0]
    split_idx = math.floor(data_length * 90 / 100)
    train_data = data[:split_idx]
    validation_data = data[split_idx:]
    return {"train": train_data, "validation": validation_data}


def get_batch(
    split: torch.Tensor, batch_size: int, context_size: int
) -> tuple[torch.Tensor]:
    """
    x: previous N tokens when N = context_size. Shape: batch_size, context_size
    y: single next token embedding. Shape: batch_size
    """
    x_rows = []
    y_rows = []
    for i in range(batch_size):
        greatest_start = len(split) - context_size - 1
        start = random.randint(0, greatest_start)
        x_end = start + context_size
        x_row = split[start:x_end]
        y_row = split[x_end]
        x_rows.append(x_row)
        y_rows.append(y_row)
    x = torch.stack(x_rows)
    y = torch.stack(y_rows)
    return (x, y)


class MLP(torch.nn.Module):
    def __init__(self, learning_rate: float = 0.05) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embed_dim)
        self.optimizer = torch.optim.Adam(
            self.embeddings.parameters(), lr=learning_rate
        )

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

    def _get_next_token(self, token: int) -> int:
        token_tensor = torch.tensor(token)
        logits = self.embeddings(token_tensor)
        softmax = torch.nn.Softmax(dim=0)
        probs = softmax(logits)
        next_token = torch.multinomial(probs, num_samples=1).item()
        return next_token

    def generate(self, start_char: str, length: int) -> int:
        continuation = ""
        current = stoi[start_char]
        for _ in range(length):
            next_token = self._get_next_token(current)
            continuation += itos[next_token]
            current = next_token
        return continuation


def train(model: MLP, max_iters: int) -> None:
    loss_accum = 0.0
    # idx, targets = get_batch_for_overfit(test_data, 0)
    # result = decode(idx[0].tolist())
    # print(f"{result=}")
    for iter in range(max_iters):
        idx, targets = get_batch(train_data)
        loss, _ = model.forward(idx, targets)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        loss_accum += loss.item()
        if iter % 100 == 0:
            print(
                f"iter {iter}: batch_loss {loss.item()}, avg_loss {loss_accum / (iter + 1)}"
            )
    val_loss_accum = 0
    val_steps = 100
    splits = get_splits(data)
    validation_data = splits["validation"]
    for iter in range(val_steps):
        idx, targets = get_batch(validation_data)
        val_loss, _ = model.forward(idx, targets)
        val_loss_accum += val_loss
    avg_val_loss = val_loss_accum / val_steps
    print(f"{avg_val_loss=}")


if __name__ == "__main__":
    data = torch.tensor(encode(text), dtype=torch.long)

    splits = get_splits(data)
    train_data = splits["train"]
    context_size = 4
    batch_size = 10
    get_batch(batch_size=batch_size, split=train_data, context_size=4)
    # model = BigramModel()
    # before_training = model.generate("\n", 500)
    # print(f"{before_training=}")

    # train(model, 5000)
    # generated = model.generate("\n", 500)
    # print(f"{generated=}")

    # print(model.generate("T", 100))
    # print(model.generate(" ", 100))
    # print(model.generate("e", 100))
