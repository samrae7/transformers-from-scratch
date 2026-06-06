from pathlib import Path
import random
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


# embedding layers and cross-entropy targets expect integer class indices as long tensors.
data = torch.tensor(encode(text), dtype=torch.long)

data_length = data.size()[0]
split_idx = math.floor(data_length * 90 / 100)
test_data = data[:split_idx]
validation_data = data[split_idx:]

batch_size = 100
block_size = 50


def get_batch_for_overfit(split, start):
    x_row = split[start : start + block_size]
    y_row = split[start + 1 : start + block_size + 1]

    x = x_row.unsqueeze(0).repeat(batch_size, 1)
    y = y_row.unsqueeze(0).repeat(batch_size, 1)

    return x, y


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
    def __init__(self, learning_rate: float = 0.05) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, vocab_size)
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


def train(model: BigramModel, max_iters: int) -> None:
    loss_accum = 0.0
    # idx, targets = get_batch_for_overfit(test_data, 0)
    # result = decode(idx[0].tolist())
    # print(f"{result=}")
    for iter in range(max_iters):
        idx, targets = get_batch(test_data)
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
    for iter in range(val_steps):
        idx, targets = get_batch(validation_data)
        val_loss, _ = model.forward(idx, targets)
        val_loss_accum += val_loss
    avg_val_loss = val_loss_accum / val_steps
    print(f"{avg_val_loss=}")


if __name__ == "__main__":
    model = BigramModel()
    before_training = model.generate("\n", 500)
    print(f"{before_training=}")

    train(model, 5000)
    generated = model.generate("\n", 500)
    print(f"{generated=}")

    print(model.generate("T", 100))
    print(model.generate(" ", 100))
    print(model.generate("e", 100))
