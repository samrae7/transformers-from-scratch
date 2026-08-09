import torch
from torch import nn
from torch.nn import functional as F

from language_modeling.data import encode_data, get_corpus_text, get_splits
from language_modeling.tokenizer import CharTokenizer
from rnn.data import get_batch
from transformer.attention import (
    SingleHeadCausalSelfAttention,
    TokenAndPositionEmbedding,
)


class SingleHeadTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        tokenizer: CharTokenizer,
        max_sequence_length: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = TokenAndPositionEmbedding(
            vocab_size=tokenizer.vocab_size,
            embed_dim=embed_dim,
            max_sequence_length=max_sequence_length,
        )
        self.attention = SingleHeadCausalSelfAttention(
            embed_dim=embed_dim, head_dim=head_dim
        )
        self.output = nn.Linear(head_dim, tokenizer.vocab_size)

    def forward(self, input: torch.Tensor):
        x = self.embedding(input)
        o = self.attention(x)
        return self.output(o)


def train(
    model: SingleHeadTransformer,
    train_data: torch.Tensor,
    batch_size: int,
    max_iters: int,
    optimizer: torch.optim.Optimizer,
):
    recent_losses = []

    for iter in range(max_iters):
        idx, targets = get_batch(train_data, batch_size=batch_size)
        logits = model(idx)
        logits = logits.flatten(0, 1)
        targets = targets.flatten(0, 1)

        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recent_losses.append(loss.item())
        if iter % 100 == 0:
            losses = recent_losses[-100:]

            print(
                f"iter {iter}: "
                f"loss_mean={sum(losses) / len(losses):.4f}, "
                f"loss_max={max(losses):.4f}, "
            )


if __name__ == "__main__":
    # get train splits
    text = get_corpus_text()
    tokenizer = CharTokenizer(text)
    data = encode_data(text, tokenizer=tokenizer)
    splits = get_splits(data)
    train_data = splits["train"]
    # call train
    embed_dim = 32
    head_dim = 32
    batch_size = 16
    sequence_length = 32
    lr = 0.001
    max_iters = 1000

    model = SingleHeadTransformer(
        embed_dim=embed_dim,
        head_dim=32,
        tokenizer=tokenizer,
        max_sequence_length=sequence_length,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(
        model=model,
        train_data=train_data,
        batch_size=batch_size,
        max_iters=max_iters,
        optimizer=optimizer,
    )
