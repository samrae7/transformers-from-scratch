import random
import torch
import torch.nn.functional as F

from language_modeling.data import encode_data, get_corpus_text, get_splits
from language_modeling.tokenizer import CharTokenizer


def get_batch(
    split: torch.Tensor, batch_size: int, context_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
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
    def __init__(
        self,
        context_size: int,
        vocab_size: int,
        embed_dim=32,
        hidden_dim=64,
    ) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embed_dim)
        #  linear takes in_features, out_features
        self.hidden = torch.nn.Linear(embed_dim * context_size, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, vocab_size)
        self.context_size = context_size

    def forward(
        self,
        idx: torch.Tensor,
        # targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.embeddings(idx)
        # Flatten the N token embeddings into one feature vector per example,
        # allowing the next linear layer to combine information across the context.
        flattened = embeddings.flatten(start_dim=1)
        hidden = torch.tanh(self.hidden(flattened))
        logits = self.output(hidden)
        return logits

    def _get_next_token(self, tokens: list[int]) -> int:
        device = self.embeddings.weight.device
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self(x)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs[0], num_samples=1).item()
        return next_token

    # disables gradient tracking -> less mem usage, faster
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        length: int,
        tokenizer: CharTokenizer,
        padding_char="\n",
    ) -> int:
        # If we had eg, dropout for training, this would disable it for inference
        self.eval()

        generated = tokenizer.encode(prompt)
        padding_token = tokenizer.stoi[padding_char]

        for _ in range(length):
            context = generated[(-1 * self.context_size) :]
            context = [padding_token] * (self.context_size - len(context)) + context

            next_token = self._get_next_token(context)
            generated.append(next_token)
        return tokenizer.decode(generated)


def train(
    model: MLP,
    train_data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    context_size: int,
    max_iters: int,
) -> None:
    model.train()

    loss_accum = 0.0
    recent_losses = []

    for iter in range(max_iters):
        idx, targets = get_batch(
            split=train_data, batch_size=batch_size, context_size=context_size
        )
        logits = model(idx)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_accum += loss.item()
        recent_losses.append(loss.item())
        if iter % 100 == 0:
            print(
                f"iter {iter}: batch_loss {loss.item()}, avg_loss {loss_accum / (iter + 1)}, recent_loss {sum(recent_losses[-100:]) / 100}"
            )
    val_loss_accum = 0
    val_steps = 100
    splits = get_splits(data)
    validation_data = splits["validation"]
    for iter in range(val_steps):
        idx, targets = get_batch(
            split=validation_data, batch_size=batch_size, context_size=context_size
        )
        val_logits = model(idx)
        val_loss = F.cross_entropy(val_logits, targets)
        val_loss_accum += val_loss.item()
    avg_val_loss = val_loss_accum / val_steps
    print(f"{avg_val_loss=}")


if __name__ == "__main__":
    text = get_corpus_text()
    tokenizer = CharTokenizer(text)
    data = encode_data(text, tokenizer=tokenizer)
    splits = get_splits(data)
    train_data = splits["train"]

    context_size = 8
    lr = 0.001

    model = MLP(
        context_size=context_size,
        vocab_size=tokenizer.vocab_size,
        hidden_dim=128,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(
        model=model,
        train_data=train_data,
        batch_size=32,
        context_size=context_size,
        max_iters=20000,
        optimizer=optimizer,
    )
    generated = model.generate(
        "Wherefore art thou Romeo?", length=1000, tokenizer=tokenizer
    )
    print(f"generated:\n{generated}")
