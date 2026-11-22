import torch
from torch import nn
from torch.nn import functional as F

from language_modeling.data import encode_data, get_corpus_text, get_splits
from language_modeling.tokenizer import CharTokenizer
from rnn.data import get_batch
from transformer.ff_mlp import FFMLP
from transformer.multi_head_attention import MultiHeadAttention
from transformer.single_head_attention import (
    TokenAndPositionEmbedding,
)


def plot_loss(val_losses, train_losses):
    max_loss = max(train_losses + val_losses)

    for train, val in zip(train_losses, val_losses):
        train_bar = "#" * int((train / max_loss) * 40)
        val_bar = "#" * int((val / max_loss) * 40)

        print(f"train {train:.2f} | {train_bar}")
        print(f"val {val:.2f} | {val_bar}")
        print()


@torch.no_grad()
def get_avg_val_loss(validation_data: torch.Tensor, iters: int, model: nn.Module):
    val_loss_accum = 0
    for _ in range(iters):
        idx, targets = get_batch(
            split=validation_data,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        val_logits = model(idx)
        val_logits = val_logits.flatten(0, 1)
        targets = targets.flatten(0, 1)
        val_loss = F.cross_entropy(val_logits, targets)
        val_loss_accum += val_loss.item()
    avg_val_loss = val_loss_accum / iters
    return avg_val_loss


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        num_heads: int,
        tokenizer: CharTokenizer,
        max_sequence_length: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.embedding = TokenAndPositionEmbedding(
            vocab_size=tokenizer.vocab_size,
            embed_dim=embed_dim,
            max_sequence_length=max_sequence_length,
        )
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim, head_dim=head_dim, num_heads=num_heads
        )

        self.mlp = FFMLP(embed_dim=embed_dim)
        self.output = nn.Linear(embed_dim, tokenizer.vocab_size)
        self.linear_norm1 = nn.LayerNorm(embed_dim)
        self.linear_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, input: torch.Tensor):
        x = self.embedding(input)
        o = x + self.attention(self.linear_norm1(x))
        o = o + self.mlp(self.linear_norm2(o))
        return self.output(o)

    @torch.no_grad()
    def generate(self, prompt: str, length: int):
        idx = encode_data(prompt, self.tokenizer)
        #  call the model to get logits
        # hardcode batch of 1 ie. no batching for generation
        generated_tokens = []
        for _ in range(length):
            # only feed in the last part - limitation of current learned position embeddings
            idx_context = idx[-self.max_sequence_length :]
            logits = self(idx_context.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)  # B,T,V
            probs_final = probs[:, -1, :]
            next_token = torch.multinomial(probs_final[0], num_samples=1)
            generated_tokens.append(next_token.item())
            idx = torch.cat((idx, next_token))

        return generated_tokens


def train(
    model: Transformer,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    batch_size: int,
    max_iters: int,
    optimizer: torch.optim.Optimizer,
):
    recent_losses = []
    train_losses = []
    val_losses = []
    for iter in range(max_iters):
        model.train()
        idx, targets = get_batch(
            train_data, batch_size=batch_size, sequence_length=sequence_length
        )
        logits = model(idx)
        logits = logits.flatten(0, 1)
        targets = targets.flatten(0, 1)

        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        recent_losses.append(loss.item())
        if (iter + 1) % 100 == 0:
            model.eval()
            losses = recent_losses[-100:]
            avg_train_loss = sum(losses) / len(losses)
            avg_val_loss = get_avg_val_loss(
                validation_data=validation_data, iters=5, model=model
            )
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            print(
                f"iter {iter}: "
                f"loss_mean={avg_train_loss:.4f}, "
                f"loss_max={max(losses):.4f}, "
            )
            model.train()

    avg_val_loss = get_avg_val_loss(validation_data, iters=100, model=model)
    print(f"{avg_val_loss=}")
    plot_loss(train_losses=train_losses, val_losses=val_losses)


if __name__ == "__main__":
    # get train splits
    text = get_corpus_text()
    tokenizer = CharTokenizer(text)
    data = encode_data(text, tokenizer=tokenizer)
    splits = get_splits(data)
    train_data = splits["train"]
    # call train
    embed_dim = 32
    head_dim = 128
    num_heads = 4
    batch_size = 32
    sequence_length = 64
    lr = 0.001
    max_iters = 20_000

    model = Transformer(
        embed_dim=embed_dim,
        head_dim=head_dim,
        num_heads=4,
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
        validation_data=splits["validation"],
    )
    prompt = "HAMLET:\nTo be or not to be"
    generated_tokens = model.generate(prompt, 1000)
    print(prompt)
    print(tokenizer.decode(generated_tokens))
