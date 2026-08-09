import torch
import torch.nn.functional as F
from torch import nn

from language_modeling.data import encode_data, get_corpus_text, get_splits
from language_modeling.tokenizer import CharTokenizer
from rnn.data import get_batch


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

    def forward(
        self, input_tokens: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = input_tokens.device
        batch_size, sequence_length = input_tokens.shape
        if hidden is None:
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
        return logits, hidden

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        length: int,
        tokenizer: CharTokenizer,
        padding_char="\n",
        temperature: float = 1.0,
    ):
        """Generate one continuation from one prompt (no batching)."""
        self.eval()
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        encoded = encode_data(prompt, tokenizer)
        # get final token
        logits, hidden = self(encoded.unsqueeze(0))

        generated_tokens = []
        for _ in range(length):
            next_token = self._sample_next_token(logits, temperature=temperature)
            generated_tokens.append(next_token.item())

            next_token_input = next_token.unsqueeze(0)
            logits, hidden = self(next_token_input, hidden)

        generated = tokenizer.decode(generated_tokens)
        return generated

    def _sample_next_token(
        self, logits: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        last_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(last_logits, dim=-1)
        return torch.multinomial(probs[0], num_samples=1)

    def _init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros((batch_size, self.hidden_dim), device=device)

    def get_total_gradient_norm(self):
        norm = 0.0
        for p in self.parameters():
            if p.grad is None:
                continue
            param_grad_norm = p.grad.norm(2)
            # Each tensor norm already includes a square root, so square it
            # before combining all parameter gradients into one total L2 norm.
            norm += param_grad_norm.item() ** 2
        return norm**0.5


def train(
    model: RNN,
    train_data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    sequence_length: int,
    max_iters: int,
    validation_data: torch.Tensor,
) -> None:
    model.train()

    recent_losses = []
    recent_grad_norms = []

    for iter in range(max_iters):
        idx, targets = get_batch(
            split=train_data, batch_size=batch_size, sequence_length=sequence_length
        )
        logits, _ = model(idx)
        logits = logits.flatten(0, 1)
        targets = targets.flatten(0, 1)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = model.get_total_gradient_norm()
        recent_grad_norms.append(grad_norm)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        recent_losses.append(loss.item())
        if iter % 100 == 0:
            losses = recent_losses[-100:]
            grad_norms = recent_grad_norms[-100:]

            print(
                f"iter {iter}: "
                f"loss_mean={sum(losses) / len(losses):.4f}, "
                f"loss_max={max(losses):.4f}, "
                f"grad_norm_mean={sum(grad_norms) / len(grad_norms):.4f}, "
                f"grad_norm_max={max(grad_norms):.4f}"
            )
    val_loss_accum = 0
    val_steps = 100
    for iter in range(val_steps):
        idx, targets = get_batch(
            split=validation_data,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        val_logits, _ = model(idx)
        val_logits = val_logits.flatten(0, 1)
        targets = targets.flatten(0, 1)
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
    validation_data = splits["validation"]

    sequence_length = 32
    lr = 0.001

    model = RNN(
        tokenizer=tokenizer,
        embed_dim=32,
        hidden_dim=128,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(
        model=model,
        train_data=train_data,
        batch_size=32,
        sequence_length=sequence_length,
        max_iters=200,
        optimizer=optimizer,
        validation_data=validation_data,
    )
    generated = model.generate(
        "Wherefore art thou Romeo?",
        length=1000,
        tokenizer=tokenizer,
        temperature=0.8,
    )
    print(f"generated:\n{generated}")
