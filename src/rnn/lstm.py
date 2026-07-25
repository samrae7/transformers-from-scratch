from torch import nn
import torch
import torch.nn.functional as F

from language_modeling.data import encode_data, get_corpus_text, get_splits
from language_modeling.tokenizer import CharTokenizer
from rnn.data import get_batch


class LSTMCell(nn.Module):
    def __init__(self, embed_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.w_fx = nn.Linear(embed_dim, hidden_dim)
        self.w_fh = nn.Linear(hidden_dim, hidden_dim)
        self.w_ix = nn.Linear(embed_dim, hidden_dim)
        self.w_ih = nn.Linear(hidden_dim, hidden_dim)
        self.w_gx = nn.Linear(embed_dim, hidden_dim)
        self.w_gh = nn.Linear(hidden_dim, hidden_dim)
        self.w_ox = nn.Linear(embed_dim, hidden_dim)
        self.w_oh = nn.Linear(hidden_dim, hidden_dim)

    def _compute_gates(self, x_t, h_prev):
        f_t = torch.sigmoid(self.w_fx(x_t) + self.w_fh(h_prev))
        i_t = torch.sigmoid(self.w_ix(x_t) + self.w_ih(h_prev))
        g_t = torch.tanh(self.w_gx(x_t) + self.w_gh(h_prev))
        o_t = torch.sigmoid(self.w_ox(x_t) + self.w_oh(h_prev))
        return f_t, i_t, g_t, o_t

    def forward(self, x_t, h_prev, c_prev):
        f_t, i_t, g_t, o_t = self._compute_gates(x_t, h_prev)
        c_t = f_t * c_prev + i_t * g_t
        h_t = torch.tanh(c_t) * o_t
        return h_t, c_t


class LSTM(nn.Module):
    def __init__(
        self, cell: LSTMCell, tokenizer: CharTokenizer, embed_dim: int, hidden_dim: int
    ):
        super().__init__()
        self.cell = cell
        self.hidden_dim = hidden_dim
        self.tokenizer = tokenizer
        self.embeddings = nn.Embedding(
            num_embeddings=tokenizer.vocab_size, embedding_dim=embed_dim
        )
        self.hidden_to_logits = nn.Linear(hidden_dim, tokenizer.vocab_size)

    def forward(
        self,
        inputs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        #  inputs shape is B, T
        # cell takes B, E
        batch_size, sequence_length = inputs.shape
        embs = self.embeddings(inputs)
        # embs.new_zeros -> h_t,c_t have same device and d_type as embs
        if state is None:
            h_t = embs.new_zeros([batch_size, self.hidden_dim])
            c_t = embs.new_zeros([batch_size, self.hidden_dim])
        else:
            h_t, c_t = state
        hidden_states = []
        for t in range(sequence_length):
            emb = embs[:, t, :]
            h_t, c_t = self.cell(emb, h_t, c_t)
            hidden_states.append(h_t)
        #  hidden_to_logits takes T hidden states, which individually
        #  are B,H and turns them into logits which are B,T,V
        #  so we want a stack of hidden_states that are B,T,H
        hidden_states_stack = torch.stack(hidden_states, dim=1)
        logits = self.hidden_to_logits(hidden_states_stack)
        return logits, (h_t, c_t)

    @torch.no_grad()
    def generate(self, prompt: str, length: int, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")
        # get logits for prompt
        encoded = encode_data(prompt, tokenizer=self.tokenizer)
        inputs = encoded.unsqueeze(0)  # 1,T - no batching but need to match
        # expected shape of model which is B,T
        self.eval()
        logits, state = self(inputs)  # logits: 1,T,V
        generated_tokens = []
        for _ in range(length):
            final_logits = logits[:, -1, :]
            probs = torch.softmax(final_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs[0], num_samples=1)
            generated_tokens.append(next_token.item())

            next_token_input = next_token.unsqueeze(0)
            logits, state = self(next_token_input, state)
        # convert list of tokens to str of characters

        return self.tokenizer.decode(generated_tokens)


def train(
    model: LSTM,
    batch_size: int,
    max_iters: int,
    train_data: torch.Tensor,
    sequence_length: int,
    optimizer: torch.optim.Optimizer,
):
    model.train()
    recent_losses = []
    # recent_grad_norms = []
    for iter in range(max_iters):
        x, y = get_batch(
            split=train_data, batch_size=batch_size, sequence_length=sequence_length
        )
        optimizer.zero_grad()
        logits, _ = model(x)
        #  calc cross entropy
        #  logits have shape B,T,V
        #  cross_entropy take N,C, which means we need to reshape to B*T, V
        #  shape of targets is B,T, C.E take shape N so we reshape to B * T
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(0, 1))

        loss.backward()
        # grad_norm = model.get_total_gradient_norm()
        # recent_grad_norms.append(grad_norm)
        optimizer.step()

        recent_losses.append(loss.item())
        if iter % 100 == 0:
            losses = recent_losses[-100:]
            # grad_norms = recent_grad_norms[-100:]

            print(
                f"iter {iter}: "
                f"loss_mean={sum(losses) / len(losses):.4f}, "
                f"loss_max={max(losses):.4f}, "
                # f"grad_norm_mean={sum(grad_norms) / len(grad_norms):.4f}, "
                # f"grad_norm_max={max(grad_norms):.4f}"
            )


if __name__ == "__main__":
    text = get_corpus_text()
    tokenizer = CharTokenizer(text)
    data = encode_data(text, tokenizer=tokenizer)
    splits = get_splits(data)
    train_data = splits["train"]
    validation_data = splits["validation"]

    sequence_length = 32
    lr = 0.001
    embed_dim = 32
    hidden_dim = 128
    cell = LSTMCell(embed_dim=embed_dim, hidden_dim=hidden_dim)

    model = LSTM(
        tokenizer=tokenizer, embed_dim=embed_dim, hidden_dim=hidden_dim, cell=cell
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(
        model=model,
        train_data=train_data,
        batch_size=32,
        sequence_length=sequence_length,
        max_iters=20000,
        optimizer=optimizer,
        # validation_data=validation_data,
    )
    generated = model.generate(
        "JULIET:\nWherefore art thou Romeo?",
        length=1000,
        temperature=0.8,
    )
    print(f"generated:\n{generated}")
