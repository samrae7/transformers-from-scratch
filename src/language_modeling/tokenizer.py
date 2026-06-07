class CharTokenizer:
    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {char: i for i, char in enumerate(self.chars)}
        self.itos = {i: char for char, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)
