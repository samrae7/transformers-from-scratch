from pathlib import Path
import numpy as np

file = Path("input.txt")
text = file.read_text()
print(f"length: {len(text)}")
print(f"first 100 chars:{text[:100]}")

chars = sorted(set(text))
vocab_size = np.unique(chars).size
print(f"{vocab_size=}")

stoi = {char: i for i, char in enumerate(chars)}
print(f"{stoi=}")

itos = {i: char for char, i in stoi.items()}
print(f"{itos=}")


def encode(text: str) -> list[int]:
    return [stoi[ch] for ch in text]


def decode(ids: list[int]) -> str:
    return "".join([itos[i] for i in ids])
