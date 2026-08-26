from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

PAD = "<pad>"
UNK = "<unk>"
CLS = "<cls>"


class Vocabulary:
    def __init__(self, max_size: int = 30000, min_freq: int = 1):
        self.max_size = max_size
        self.min_freq = min_freq
        self.stoi = {PAD: 0, UNK: 1, CLS: 2}
        self.itos = [PAD, UNK, CLS]

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_']+|[^\w\s]", str(text).lower())

    def build(self, texts: Iterable[str]) -> None:
        counts = Counter()
        for text in texts:
            counts.update(self.tokenize(text))
        for tok, freq in counts.most_common(self.max_size - len(self.itos)):
            if freq < self.min_freq:
                break
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, text: str, max_length: int) -> tuple[list[int], list[int]]:
        toks = [CLS] + self.tokenize(text)
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in toks[:max_length]]
        mask = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.stoi[PAD])
            mask.append(0)
        return ids, mask

    def __len__(self) -> int:
        return len(self.itos)
