"""Core braid-inspired representation primitives for TBIN."""

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

BraidWord = Tuple[int, ...]
Transition = Tuple[int, int, int]

@dataclass(frozen=True)
class BraidEncoder:
    window: int = 3
    basin: int = 18
    strands: int = 11

    def encode(self, values: Sequence[float]) -> BraidWord:
        if self.window < 1 or self.basin < 1 or self.strands < 2:
            raise ValueError("window>=1, basin>=1 and strands>=2 are required")
        pooled = [
            sum(values[i:i+self.window]) / len(values[i:i+self.window])
            for i in range(0, len(values), self.window)
            if values[i:i+self.window]
        ]
        q = [round(v / self.basin) for v in pooled]
        word = []
        for i, value in enumerate(q):
            generator = 1 + ((abs(value) * 5 + i * 2) % (self.strands - 1))
            sign = 1 if i == 0 or value >= q[i-1] else -1
            g = sign * generator
            # Exact local inverse cancellation only.
            if word and word[-1] == -g:
                word.pop()
            else:
                word.append(g)
        return tuple(word)

def word_distance(a: Sequence[int], b: Sequence[int]) -> float:
    length = max(len(a), len(b), 1)
    overlap = min(len(a), len(b))
    mismatches = sum(a[i] != b[i] for i in range(overlap))
    return (mismatches + abs(len(a)-len(b))) / length

def transition_signature(word: Sequence[int]) -> Tuple[Transition, ...]:
    return tuple(
        (1 if b > 0 else -1, abs(b)-abs(a), int((a > 0) == (b > 0)))
        for a, b in zip(word, word[1:])
    )

def transition_distance(a: Sequence[Transition], b: Sequence[Transition]) -> float:
    length = max(len(a), len(b), 1)
    overlap = min(len(a), len(b))
    mismatches = sum(a[i] != b[i] for i in range(overlap))
    return (mismatches + abs(len(a)-len(b))) / length

def medoid(items: Sequence, metric):
    if not items:
        raise ValueError("medoid requires at least one item")
    return min(items, key=lambda x: sum(metric(x, y) for y in items))
