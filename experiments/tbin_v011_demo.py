"""Reproducible synthetic smoke test for the packaged TBIN prototype."""

import random, math
from tbin import BraidEncoder, OpenWorldTBIN

rng = random.Random(1111)
encoder = BraidEncoder()

def world(latent, noise=2):
    return [
        max(0, min(255, 128 + 44*latent[t % len(latent)]
                   + rng.gauss(0, noise)))
        for t in range(30)
    ]

known_latent = [rng.uniform(-1, 1) for _ in range(8)]
novel_latent = [rng.uniform(-1, 1) for _ in range(8)]

known_samples = [encoder.encode(world(known_latent)) for _ in range(30)]
# Deterministic representative adequate for smoke testing.
known_proto = min(known_samples, key=lambda a: sum(
    sum(x != y for x, y in zip(a, b)) + abs(len(a)-len(b))
    for b in known_samples
))

model = OpenWorldTBIN({"KNOWN": known_proto}, encoder=encoder)
for _ in range(30):
    result = model.ingest(world(novel_latent))
    # Separate encounters temporally with known observations.
    model.ingest(world(known_latent))

print("crystals:", sorted(model.crystals))
print("last novel event:", result)
print("NOTE: synthetic smoke test; not a real-language benchmark.")
