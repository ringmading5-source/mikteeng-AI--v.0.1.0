"""Hierarchical TBIN research primitives.

Experimental: observation features -> stable invariant -> bounded residual variants.
These routines do not establish semantic understanding.
"""
from collections import Counter

def jaccard_distance(a, b):
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / max(1, len(a | b))

def stable_invariant(feature_sets, support=0.45):
    if not feature_sets:
        return set()
    counts = Counter(x for fs in feature_sets for x in fs)
    n = len(feature_sets)
    return {x for x, c in counts.items() if c / n >= support}

def bounded_residual_bank(feature_sets, invariant, limit=3, novelty=0.55):
    bank = []
    for view in feature_sets:
        if len(bank) < limit:
            bank.append(view)
            continue
        nearest = min(jaccard_distance(view, r) for r in bank)
        if nearest > novelty:
            weakest = min(range(len(bank)),
                          key=lambda i: jaccard_distance(bank[i], invariant))
            bank[weakest] = view
    return bank
