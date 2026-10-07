"""Transformation-stable identity extraction from TBIN v0.11."""

import math
from collections import Counter

def relational_features(descriptors):
    """Extract relative local structure, suppressing absolute amplitude."""
    features = []
    for j in range(1, len(descriptors)):
        _, idx, sign, delta = descriptors[j]
        _, pidx, psign, pdelta = descriptors[j-1]
        features.extend([
            ("dir", j, 1 if delta > 0 else (-1 if delta < 0 else 0)),
            ("turn", j, int(sign != psign)),
            ("idxord", j, 1 if idx > pidx else (-1 if idx < pidx else 0)),
            ("curv", j, 1 if delta-pdelta > 0 else (-1 if delta-pdelta < 0 else 0)),
        ])
    return set(features)

def stable_identity(view_feature_sets, support=0.75):
    if not view_feature_sets:
        return frozenset()
    counts = Counter(x for fs in view_feature_sets for x in fs)
    required = math.ceil(len(view_feature_sets) * support)
    return frozenset(k for k, count in counts.items() if count >= required)

def jaccard_distance(a, b):
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)

def descriptor_encode(values, window=3, basin=18, strands=13):
    """Position-aware descriptors used for invariant experiments."""
    pooled = [
        sum(values[i:i+window])/len(values[i:i+window])
        for i in range(0, len(values), window) if values[i:i+window]
    ]
    q = [round(v/basin) for v in pooled]
    result, prev = [], None
    for i, value in enumerate(q):
        idx = 1 + ((abs(value)*5+i*2) % (strands-1))
        sign = 1 if prev is None or value >= prev else -1
        result.append((i, idx, sign, 0 if prev is None else value-prev))
        prev = value
    return tuple(result)
