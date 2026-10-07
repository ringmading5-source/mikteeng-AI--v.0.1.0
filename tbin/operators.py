"""Few-shot structural operator induction for TBIN experiments.

Learns sparse signed coordinate maps y_i ~= sign*x_j + bias directly from pairs.
The hypothesis family is explicit; this is not unrestricted rule discovery.
"""
from statistics import median

def discover_signed_coordinate_operator(pairs):
    if not pairs:
        raise ValueError("at least one pair is required")
    d = len(pairs[0][0])
    mapping = []
    for oi in range(d):
        best = None
        for j in range(d):
            for sign in (-1, 1):
                residuals = [b[oi] - sign * a[j] for a, b in pairs]
                bias = int(round(median(residuals)))
                err = sum(abs(sign * a[j] + bias - b[oi]) for a, b in pairs) / len(pairs)
                item = (err, j, sign, bias)
                if best is None or item < best:
                    best = item
        _, j, sign, bias = best
        mapping.append((j, sign, bias))
    return mapping

def apply_signed_coordinate_operator(x, mapping, lo=-12, hi=12):
    return tuple(max(lo, min(hi, sign * x[j] + bias))
                 for j, sign, bias in mapping)
