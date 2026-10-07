"""TBIN experimental prototype.

Non-neural, braid-inspired structural representation experiments.
The package is research code; current results are synthetic, not real-language benchmarks.
"""
from .core import BraidEncoder, word_distance, transition_signature, transition_distance
from .open_world import OpenWorldTBIN
from .invariants import relational_features, stable_identity, jaccard_distance

__all__ = [
    "BraidEncoder", "word_distance", "transition_signature", "transition_distance",
    "OpenWorldTBIN", "relational_features", "stable_identity", "jaccard_distance",
]
__version__ = "0.11.0"
