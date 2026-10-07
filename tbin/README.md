# TBIN experimental prototype

TBIN is a braid-inspired, non-backpropagation research prototype extracted from the v0.1-v0.11 synthetic experiments.

Current implemented layers:
- temporal pooling and stable quantization;
- signed braid-like discrete words with exact local inverse cancellation;
- structural and transition distances;
- open-world provisional states;
- predictive-transition-gated crystallization;
- transformation-stable relational features and identity signatures.

## Status

This is **experimental research code**. The reported conversation experiments used synthetic observations. The package does not yet establish semantic understanding, real-speech robustness, state-of-the-art retrieval, Jones-polynomial semantic equivalence, or O(1) end-to-end inference.

## Quick smoke test

Run from the repository root:

    python experiments/tbin_v011_demo.py

## Next falsification step

Use small real speech recordings with repeated concepts, multiple speakers, noise conditions, and held-out speakers. Measure same-concept/cross-speaker distance against different-concept distance before expanding the architecture.
