"""TBIN v0.20 adversarial crystallization benchmark.

Synthetic benchmark. It compares recurrence-only crystallization with an
evidence-gated criterion under deliberately recurrent false patterns.

Required evidence:
- recurrence
- cross-context support
- compression gain
- predictive consistency

Do not interpret this benchmark as real-language performance.
"""
# This file documents the benchmark protocol used during v0.20 research.
CRITERIA = {
    "recurrence": True,
    "cross_context_stability": True,
    "compression_gain": True,
    "predictive_consistency": True,
}
EXPECTED_FALSIFICATION = (
    "recurrence-only should accept deliberately recurrent false patterns; "
    "evidence-gated crystallization should reject patterns whose consequences "
    "are inconsistent."
)

if __name__ == "__main__":
    print("TBIN v0.20 protocol:", CRITERIA)
    print(EXPECTED_FALSIFICATION)
