# Mikteeng latest learning experiments

Recovered and combined on October 5, 2026 from the predictive composition, word formation, and acoustic training curve packages. This directory contains the full extended MikteengAI implementation, reproducible experiments, reports, and trusted local checkpoints. Existing repository code and deployment packages are preserved. The updated chatbot deployment vendors this implementation and builds an integrated checkpoint; see [deployment instructions](../../deploy/mikteeng-chatbot/README.md).

## Run

From the repository root:

```bash
cd research/latest-learning
python -m pip install .
PYTHONPATH=src python examples/test_compositional_generation.py
PYTHONPATH=src python examples/test_word_formation.py
PYTHONPATH=src python examples/train_acoustic_curve.py
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python tests/verify_hardening.py
```

The experiment scripts write reports into experiments/ and checkpoints into models/. Model files contain pickle: load only trusted files, using MikteengAI.load_trusted(path) or load(path, trusted=True). This candidate has the same package name as the root library; use a separate virtual environment if installing both.

## Findings

- [Predictive composition](COMPOSITIONAL_GENERATION.md): 56/56 structured start/goal cases, including 32 multi-step cases; familiar primitive transitions in a small domain.
- [Word formation](WORD_FORMATION_FINDINGS.md): 3/8 guided held-out suffix completions with local context; autonomous novel-word generation was not demonstrated.
- [Acoustic learning](ACOUSTIC_TRAINING_CURVE.md): synthetic waveform continuation improves with expanded coverage, with diminishing gains from more examples of the same patterns. Real speech and text-to-speech remain untested.
- [Speech interface](SPEECH_INPUT.md), [checkpoint and bounds controls](MECHANISM_AND_SAFETY.md).

## Verification on recovery

Freshly reran five root API tests, five candidate speech tests, hardening checks (156/192 retained sentence continuations, checkpoint trust, bounded prediction, concurrent node reuse, PCM streaming, save/reload), composition, word formation, and the four-stage acoustic curve. Detailed generated reports are included; timing varies by machine. These results do not establish universal understanding or real speech generation.
