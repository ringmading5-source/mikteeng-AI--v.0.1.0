# Mikteeng: bounded evidence-driven prediction

Core remains Observation -> first predictions -> prediction-history Self -> learned response -> feedback observations. The original regression experts and routing remain; no pretrained model, Transformer, LLM API or GPU requirement was added. The upward hierarchy stays optional because its earlier score regressed. No speech comprehension capability is claimed.

Implemented concepts:
- Bounded temporal state: original rollouts retain only the history needed by the configured prediction windows, with 1..64 steps per request.
- Evidence separate from activation: routing exposes observation counts separately from cosine alignment. Neither is calibrated confidence.
- Identity-preserving updates: repeated assigned patterns reuse node identities and retain unit anchors; exact child groups determine byte-node identity.
- Sparse allocation: vector storage grows in blocks, with configured storage capped at 16,777,216 float64 entries. Zero nodes allocate zero anchor bytes.
- Protected mutation: node and hierarchy updates use locks. Inference should use frozen model snapshots; concurrent model retraining is not supported.
- Lossless transport separate from routing: exact byte payloads remain the recovery source. Compact similarity features may collide.
- Safe node storage: save_nodes/load_nodes uses validated NPZ arrays and JSON metadata with allow_pickle=False. This stores node registries, not the full expert predictor.

Safety boundaries:
Packets are limited to 16 MiB, metadata to 4096 characters; array dtype/shape/byte consistency is checked. Node IDs are at most 256 characters. Image dimensions are checked before conversion. Full WAV reads have a file cap; iter_wav yields bounded PCM chunks (8/16/32-bit, at most eight channels). Streaming chunk boundaries currently reset byte grouping; continuous cross-chunk pattern state is not implemented.

Legacy full-model files remain pickle-based. MikteengAI.load now rejects them by default. Only explicitly verified files should use load(path, trusted=True) or load_trusted(path). These calls still execute pickle and are not safe for untrusted uploads. Model archives have compressed and expanded size limits. Shipped local examples/test fixtures now use explicit trusted loading. This is an intentional API change. Runtime dependency pinning remains important.

The local chatbot retains localhost binding and gets a five-second socket timeout. No public authentication or rate limiting has been added; it should not be exposed directly as a public service.

Validation: the original held-out result remained 156/192. Checks passed for step caps, explicit trust rejection, 100 concurrent observations on one identity, lazy allocation, unsafe/inconsistent metadata rejection, exact PCM chunk reconstruction, model reload, numeric-only node reload, eight tokenizer tests and four legacy recursive tests. Power usage has not been measured. The system is CPU-only in these tests.

Use from package folder:
`python -m pip install .`
`PYTHONPATH=src python tests/verify_hardening.py`

Example safe node checkpoint:
```python
space.save_nodes('pattern_nodes.npz')
space = PatternNodeSpace.load_nodes('pattern_nodes.npz')
```
