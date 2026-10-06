# Mikteeng RSPM

Mikteeng RSPM is the primary model in this repository. It replaces the earlier character, role, sentence, self-model and waveform learners in the active package and deployment. Git history preserves the old release; `research/` and `validation/` are historical experiments, not active model code.

The model learns context-local operators over a fixed orthonormal basis and retains bounded, per-trigger outcome hypotheses. It runs on CPU with NumPy. No LLM, RNN or adaptive global basis is involved.

## Install

```bash
python -m pip install .
```

```python
from mikteeng_rspm import MikteengRSPM, save, load
model = MikteengRSPM(dim=64)
result = model.train(history_vector, trigger_vector, target_vector)
answer = model.query(history_vector, trigger_vector)
save(model, 'trained.json')
model = load('trained.json')
```

Each vector must have exactly `dim` finite values and a nonzero norm. History is projected through the fixed positive basis selection. Trigger and target vectors are normalized. Dataset representation is supplied by the caller; this release does not automatically interpret raw text, audio or images.

## Stream larger datasets

Use JSONL with one observation per line:

```json
{"history": [0.2, 0.8], "trigger": [1.0, 0.0], "target": [0.0, 1.0]}
```

The example above has dimension two. All rows in a training job must use the configured dimension.

```bash
python -m mikteeng_rspm train dataset.jsonl --dim 64 --output trained.json \
  --max-contexts 24 --max-triggers 32 --max-modes 8 --max-candidates 16 \
  --ttl 1000 --checkpoint-every 10000
python -m mikteeng_rspm train next_dataset.jsonl --resume trained.json --output trained.json
```

Rows are processed incrementally; the complete dataset is not loaded into RAM. Checkpoints are written atomically and can resume training. Resume uses the checkpoint's dimensions, capacities and hyperparameters. CLI arguments for new-model construction do not override a resumed model. Allocation rejections and ambiguous routes are included in training summaries. Do not infer predictive accuracy from observation counts.

`python examples/train.py` creates a small dataset using the model's fixed basis. Large training is supported as sequential CPU ingestion with bounded stores, not as a validated multi-GPU or distributed trainer. Dense local operators require approximately 8 × contexts × dim² bytes before other records. Checkpoint loading currently has a 256 MiB file limit.

## Serve inference

```bash
python -m mikteeng_rspm serve --checkpoint trained.json --host 0.0.0.0 --port 8000
```

Open `/` for the vector console. `GET /health` identifies the model and dimension. `POST /api/predict` accepts history and trigger arrays and returns routing status, learned vector prediction and copied outcome hypotheses with evidence frequencies. These frequencies are not calibrated confidence scores. The HTTP service exposes no training endpoint and never advances the training clock during inference.

The committed `models/synthetic_vector_demo.json` checkpoint has dimension 32 and was trained on two synthetic coordinate rules. `/api/example` supplies one compatible query. It is a vector demonstration, not a curriculum-trained chatbot. Old `/api/chat`, text completion and audio endpoints have been retired.

## Model guarantees tested

- Fixed basis B; independent local R per history context.
- Context and trigger similarity/margin gates; explicit unmatched and ambiguous statuses.
- Independent confirmed outcome modes and provisional candidates.
- Bounded context, trigger, mode and candidate allocations; confirmed memory protection and explicit capacity rejection.
- Global training-step TTL maintenance; no inference side effects.
- Complete byte-aware snapshots, copied query outputs and atomic checkpoint writes.
- Conservative staged merges using exact one-to-one trigger/mode correspondence. Approximate domain union remains unsupported.

## Tests and measured scope

```bash
python -m unittest discover -s tests -v
python benchmarks/benchmark_bounded.py
python benchmarks/train_generalization.py
```

The core audit passed ten tests; integration tests cover checkpoint/resume and live HTTP inference. The synthetic novelty benchmark allocated 1,000 novel contexts through provisional recycling, retained two established rules and retrieved 100/100 noisy queries. Five training runs of 960 observations learned unseen positive combinations of familiar basis symbols with diagnostic error around 10⁻¹⁵.

**Important:** broad combinations still fail the public trigger gate. Direct operator diagnostics show learned prediction quality but do not establish end-to-end success on rejected queries. Nearby combinations pass public routing. The package does not claim arbitrary unseen-rule learning, language understanding, conversational generation, speech generation or GPU throughput. See `benchmarks/GENERALIZATION_TRAINING.md`.

## Deployment and migration

The Render blueprint starts this package from the repository root. Existing services using the old deployment root can use its thin compatibility launcher; new services use the repository root; see `DEPLOYMENT.md`. Old `.mkteeng` checkpoints are incompatible and are not silently converted. Convert source training records into vector JSONL and retrain RSPM. Import compatibility exposes `MikteengRSPM` from `mikteeng_ai`; the old `MikteengAI` multi-learner API is retired.

## Experimental WAV voice training

The optional [voice prototype](experiments/rspm_voice/README.md) supports WAV/transcript training and voice-only acoustic pattern discovery, using the Unicode text adapter. It is separate from the deployed vector service.

```bash
python -m pip install ./experiments/rspm_voice
python experiments/rspm_voice/demo.py
python experiments/rspm_voice/test_voice.py
mikteeng-voice train --manifest my_recordings/manifest.jsonl --out voice_run --mode paired --epochs 20
```

Paired mode selects among known complete transcripts; it is not free-form ASR. Voice-only mode discovers acoustic patterns without word labels. Generated-tone validation passed 9/9 held-out clips and eight tests; no real Dinka speech has been trained or evaluated. A stronger sequence encoder remains future work.

