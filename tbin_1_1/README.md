# TBIN 1.1 — repaired experimental CPU learner

This updates the inspected TBIN 1.0 package. It is a trainable multimodal retrieval and restricted relational-learning prototype, not a general-purpose conversational model. No braid invariants, speech transcription, speech generation, or open-ended language generation are implemented.

## Install and run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python evaluate_repairs.py
python tbin.py train pairs.jsonl --model my_model.pkl --epochs 80 --batch-size 64
python tbin.py retrieve my_model.pkl text cow image cow.png goat.png tree.png
```

Use Python 3.10 or later. The implementation uses NumPy, Pillow and SciPy on CPU. No GPU is required. The evaluation writes `TBIN_1_1_test_report.json` in the current directory.

## Paired data

Use at least three JSONL rows, with the same two or three modalities in every row:

```jsonl
{"image":"cow1.png","audio":"cow1.wav","text":"cow","group":"cow"}
{"image":"goat1.png","audio":"goat1.wav","text":"goat","group":"goat"}
{"image":"tree1.png","audio":"tree1.wav","text":"tree","group":"tree"}
```

Paths are relative to the manifest. `group` is optional; supply it on every row or none. Equal group IDs designate multiple positive observations instead of false negatives. Interleave different groups throughout the file: a batch with only one group is rejected. If groups are omitted, each row is treated as a distinct positive pair. Group labels and pairings are supervision.

Training reopens and streams the manifest for each pass. It does not load all observations or build a dataset-wide similarity matrix. Batch size must be at least six; datasets with three to five observations use one smaller batch. Tail observations are retained. The raw buffer contains at most batch size + 3 rows; comparisons contain at most batch size squared elements. Training time still grows with examples and epochs. A validation pass checks all data first; training failures after updates do not roll back parameters.

Feature extraction is repeated per epoch to keep RAM bounded; this trades throughput for memory. This is not a million-example throughput benchmark. The older `tbin_v26.py` evaluation utility accepts a materialized dataset; use `tbin.py train` for streaming training.

## What changed

- **Consolidation:** prunes transition counts while preserving Counter objects, allowing further learning.
- **Audio:** reads the first three seconds of 16-bit PCM WAV and resamples to a common 8 kHz with anti-alias filtering. Duration is preserved, rather than compressing every clip to 2,048 samples. Very short clips are zero-padded to 64 samples. Long-audio observation remains separate from this training window.
- **Relations:** three-token observations bootstrap predicates. Longer sentences containing exactly one known single-token predicate can supply multiword subject/object spans. Optional English article stripping is a declared heuristic, not universal learned grammar. New/ambiguous predicates need explicit facts or demonstrations; multiword predicates can be taught through those APIs.
- **Image features:** retrieval combines 48 spatial colour means with a 64-bin histogram over learned patch prototypes. A maximum of the first 256 training images fits the retrieval codebook once. It is then frozen to keep embedding features stable. Ordinary image observation uses a separate, continuing codebook. Codebooks are included in checkpoints.
- **Neural relation ranking:** demonstrations train the ranking head when stored negative candidates are available. Untrained random scores no longer influence answers. Its benefit over graph-only answering remains unproven, and incremental training can forget earlier examples.
- **Tests:** exercise accuracy, streaming tails, audio timing and anti-aliasing, consolidation, relation learning, checkpoint consistency, and invalid inputs.

## Relation example

```python
from tbin import TBIN
m = TBIN()
m.teach_relation('cow likes grass', 'what does cow like', 'cow', 'grass')
m.observe_text('the sheep likes green grass')
assert m.answer_relation('what does sheep like', 'sheep') == 'green grass'
m.core.reasoner.consolidate()
m.observe_text('goat likes leaves')  # continues learning after consolidation
m.observe_fact('young goat', 'eats', 'fresh leaves')
m.teach_relation('young goat eats fresh leaves',
                 'what does young goat eat', 'young goat', 'fresh leaves')
m.save('my_model.pkl')
```

Paired training no longer automatically accumulates text/graph memory. Use `--learn-text` or `learn_text=True` to opt in, or teach text/facts separately. Explicit text statistics and graph knowledge grow with unique content; only the contrastive training buffers are bounded. The recent-passage list retains 256 entries, but transition maps are not globally capped.

## Evaluation and compatibility

`TBIN_1_1_test_report.json` records actual local results. The retrieval benchmark uses flat colour images and sine tones, with unseen variations of six familiar synthetic concepts. Three seeds and a deliberately wrong pairing control are included. This does not demonstrate real-world semantics or unseen-concept generalization.

For real data, split by speaker/object identity and recording session before training, and keep validation separate from test. Record untrained and shuffled-pair baselines. Do not tune on the final test set. No real paired held-out dataset was available for this repair run.

Retrieval returns a gallery index and cosine scores, not a generated answer or calibrated confidence. It always selects an item from a nonempty gallery; unknown-input rejection remains future work.

Version 1.0 checkpoints are not compatible with the expanded image projection/codebook: retrain from paired data. New projection-only NPZ files include the frozen codebook; full model checkpoints use pickle and must only be loaded from trusted local sources. The archive preserves the legacy module names for imports.
