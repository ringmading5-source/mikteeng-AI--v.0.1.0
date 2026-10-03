# mikteeng AI

An installable Python library for the experimental character, word-role, sentence-state and passage predictors developed in this project. Version 0.1.0 is a research prototype, not a universal AI or a production framework.

## Install

Clone this repository and open its folder, or extract the downloadable archive:

```bash
git clone https://github.com/ringmading5-source/mikteeng-AI--v.0.1.0.git
cd mikteeng-AI--v.0.1.0
```

Then install:

```bash
python -m pip install .
```

Python 3.10 or newer is required. Dependencies: NumPy, SciPy and scikit-learn. A built wheel is included under `dist/`. The project is not published on PyPI: `pip install mikteeng-ai` by name is not the installation route for this release.

## Generate

```python
from mikteeng_ai import MikteengAI

ai = MikteengAI.load("models/biology_physics.mkteeng")
result = ai.generate("Explain cells and velocity.")
print(result["text"])
```

The bundled model generates next words from weights. It does not retrieve stored answer sentences. Its scientific vocabulary and composition tests are narrow and familiar; memorized sentence wording remains a limitation. Model files use pickle internally: load trusted files only.

## Revise word roles

```python
session = ai.role_session()
for word in ["the", "cat", "was", "chased", "by", "the", "dog", "."]:
    updated_roles = session.push(word)

roles = ai.predict_roles("The cat was chased by the dog.")
```

Role outputs distinguish grammatical subject, actor, receiver and relation cue, with uncalibrated scores and subject BIO phrase labels. Adding words revises earlier predictions in the current sentence. Starting the next sentence resets the session. Predictions on observed prefixes never read future words. They can still be wrong.

## Train

```python
ai = MikteengAI()
ai.train_generation([
    {"input": "What is speed?", "answer": "Speed is distance divided by time."},
    {"input": "What is a cell?", "answer": "A cell is a basic unit of life."},
])
ai.save("models/my_model.mkteeng")
ai = MikteengAI.load("models/my_model.mkteeng")
```

This tiny example demonstrates the API, not meaningful accuracy. Each training call fits that head from the supplied dataset; it does not append knowledge automatically. Retain old data when retraining if you want to preserve it. Training is CPU based and synchronous.

| Method | Purpose |
|---|---|
| `train(data, task=...)` | Dispatch to generation, roles, sentences or passages |
| `train_generation(rows, sentence_state=True, use_subjects=False)` | Fit next-word generation; enable subject features after role training |
| `train_roles(rows)` | Fit revisable subject, actor, receiver, relation and BIO classifiers |
| `train_sentences(rows)` | Fit sentence actor/receiver prediction |
| `train_passages(rows, seed=811)` | Fit action-conditioned passage roles |
| `generate(prompt, max_words=85)` / `ask(...)` | Generate text and token probability trace |
| `predict_roles(text)` / `role_session()` | Inspect whole sentences or revisable prefixes |
| `predict(sentence)` | Predict sentence actor/receiver |
| `answer_passage(text, action)` | Predict actor/receiver of an action in a passage |
| `new_character_stream()` | Access the bundled earlier character state for incremental character updates |
| `save(path)` / `MikteengAI.load(path)` | Save/load a versioned local checkpoint |

Training a new role head does not silently change an already fitted generator. Retrain generation with `use_subjects=True` to connect the new role head. Models are separately trained; this is not joint end-to-end optimization. Older numeric and free-story modules are not included.

## Dataset formats

Generation: dictionaries with `input` and `answer` strings.

Sentence roles: `sentence`, `actor`, `receiver`, where lowercase answer words occur in the sentence.

Word roles: `words` plus equal-length `subject`, `actor`, `receiver`, `relation` lists (0/1, or -1 for unknown labels), and `subject_BIO` (`B`, `I`, `O`). Provide both positive and negative examples for each binary head and multiple BIO classes. See `docs.md` for examples.

Passages: `passage` plus `queries`, each containing `action`, `actor_index`, `receiver_index` into the tokenizer's word/punctuation sequence. Both roles are single token positions in this passage head; it is distinct from BIO word-role learning.

## Command line

```bash
mikteeng-ai models/biology_physics.mkteeng "Explain cells and velocity."
mikteeng-ai models/biology_physics.mkteeng "The cat was chased by the dog." --task roles
```

## Included model limitations

The bundled checkpoint preserves the latest tested research models. Role training: 3,238 annotated rows including repeats. Generation training: 1,028 synthetic single/paired biology/physics examples. Paired generation 108/110; single-answer generation 20/48. Exact subject spans: 720/720 on familiar-structure unseen entity pairs, but 0/120 on the tested new passive-question structure, and only 1/5 on held-out science facts. These are small synthetic experiments. They do not establish general understanding or reliable unseen-problem solving. Earlier single-only generation scored 35/48, so the richer system does not improve every task.

`experiments/` contains datasets and evaluation results separately from the installed engine. Models are under `models/`; installing the wheel does not automatically install datasets or a default model. Scores describe those saved checkpoints, not any new training run.

## Validation

Run `python -m unittest discover -s tests`. Source modules never train on import. The saved model uses only `mikteeng_ai` class paths and can load outside the project directory without legacy files on `sys.path`. See `VALIDATION.md` for this release's checks.
