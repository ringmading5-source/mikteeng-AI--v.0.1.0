# Data and state interfaces

The public API is `mikteeng_ai.MikteengAI`. Classes in `_models` are private implementation details. Version 0.1.0 uses checkpoint schema 1; unknown schemas are rejected. Pickle checkpoints depend on compatible NumPy/scikit-learn versions. The included checkpoint was built using the versions in `requirements-tested.txt`.

## Annotated roles

```python
row = {
    "words": ["the", "cat", "chased", "the", "dog", "."],
    "subject": [1, 1, 0, 0, 0, 0],
    "actor": [1, 1, 0, 0, 0, 0],
    "receiver": [0, 0, 0, 1, 1, 0],
    "relation": [0, 0, 1, 0, 0, 0],
    "subject_BIO": ["B", "I", "O", "O", "O", "O"],
}
```

A dataset needs varied examples; one row cannot teach a general parser. An unannotated semantic role uses -1 and is masked from that head's training. Scores are separate binary classifier probabilities; they need not sum to one across roles. A phrase can be subject and actor simultaneously. BIO labels are independent predictions and are not repaired by a handcoded parser.

## Token indexing for passage training

Punctuation is a token. `the cat chased the dog .` indexes as 0=the, 1=cat, 2=chased, 3=the, 4=dog, 5=period.

```python
row = {
    "passage": "The cat chased the dog.",
    "queries": [{"action": "chased", "actor_index": 1, "receiver_index": 4}],
}
```

At prediction time the API receives the action word, not role indexes. The model scores candidate tokens from learned weights.

## Training order

1. Fit annotated word roles with `train_roles`.
2. Fit question/answer next-word generation with `train_generation(use_subjects=True)`.
3. Independently fit optional sentence and passage role heads.
4. Save the complete model.

Generation uses character n-grams for the input question, word-prefix features and sentence summaries. Subject-enabled generation adds predicted word roles. The earlier recurrent character encoder is available in the bundled sentence model, but does not automatically train from generation examples. Character-stream snapshots grow with consumed text. No external model, search service, hosted account or GPU is required for these components.
