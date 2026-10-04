# Existing prediction-pattern prototype: fresh validation

This test uses the earlier mikteeng_ai implementation and its existing saved checkpoints. It does not use research/pattern-self, change architecture, or retrain the audited checkpoints. Hashes before and after inference agree. The project unit suite also runs its existing small fitting/roundtrip tests with temporary models; those are separate from the unchanged checkpoint inference audit.

Results:

| Test | Fresh result |
|---|---|
| Existing full project unit suite | 91 passed, 0 failures |
| Numeric continuation (12 contexts, 3-step predictions) | 12/12 contexts correct within 1e-5 |
| Separate learned pattern routes (+2, doubling, tripling examples) | 3/3 correct within .01 |
| Sentence continuation on held-out familiar name/crop combinations | 192/192 sentences, 64 contexts |
| Question + prediction-self evidence responses | 384/384 |
| Question + prediction-self generated responses | 544/544 |

These fresh inference evaluations reproduce existing task results; they are not new unseen-domain benchmarks. Sentence vocabulary and grammatical patterns are familiar. Numeric tests are bounded continuation families. Five separate saved checkpoints cover these paths; the report does not claim one checkpoint implements every task.

Mechanism exercised: observations are supplied to build_prediction_self/build_sentence_prediction_self; first-level predictions are created; self stores their prediction window; respond_prediction_self/continue_sentence_patterns use the trained higher predictor; ask(question,predicted_self=snapshot) uses the existing question-conditioned response path. The sentence checks verify that the snapshot pattern equals the flattened last prediction window. Learned pattern routing keeps the numeric example paths distinct. Checking that this path executes does not establish that every question answer requires the higher level; prior ablations remain relevant.

Examples reproduced:

- [2,4,6,8] was covered by the unit tests and predicts approximately [10,12,14].
- [4,8,16] predicts approximately [32,64,128] in the saved adaptive-pattern checkpoint.
- alice plants beans / alice waters beans / alice checks beans continues with alice harvests beans / alice plants beans / alice waters beans.
- Who waters the beans? is covered by the question-self unit test and yields alice waters the beans in the evidence-reader checkpoint.

Limits observed directly in the generated-answer checkpoint: Who eats the beans? returned alice without supporting evidence. Why does alice water the beans? also returned alice. A zebra/eats/leaves passage was rejected because words are outside the trained sentence vocabulary. These failures define the scope of the current working prototype; they do not invalidate its successful tested cases or require rebuilding it before integration.

The detailed JSON includes every evaluated answer, continuation and numeric case, checkpoint SHA256 values, and the false retraining/new-research flags. No further algorithm change was made.

To reproduce, extract the earlier full project archive with src/, models/ and data/. Run from this repository:

```bash
python validation/test_existing_prototype.py --project /path/to/mikteeng-ai --output /path/to/results.json
```

For the suite, from the extracted project folder:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The checkpoints use trusted Python pickle internally; use the provided trusted project models and compatible dependencies listed in requirements-tested.txt. The deployment source folder alone does not include all five audited trained checkpoints. The live chatbot remains unverified and was not changed in this task.
