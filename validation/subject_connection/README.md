# Existing subject/actor head connected to prediction-self responses

The existing learned RevisableRoles model from biology_physics.mkteeng is now connected to the existing QuestionConditionedPredictor and MikteengAI.ask(predicted_self=...). No domain answer dispatch, new tokenizer, or replacement core learner was introduced.

## Connection

The response features still include first-level prediction summaries. This optional path also retains ordered observation vectors and the existing head's predicted subject, actor, receiver and relation scores at each sentence/token/vocabulary position. Those role channels interact with the question features. They are not collapsed into a passage-wide role average. The old role head and observation/higher predictors are frozen; only response weights are fitted to the new input features.

A new public method is available in the updated library sources:

```python
from mikteeng_ai import MikteengAI
import json
ai = MikteengAI.load('models/question_conditioned_predictions.mkteeng')
role_source = MikteengAI.load('models/biology_physics.mkteeng')
original = json.load(open('data/question_conditioned_training.json'))
prior = json.load(open('validation/broad_patterns/training_additions.json'))
mixed = json.load(open('validation/subject_connection/mixed_training_additions.json'))
ai.train_subject_conditioned_questions(original + prior + mixed,
                                        role_model=role_source.roles)
snapshot = ai.build_sentence_prediction_self([
    'alice selects the beans.', 'alice plants the beans.',
    'alice waters the beans.', 'brian checks the maize.',
    'brian harvests the maize.', 'brian cleans the maize.',
    'brian stores the maize.',
])
print(ai.ask('Who waters the beans?', predicted_self=snapshot)['text'])
```

Load subject_connected_candidate.mkteeng with the UPDATED library to use the saved result directly. Candidate class names remain mikteeng_ai-qualified. Loading it with older feature code would not exercise the new connection. The updated source is under deploy/mikteeng-chatbot/src/mikteeng_ai in this repository. It can be installed from that package directory. The previous full archive supplies the original trained models and data needed for reproduction.

The optional subject-conditioned predictor takes priority for PredictionPatternSelf inputs when attached. Existing checkpoints without it keep their old path. Training attaches a fully fitted replacement only after fitting succeeds. It deep-copies the supplied role component. Retraining the donor does not silently change the attached component. No old saved checkpoint is overwritten. This does not connect roles to every other head, discover new grammatical categories, or change chatbot checkpoint selection.

## Controls and results

We compared: prior summary baseline; ordered observations without roles; ordered observations with roles; each of the last two with mixed-actor examples. Training replays 1,088 original rows and 192 prior additions. Mixed variants add 192 checked rows with different actors/crops in different sentence segments. New mixed training observation/question tuples do not overlap the evaluation cases. A fresh binding set uses a separate actor/crop partition and shift. All words and sentence structures remain familiar.

| Variant | Existing changing-actor cases | Fresh actor binding | Fresh object binding |
|---|---:|---:|---:|
| Prior prediction summary | 4/64 | 0/16 | 16/16 |
| Ordered observations, same training | 30/64 | 0/16 | 12/16 |
| Roles connected, same training | 32/64 | 0/16 | 12/16 |
| Ordered observations + mixed training | 64/64 | 16/16 | 16/16 |
| Roles connected + mixed training | 64/64 | 16/16 | 16/16 |

The prior-summary fresh scores are stored in results.json; consult the actual report if comparing variants. The large gain is supported by ordered binding preservation and mixed training; this benchmark does NOT demonstrate that role features were necessary. With identical original training, adding roles improved 30/64 to 32/64 on one actor set and did not fix the fresh set. There is no claim of independently validated causal benefit from subjects alone.

Final connected candidate: changing actors 64/64; changing objects 64/64; fresh actor/object binding 32/32; original answer retention 544/544. Missing-evidence and tested wording results were retained. New grammar/vocabulary, possession, explicit causes and multi-actor outputs retain their earlier limits. Existing role predictions themselves can be wrong; supplied probabilistic features are not guaranteed role truth. The report records sample head outputs for inspection.

All 93 full project tests pass, including two new connection checks. Zeroing role scores changes the actual response feature matrix; routing through ask and checkpoint roundtrip are tested. Candidate reload reproduced the full evaluation. Original checkpoint hashes are unchanged. The role donor's coefficients were unchanged in the unit test. Full source/report/candidate files are versioned separately.

## Reproduce

With the updated source and extracted original full project:

```bash
python validation/subject_connection/run_connection.py --project /path/to/mikteeng-ai --broad validation/broad_patterns --output /path/to/subject-results
```

It compares four refits against the earlier broad candidate, saves a connected candidate, records all results, checks disjoint new data, and checks reload equivalence. Original retention was evaluated separately using the 544 cases already stored in the broad report. These remain synthetic development/retention metrics, not general-language benchmarks. No external AI API, role-head retraining, first/higher retraining, or public deployment occurs in this task.
