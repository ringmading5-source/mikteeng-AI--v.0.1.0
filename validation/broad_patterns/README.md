# Broad pattern audit and controlled training of the existing prototype

This uses the earlier saved Mikteeng prediction-pattern implementation, not research/pattern-self. No architecture or domain-specific inference rules were added. Numeric checkpoints were evaluated only. A separate question-conditioned candidate was trained with the existing API, replaying 1,088 earlier labeled examples plus 192 checked synthetic additions. Original checkpoint hashes remained unchanged. The candidate is not the baseline or chatbot default.

## Tests

Twelve numeric families, ten contexts each, tested separately on the earlier linear checkpoint and adaptive expert bank: arithmetic, constant, doubling, tripling, halving, quadratic, cubic, alternating, period-three, triangular, Fibonacci and alternating growth. Each context predicts three steps. Expected continuations follow explicitly specified generating functions; a finite prefix does not uniquely determine a mathematical continuation. The linear checkpoint passed arithmetic/constant only (20/120). The adaptive bank passed arithmetic/constant/doubling/tripling (40/120). The others failed here. This is evidence about those saved models and test distributions, not a proof that the architecture cannot learn those families with other training.

Fourteen language families total 292 cases: paraphrase, missing relation, absent why evidence, negative question, changing actor, changing object, passive grammar, negated observation, new vocabulary, fresh question wording, fresh missing wording, multiple actors, possession and explicit cause. Expected labels and full outputs are retained. Multi-actor exact labels are ordered by mention and use 'and'; exact-match failure can include formatting differences, so it is not itself a complete semantic evaluation. Missing/negative questions request insufficient evidence because the observations do not establish the requested relation or its negation; absence of a positive action is not asserted to prove a negative fact.

## Before / after

| Group | Before | After |
|---|---:|---:|
| Broad language total | 97/292 | 148/292 |
| Missing relation | 0/16 | 16/16 |
| Absent why evidence | 0/16 | 16/16 |
| Negative question | 0/16 | 16/16 |
| Fresh missing wording | 0/8 | 8/8 |
| Paraphrase | 16/16 | 16/16 |
| Fresh positive wording | 8/8 | 8/8 |
| Changing actor | 9/64 | 4/64 |
| Changing object | 64/64 | 64/64 |
| Multiple actors | 0/16 | 0/16 |
| Original question-answer retention | 544/544 | 544/544 |

Passive grammar 16, negative-observation grammar 16, new vocabulary 4, possession 16 and explicit cause 16 were all rejected by the bounded sentence vocabulary in both versions. These 68 rejections are counted as unsuccessful cases in the broad total, separately from wrong generated answers. The negative-question score is not evidence of interpreting negated observations. The candidate's changed-actor regression means improved aggregate accuracy is insufficient for promotion.

## Training and separation

The additions are deterministic labeled question/observation examples authored in this workflow, not an external AI service's unchecked claims. They cover paraphrasing and withheld answers. Actor/crop combinations in new feedback and the matching evaluation groups are separated by a fixed partition. Exact new feedback observation/question tuples do not overlap the evaluation set. Fresh wording templates, changed-binding cases and boundary probes are excluded from new feedback. Original familiar training examples are intentionally replayed. This is a single controlled synthetic benchmark; its categories were chosen from known weaknesses, not a claim of a pristine broad real-world benchmark.

Only question_conditioned_predictor was refitted; first-level and higher prediction learners, sentence vocabulary and output architecture type are unchanged. Therefore this experiment tests question-response adaptation over existing self features, not coordinated adaptation of the entire hierarchy. Regression is preserved in results.json. The checkpoint broader_questions_candidate.mkteeng retains the separate trained result. The original models are not overwritten.

## Reproduce and feed training data

Extract the earlier full archive containing src/, models/, data/. From this repository:

```bash
python validation/broad_patterns/run_benchmark.py --project /path/to/mikteeng-ai --output /path/to/broad-results
python validation/broad_patterns/check_retention.py --project /path/to/mikteeng-ai --output /path/to/broad-results
```

The first command evaluates before training, writes separate data, trains and saves the candidate, then evaluates it. The second reloads the saved candidate and tests retention plus additional boundaries. Use compatible dependencies from the existing project. Pickle-based checkpoints must be trusted.

The callable existing training interface is:

```python
from mikteeng_ai import MikteengAI
import json
ai = MikteengAI.load('models/question_conditioned_predictions.mkteeng')
original = json.load(open('data/question_conditioned_training.json'))
checked_additions = json.load(open('training_additions.json'))
ai.train_question_conditioned_predictions(original + checked_additions)
ai.save('models/my_candidate.mkteeng')
```

New observation tokens outside the fixed sentence vocabulary cannot be added through this response-head training call. It is a full refit of that head, so replay existing examples. Do not promote a candidate based only on aggregate accuracy or on success at the training templates. Test unsupported inputs and old capabilities separately.

I can prepare datasets and run this workflow here. A permanent external teacher API is not connected by these files. If added later, it should propose examples into a review dataset, with source/check metadata and separate evaluation. Calling an external teacher for user answers would not measure Mikteeng's predictions.

This task does not deploy the candidate or change the chatbot. It tests and trains the existing prototype in its actual scope. Not all possible world patterns have been tested.
