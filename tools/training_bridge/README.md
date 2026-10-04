# External teacher training bridge

This bridge feeds the existing subject-connected Mikteeng learner. It does not replace the core model or use the teacher to answer chatbot users. An external service is contacted only when `propose` runs. There is no background schedule.

Install the existing deployment package and use its updated source:

```bash
python -m pip install -e deploy/mikteeng-chatbot
```

Configure `MIKTEENG_TEACHER_API_KEY` as a server environment variable. Keep it out of source control. Supply an HTTPS endpoint implementing the chat-completions request/response format and the model identifier supported by that service. No credential is included and a live service has not been tested.

```bash
python tools/training_bridge/bridge.py propose \
  --endpoint "$MIKTEENG_TEACHER_ENDPOINT" --model "$MIKTEENG_TEACHER_MODEL" \
  --checkpoint validation/subject_connection/subject_connected_candidate.mkteeng \
  --topic 'Different actors performing different actions in a passage' \
  --count 16 --output pending_examples.json
```

Check proposed answers against their observations, correct or discard mistakes, and save the checked rows as `checked_examples.json`. Structural validation is not factual verification. The teacher sees the topic and observation vocabulary, not the full checkpoint or private training dataset. Proposals remain limited to familiar observation words; this bridge does not fix unfamiliar-word encoding.

Refitting requires the full original training replay (1,088 original rows, 192 broad additions, and 192 mixed actor rows). Recover the original `data/question_conditioned_training.json` from the full project archive, then create the replay:

```python
import json
from pathlib import Path
paths = ['data/question_conditioned_training.json',
         'validation/broad_patterns/training_additions.json',
         'validation/subject_connection/mixed_training_additions.json']
rows = [row for path in paths for row in json.loads(Path(path).read_text())]
Path('replay.json').write_text(json.dumps(rows))
```

Use an independent held-out set with `observations`, `question`, and `answer`. Do not use training examples as held-out tests. Include previously correct cases from the original retention set as well as actor/object tests to protect those abilities.

```bash
python tools/training_bridge/bridge.py train \
  --checkpoint validation/subject_connection/subject_connected_candidate.mkteeng \
  --replay replay.json --checked checked_examples.json \
  --holdout validation/subject_connection/fresh_evaluation_cases.json \
  --output candidates/teacher_candidate.mkteeng
```

A new checkpoint is saved only if every previously correct held-out answer stays correct. This gate applies only to the provided holdout and cannot guarantee general accuracy. It does not require improvement. Training/evaluation overlap and conflicting labels are rejected. The original checkpoint remains unchanged. The CLI refuses existing output paths. Checkpoints use pickle: load only trusted model files.

Validation:

```bash
python -m unittest discover -s tools/training_bridge -p 'test_bridge.py' -v
```

Tests cover a mocked teacher request, missing credentials, evaluation leakage, regression rejection, and real existing learner retraining/checkpoint roundtrip. They do not establish a live external API connection or new generalization results.
