# Continued training and data-size experiment

This keeps the sequence architecture fixed at 22,265 parameters. It compares 20, 80 and 240 training examples, each from the same initialization (seed 7) and trained for 120 epochs. It also loads the earlier 20-example pilot checkpoint and continues it on the 240-example corpus for 120 additional epochs. Optimizer moments reset when train() is called; inference weights continue from the saved checkpoint.

The larger corpora contain the original 20 pairs plus deterministic synthetic examples with new identifiers. All examples use the same three patterns: greeting a named person, producing an addition function, or producing a subtraction function. This tests more examples within these tasks, not diverse real conversation, reasoning, speech, or general coding.

Thirty test prompts use fresh identifiers excluded from every training set. Exact full-response matching is required. No generated code is executed. Copying is evaluated separately, and out-of-template abstention is checked on six prompts. Hyperparameters and training durations are fixed before test evaluation. This is a one-seed pilot; more data also means more optimizer updates because epochs are held constant, so the experiment does not isolate dataset size from total compute.

## Reproduce in Colab

```python
!git -C /content/mikteeng pull --ff-only
%cd /content/mikteeng/tbin_1_1/sequence_learning
!OPENBLAS_NUM_THREADS=1 python train_scaling.py
```

The script writes checkpoints, full predictions and exact corpora to `scaling_run/`. The continued checkpoint and learned copying model are included in GitHub. The other three independent checkpoints can be reproduced locally with the script.

## Use the continued checkpoint

```python
from sequence import SequenceLearner
m = SequenceLearner.load('scaling_run/continued_model.npz')
print(m.generate('My name is Mading. Say hello.'))
```

Outputs remain experimental. For the separately measured copying behavior:

```python
from copy_conditioning import CopyConditioner
c = CopyConditioner.load('scaling_run/copy_model.json')
print(c.generate('My name is Mading. Say hello.'))
```

The report is `scaling_run/results.json`; it includes every test prediction, losses, timing, exact counts, and limitations. It does not claim real-world readiness or Dinka speech training.

## Recorded results

| Run | Exact training answers | Exact new answers |
|---|---:|---:|
| from_scratch / 20 examples | 1/20 | 0/30 |
| from_scratch / 80 examples | 1/80 | 0/30 |
| from_scratch / 240 examples | 1/240 | 0/30 |
| continued_prior_pilot / 240 examples | 1/240 | 0/30 |

Copying separately: 30/30. Out-of-template abstentions: 6/6. Checkpoint reload preserved predictions.
