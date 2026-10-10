# Update: learned copying companion

The new `copy_conditioning.py` learns how one changing span in a prompt maps into an output. It infers common prefixes/suffixes from paired demonstrations; it contains no hardcoded greeting, addition, subtraction, or function-name templates. The copying algorithm itself is designed by us. It is a restricted symbolic learner, not neural attention.

## Measured result

Using the same 20 training examples and **14 fresh tests** (six names, eight Python function identifiers):

| Method | Exact answers |
|---|---:|
| Previously trained neural generator alone | 0/14 |
| Learned copying wrapper | 14/14 |

All eight generated coding answers matched the requested function AST, including name, arguments, and operator. Generated code was not executed. Six out-of-pattern prompts all returned abstentions. This includes unfamiliar wording and an untrained multiply operation. These are selected synthetic tests, not an estimate of real-world accuracy. See `copy_results.json` for every prompt and answer.

The underlying neural weights have not improved. Disabling copying returns to the 0/14 baseline. The wrapper defaults to abstention on unknown or conflicting templates; unverified neural fallback must be enabled explicitly.

## Run on your phone in Colab

```python
!git -C /content/mikteeng pull --ff-only
%cd /content/mikteeng/tbin_1_1/sequence_learning
!python -m unittest test_copy test_sequence -v
!python evaluate_copy.py
```

The evaluation reuses `sequence_pilot.npz` if present, or trains the original neural pilot if absent. The saved `copy_model.json` contains the learned templates and supports immediate use:

```python
from copy_conditioning import CopyConditioner
model = CopyConditioner.load("copy_model.json")
print(model.generate("My name is Mading. Say hello."))
print(model.generate("Write Python function sum_two to add a and b."))
```

To fit different examples, use `CopyConditioner().fit(rows)` with prompt/response dictionaries. At least two different demonstrations must support a rule. Training is a pairwise search capped at 256 examples; it is not a large-corpus trainer. One nonempty span is copied unchanged, up to 128 characters; newlines inside that span are rejected. Exact surrounding wording must match. Unicode is preserved. Multiple slots, paraphrases, new algorithms, and speech transcription remain unsupported.

Eight combined unit tests passed: the original three neural checks plus five copying checks covering new Unicode spans, contradictions, conflicts, persistence, bounds, and explicit fallback. The new code and results are a separate optional companion to TBIN 1.1; no changes to its existing multimodal matcher.

---

# TBIN sequence-learning experiment

Optional research companion to TBIN 1.1. This adds a byte-by-byte response generator initialized from random weights, using NumPy on CPU. It is a conventional small recurrent neural network trained with backpropagation through time and Adam, not a braid implementation or a pretrained external AI.

## Outcome: not ready for use

The recorded pilot uses 20 synthetic examples, 120 epochs, and one seed:

- Mean teacher-forced training loss: 5.5328 -> 0.3384.
- Exact training answers: 1/20.
- Exact held-out answers: 0/8.
- Held-out prompts repeatedly generated `Hello, Ian!`, including coding prompts.

Lower teacher-forced loss did not yield dependable free-running generation. Names and function identifiers were held out, but templates were shared. No speech training or real conversation corpus was used. Do not describe this result as coding or conversation capability. This component does not yet consume TBIN audio/image embeddings or replace its matching learner.

## Run in Colab

After cloning the repository:

```python
%cd /content/mikteeng/tbin_1_1/sequence_learning
!python -m unittest test_sequence -v
!python evaluate_sequence.py
```

Evaluation saves `sequence_pilot.npz`, `sequence_results.json`, `train.jsonl`, and `heldout.jsonl` locally. To try your own small JSONL corpus:

```jsonl
{"prompt":"Hello","response":"Hello!"}
{"prompt":"What is your name?","response":"TBIN"}
```

```bash
python sequence.py train your_data.jsonl --epochs 120 --model your_model.npz
python sequence.py generate your_model.npz "Hello"
```

The generator uses 256 UTF-8 byte values plus one end-of-response control symbol, 22,265 parameters, and 89,060 bytes of raw float32 weights. It uses a 512-byte training-example cap and defaults to a 128-byte generation cap. Invalid generated UTF-8 is flagged and displayed with replacement characters. It loads the small text corpus into RAM; it is not the streaming TBIN multimodal trainer. Optimizer state resets on each train() call; saved NPZ restores inference weights, not exact training state.

Three unit checks passed: numerical gradient agreement, checkpoint round trip, and input/output length limits. Generated Python is parsed only; it is not executed. Syntax success would not establish functional correctness.

## Next experiment

Investigate prompt retention using attention or explicit copying, and compare on a separate validation set before touching the held-out test set. Broader conversation and coding need representative data and stronger sequence conditioning. Speech transcription additionally needs a time-preserving acoustic encoder and alignment/decoding; this text experiment does not add those mechanisms.
