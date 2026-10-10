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
