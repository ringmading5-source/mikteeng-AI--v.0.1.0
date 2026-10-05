# Mikteeng experimental chatbot

## Run locally

```bash
python -m pip install -r requirements-chatbot-tested.txt
python -m pip install .
python build_model.py
python start_chatbot.py
```

Open http://127.0.0.1:8000. For Render use the repository root render.yaml blueprint, which runs this directory's build and start commands.

## October 5 learning update

The chatbot vendors the latest implementation from research/latest-learning/src and builds a single checkpoint with the existing definition/continuation heads, learned transition and sentence-pattern heads, a character completion head, and the synthetic acoustic head. No external AI provider is used. It does not train on user messages, run external actions, or persist uploaded WAV files. This does not establish cross-modal semantic learning.

Choose an answer mode in the interface:

- Mikteeng generation: experimental token generation for introductory concepts.
- Sentence prediction: continuation using editable preceding sentences.
- Stored reference: exact reviewed curriculum definitions.
- Word completion: a lowercase English word beginning such as looki; the character head predicts a bounded completion and can produce invalid forms.
- Structured planning: edit the supplied JSON state/goal/actions for binary fields a, b, c. A generated plan composes learned toy transitions; it does not interpret free-language tasks or execute actions.
- Waveform continuation: upload mono PCM16 WAV at 16 kHz, 40 ms to four seconds. The model predicts the next 128 ms. It was trained on synthetic sounds, so real speech quality and text-to-speech remain unverified.

Build loads the repository-owned composition and acoustic seed checkpoints with explicit trust, adds the existing curriculum heads, and fits the word head using the original 36 training words. Its generated checkpoint stays under models/ and is recreated during each build. Never replace seed files with untrusted pickles.

## API and verification

GET /health returns release 2026-10-05-learning and active capabilities. GET /api/coverage lists curriculum topics. POST /api/chat accepts message, mode (generated/reference/continuation/word/composition), and optional passage context. For composition, message contains the structured JSON as a string. POST /api/acoustic accepts raw bounded WAV bytes and returns a base64 WAV prediction plus limitations. Health readiness means the checkpoint loaded; it does not certify response accuracy.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Integration tests cover existing text modes, learned word completion, multi-step planning and forbidden-state rejection, WAV output, malformed inputs, and same-origin protection. The JavaScript passes node --check. Live deployment must be checked separately using its actual Render URL and /health release field.
