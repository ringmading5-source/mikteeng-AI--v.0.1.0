# Mikteeng RSPM voice-training prototype

Built two operational modes using the existing fixed-basis RSPM learner and Unicode transcript adapter. Separate audio model: previous English checkpoints and operators are untouched. No microphone upload service or hosted deployment added.

- paired: WAV + transcript -> supervised audio-vector to transcript-vector learning. At prediction, the operator's vector is decoded by selecting the nearest transcript in a bounded known-transcript list. This is WHOLE-KNOWN-TRANSCRIPT CLASSIFICATION, not free-form speech recognition, generated unseen transcripts, translation, or meaning learning.
- voice-only: WAV -> bounded acoustic pattern discovery using RSPM multimodal outcome memory. Predict returns a checkpoint-local acoustic pattern index or unknown. No words/meanings/transcripts learned without labels. Pattern index is not a stable semantic ID.

Audio encoder: PCM/float WAV,8–96 kHz, mono/stereo up to8 channels, <=60 seconds and64MiB. Resamples to16 kHz, removes DC, normalizes amplitude, summarizes64 spectral bands into256 dimensions using means, deviations, and first/second-half summaries. Projects into the existing fixed basis. This summary loses detailed phoneme order and is not a production speech encoder. MP3/M4A unsupported; convert recordings to WAV before training.

## Install and run

From the parent of rspm_voice:

```bash
python -m pip install ./rspm_voice
python -m rspm_voice.demo
python -m rspm_voice.cli train --manifest rspm_voice/demo_data/manifest.jsonl --out paired_run --mode paired --epochs 20
python -m rspm_voice.cli predict --checkpoint paired_run/model.json --audio rspm_voice/demo_data/tone_0_4.wav
python -m rspm_voice.cli train --manifest rspm_voice/demo_data/manifest.jsonl --out acoustic_run --mode voice-only --epochs 20
```

An installed CLI is also available as mikteeng-voice. Source execution works with python rspm_voice/cli.py. requirements.txt pins the dependencies tested in this environment. Checkpoint Unicode database metadata must match the runtime when loading; incompatible encoders fail explicitly.

## Real Dinka recordings

Create a UTF-8 JSONL manifest beside the WAV files. Start with paired_manifest_template.jsonl or voice_only_manifest_template.jsonl, replace placeholder paths/transcripts with your actual recordings and correct Unicode transcripts. Each line is one recording. Use split=train/test. Audio paths must remain beneath the manifest directory. Test speakers must be excluded from training if speaker_id metadata is supplied. Exact audio-byte duplication across train/test rejected.

```json
{"audio":"train_01.wav","transcript":"YOUR_DINKA_TRANSCRIPT","split":"train","speaker_id":"speaker_A"}
{"audio":"test_01.wav","transcript":"YOUR_DINKA_TRANSCRIPT","split":"test","speaker_id":"speaker_B"}
```

For voice-only omit transcript. Paired test transcripts should initially be phrases present in training but recordings/speakers should differ. With no transcript vocabulary for an unseen phrase the current model cannot freely transcribe it. Voice-only evaluation reports match/unknown counts, not semantic accuracy.

```bash
mikteeng-voice train --manifest my_recordings/manifest.jsonl --out dinka_paired --mode paired --epochs 20
mikteeng-voice train --manifest my_recordings/manifest.jsonl --out dinka_more --mode paired --epochs 10 --resume dinka_paired/model.json
```

Outputs model.json and report.json with before/after held-out metrics, rejected cases, and training/capacity events. Missing test rows mean no accuracy evidence. Labels are required to learn correspondence, not simply to store sounds. All predictions remain UNVERIFIED; cosine scores are not calibrated confidence probabilities.

## Python API

```python
from rspm_voice import VoiceModel
m = VoiceModel(mode="paired")
m.train("recording.wav", "YOUR_DINKA_TRANSCRIPT")
result = m.predict("other_recording.wav")
m.save("model.json")
m = VoiceModel.load("model.json")
# Unsupervised acoustic discovery:
a = VoiceModel(mode="voice-only")
a.train("recording.wav")
```

Capacity: paired64 distinct transcript strings,64 triggers/context. Voice-only16 confirmed acoustic patterns and16 bounded candidates. Capacity rejection reported; no unbounded raw audio stored in checkpoints. Training retains the existing conservative store/promotion and trigger-ambiguity policies, so learning can stall. Raw recording files remain your dataset.

## Validation actually performed

Generated tones, NOT human speech and NOT Dinka recordings:12 training clips across3 tone classes;9 held-out clips with altered frequency, phase, amplitude/duration and sample rate. Synthetic generator IDs exercise speaker-split checks; this does not prove cross-speaker speech accuracy.

- paired:9/9 held-out known-tone transcripts selected; a different tone rejected.
- voice-only:3 confirmed acoustic patterns;9/9 held-out clips matched; different tone rejected.
-8 tests pass: paired pipeline, voice-only clustering/counts, Unicode transcript, read-only inference, bitwise checkpoint continuation, invalid/silent input, stereo/float WAV, split leakage, bounded label-capacity rejection.
-Local installable wheel build and module CLI verified.

Run tests with OPENBLAS_NUM_THREADS=1 python rspm_voice/test_voice.py. Demo source, manifest templates, and measured summaries included. Run the demo to generate WAV clips and synthetic checkpoints locally. Real speech recordings are still needed to measure usefulness. This build does not claim real Dinka ASR or generated speech.
