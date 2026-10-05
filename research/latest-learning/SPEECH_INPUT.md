# Acoustic input in the original Mikteeng model

The original MikteengAI class now provides train_speech_predictions, build_speech_prediction_self and predict_speech. The speech head uses the same original PredictionPatternLearner: prior observation windows -> first regression predictions -> prediction-history Self -> higher regression response. This adds a modality head inside the model; speech and text do not yet share learned semantic weights.

An AcousticEncoder converts bounded mono PCM samples into full orthonormal DCT frames. All coefficients are retained: amplitude and temporal information can be reconstructed, within numerical precision. This is a deterministic representation, not a pretrained speech encoder. Original integer byte recovery remains handled by the byte tokenizer. Input sample rates must match explicitly. No automatic resampling, downmixing or transcription occurs.

Sound prototypes are discovered from normalized frame similarity and stored in bounded nodes. Silence is counted separately. The sound-node registry is currently a diagnostic catalogue, not the feature input used by the prediction regressions. Similarity does not establish that nodes are phonemes or words. Learned word boundaries, grounding and speech-language understanding remain unfinished.

Prediction decodes generated acoustic vectors back to a waveform. Outputs outside [-1,1] are clipped and the clipped-sample count is reported. The result is experimental waveform continuation, not a text-to-speech system. Frame rollout is capped at 64 steps. WAV output is PCM16. Sound data should be evaluated with new speakers and sessions before any language-learning claim.

Validation:
- Arbitrary floating sample reconstruction maximum error: 2.22e-16.
- Twelve held-out phases of familiar 220/330/440 Hz synthetic tones: mean prediction MSE 2.12e-15; repeating the last frame MSE 0.0765.
- Prediction horizon was only three 128-sample frames: 24 ms at 16 kHz. These are simple predictable signals, not real speech or unknown-frequency tests.
- Saved model reload reproduced output.
- Five speech tests passed: reconstruction, silence, PCM scaling, invalid inputs and exact train-group duplicate rejection.
- Existing hardening checks retained the original sentence score of 156/192; eight tokenizer tests passed.

Example using recordings with no written transcripts:
```python
from scipy.io import wavfile
from mikteeng_ai import MikteengAI
# Each tuple is (mono_samples, sample_rate), all at the configured rate.
primary = [wavfile.read(p) for p in primary_paths]
secondary = [wavfile.read(p) for p in self_paths]
primary = [(samples, int(rate)) for rate, samples in primary]
secondary = [(samples, int(rate)) for rate, samples in secondary]
ai = MikteengAI().train_speech_predictions(
    primary, self_recordings=secondary,
    sample_rate=16000, frame_size=256, max_sound_nodes=256,
)
state = ai.build_speech_prediction_self(context_samples, 16000)
response = ai.predict_speech(state, frames=16)
ai.speech_prediction_model.write_wav('predicted.wav', response)
```

Use short clips initially. Each recording must contain at least observation_window + prediction_window + 1 frames. Full input limit is 60 seconds per mono clip. Node capacity may be reached on diverse real recordings; the system rejects overflow rather than silently discarding evidence. Training uses full clip vectors, not streaming regression. Long-term speech generation will require further representation/training work. No GPU or external model API is required for this implementation.
