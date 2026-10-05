# Acoustic training learning curve

Same architecture and hyperparameters at every stage. The learner was refitted on growing synthetic datasets, not updated using an incremental optimizer. Each clip contains 40 frames at 128 samples/frame and 16 kHz, so each is 0.32 seconds. Tests were fixed before fitting: 80 combinations, held-out phases, two horizons (24 and 128 ms), totaling 160 cases. No real speech was supplied.

| Training recordings | Mean waveform MSE | Wins against both trivial baselines | CPU fitting seconds |
|---:|---:|---:|---:|
| 24 | 0.016495 | 104/160 | 0.063 |
| 32 | 0.014130 | 128/160 | 0.079 |
| 160 | 0.010833 | 128/160 | 0.364 |
| 320 | 0.010757 | 128/160 | 0.621 |

A win requires at least 10% lower waveform MSE than both repeating the last frame and predicting zero. This is not speech accuracy. Overall error fell approximately 35% between the first and last stages. Gains from 160 to 320 clips were under 1%.

The datasets change coverage as well as count: first three familiar steady frequencies, then four, then rising/falling pitch, amplitude ramps and mixtures, then more phase examples of the same families. Only the final doubling isolates additional phase examples within the same task families. Training sets are nested. Larger, more varied fitting data made steady-tone predictions worse than the focused four-frequency checkpoint. Long rising/falling-pitch continuations still failed the two-baseline win criterion at 128 ms. No checkpoint is promoted as universally better.

Each stage stores a checkpoint. The JSON report includes all cases, category scores, sound-node counts and timings. The original algorithm was not changed; the fit is CPU-only. Power consumption was not measured. Synthetic recordings are regenerated from the included script; they are not human voice data. Sound nodes remain a diagnostic catalogue and do not drive acoustic prediction.

Run `PYTHONPATH=src python examples/train_acoustic_curve.py`. The audio encoder plus original prediction/Self loop are included. Next improvement should target variation-specific routing or representations and verify it on separate real recordings; more identical data alone is not supported as a solution by this curve.
