# Training and unseen-combination experiment

The unchanged bounded RSPM learner was trained across five seeds. Each run received 960 independent single-symbol transition observations: 480 per context, eight distinct triggers per context. Two opposite coordinate mappings used separate history keys. This is a controlled linear sequence-transition task, not natural language training or a test of temporal order discovery.

No mixed input vectors were included in training. Test combinations contain two to four familiar symbols with new positive coefficients. Nearby test inputs combine one familiar symbol with a small contribution from its neighbour. No training targets enter evaluation.

| Case | Queries across five seeds | Public matches | Mean operator diagnostic error | Mean nearest-example error |
|---|---:|---:|---:|---:|
| seen | 1000 | 1000 | 1.49e-16 | 0.0000 |
| unseen_combinations | 1000 | 0 | 1.3e-15 | 0.7979 |
| nearby_unseen | 1000 | 1000 | 3.85e-16 | 0.1194 |
| noisy_combinations | 1000 | 0 | 1.53e-15 | 0.7908 |

## Interpretation

The operator learned the supplied linear transformation, including novel positive combinations, without retrieving targets. This is limited compositional generalization within a fully covered training subspace. It does not show extrapolation to unknown symbols, discovery of arbitrary nonlinear rules, or language understanding.

Operator diagnostics access the existing context-local matrix directly to isolate prediction quality. They do not change the public routing gate. The public engine rejects all broad/noisy combinations in this test. Therefore this package does not claim end-to-end success on those rejected queries. Nearby combinations passed public routing and operator prediction, while stored centroids remained approximations.

All five evaluations left complete engine snapshots unchanged. Training the second domain preserved the first domain operator bit-for-bit. After evaluation, six conflicting observations produced two stored hypotheses for the same context/trigger. The trained checkpoint includes that conflict evidence.

## Reproduce

```bash
python train_generalization.py
python test_bounded_rspm.py
```

`generalization_results.json` contains per-seed curves and separate public, diagnostic, retrieval and baseline errors. `trained_bounded_generalization.json` stores the complete seed-42 trained state using explicit base64 arrays and JSON; it does not use executable pickle serialization. Use `pack`/`unpack` in the script with `snapshot`/`restore` to load it.

## Next engineering question

How should the public engine issue a tentative operator forecast when history matches but the trigger is a novel combination? That policy must retain ambiguity rejection and distinguish a learned forecast from an established stored rule. The current training experiment intentionally leaves the learner and routing policy unchanged.
