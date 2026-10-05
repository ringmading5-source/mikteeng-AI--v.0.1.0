# Mikteeng fixed-basis bounded RSPM

The coherent memory-managed implementation is `bounded_rspm.py`. Earlier `engine.py`, `context.py`, and `discovery.py` remain comparative experiments; their results are not results of the new implementation.

## Run

Requires Python 3.10+ and NumPy. From this directory:

```bash
python test_bounded_rspm.py
python benchmark_bounded.py
```

```python
from bounded_rspm import BoundedRSPM
engine = BoundedRSPM(dim=64)
# Supply finite, nonzero 64-dimensional vectors:
engine.train(history_vector, trigger_vector, target_vector)
answer = engine.query(history_vector, trigger_vector)
```

## Actual mechanisms

A seeded QR basis B stays fixed. L(v) normalizes the positive basis projection. History selects a context using cosine similarity and an ambiguity margin. A second cosine/margin gate selects a trigger inside that context. Context-local R predicts L(t + R t). Outcome stores maintain separate confirmed modes and provisional candidates, each with vector sums, counts and normalized directional centroids. Vector sums avoid repeated averaging of already normalized means.

Training captures a pre-target forecast, advances a global training clock, expires provisional evidence across every store, routes or allocates memory, then updates evidence. R adapts only when the store has one confirmed mode; it freezes for that store's multimodal observations. A context becomes persistent after sufficient accurate pre-update forecasts. No learned global basis or external language model is used.

Inference validates input, computes routing and returns copied hypotheses/probabilities. It does not change time, usage, TTLs, weights, allocation counters or the basis. Ambiguous and unmatched queries have explicit statuses and no forecast hypotheses. Invalid input fails before state changes.

Contexts, triggers per context, confirmed modes and candidates all have explicit capacity limits. Context and trigger allocation recycle provisional records first. Confirmed memory is protected and saturation returns capacity rejection. Rejected outcome promotion retains bounded provisional evidence; it does not report successful promotion or silently replace a confirmed mode. This means unrestricted novelty cannot be retained when all capacity is confirmed.

Snapshots include configuration, all stores, statistics, counters and basis immutability metadata. The recursive comparator checks array shape, dtype and bytes, including signed zeros. Snapshot restore re-freezes B. Hypothesis arrays returned by query are copies.

## Deliberately conservative merging

`merge(a, b)` returns MERGED or NO_OP with a reason. It stages a complete copy and commits only after validation. It requires similar histories, bounded operator difference, identical one-to-one trigger keys, no provisional candidates, identical one-to-one confirmed centroids, compatible mode frequencies and negligible survivor prediction change on removed triggers. It keeps the survivor's operator rather than averaging operators, merges directional sums/counts and removes the other context as a whole. A rejected merge leaves the complete snapshot unchanged.

This narrow exact correspondence policy avoids claiming functional equivalence from a few nearby keys. Approximate Hungarian correspondence and union of untested trigger domains are not implemented. They remain future work. This implementation does not copy the unsafe merge snippets into production.

## Verification

Ten targeted tests cover local predictor learning and promotion; read-only matched/unmatched queries and safe returned arrays; conflicting-context isolation; a 100-observation novelty flood; multimodal retention and outcome capacity rejection; true trigger ambiguity; global TTL expiration and monotonic trigger IDs; conservative merge success and unchanged rejected-merge state; signed-zero snapshot restoration; protected trigger capacity; invalid-input rejection.

The separate reproducible benchmark uses 64-dimensional synthetic vectors, explicitly supplied histories/triggers/targets, two established rules, 1,000 novel observations and 100 independent noisy frozen queries. Recorded results: 1,000 new context allocations, zero capacity rejections, 100/100 correct stored-outcome retrievals, two persistent contexts retained, ten final contexts, unchanged basis and full inference snapshot. These are stored-outcome retrieval results, not unseen-rule prediction accuracy.

## Limits

This is an importable experimental vector learner, not a chatbot or trained language/audio model. History and trigger representation are supplied by the caller. No claim of universal intelligence or production readiness follows from these synthetic tests. History ambiguity is rejected; hidden context variables cannot be inferred from identical observable histories. Promotion requires consistent parametric accuracy and may not occur for intrinsically conflicting domains. Capacities trade novel intake against retention explicitly. Counters grow with lifetime; allocation bounds constrain stored arrays/records, not every integer's eventual bit length.
