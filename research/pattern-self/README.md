# Pattern Self: independent mechanism prototype

A standalone standard-library Python experiment. It does not use sklearn, neural networks, or domain-specific answer dispatch. It is separate from the deployed chatbot and does not replace its trained models.

## Mechanism

For every observation sequence, count outcomes following each suffix context up to a chosen maximum length. Self is the collection of these context/outcome counts plus checked prediction/outcome records.

For context c and outcome y:

    N(c,y) <- N(c,y) + 1
    p(y|c) = N(c,y) / sum_z N(c,z)
    prediction = argmax_y p(y|c)

At prediction, select the longest observed suffix. If that suffix is unknown, fall back to a shorter suffix. This is a programmed inductive assumption, not a learned context-selection algorithm. Conflicting outcomes remain in separate counts rather than overwriting each other. Feedback increments transitions to the actual checked outcome; older evidence remains.

NumericPatternSelf explicitly represents successive differences d_i = x_(i+1)-x_i, learns those transitions, and reconstructs the next value x_next=x_last+predicted_difference. Difference encoding and reconstruction are designed representations; the observed change values are learned. This is a conditional-frequency model implemented directly, not a claim of a novel established algorithm. Text input uses supplied hashable tokens; token grouping is not discovered here.

## Use

```python
from pattern_self import NumericPatternSelf, PatternSelf
ai = NumericPatternSelf().observe([2,4,6,8,10])
print(ai.predict([100,102,104,106]))  # 108

self = PatternSelf().observe(['alice','holds','book'])
self.observe(['brian','holds','cup'])
print(self.predict(['alice','holds']))
self.feedback(['alice','holds'], 'bag')
```

Run `python -m unittest -v` from this folder.

## Results and limits

Seven unit checks passed, including checks exposing failure behavior. On 300 held-out numeric starting values with familiar +2, +3 and -2 changes, 300 predictions were correct. All three changes were present in training; this is transfer of known changes, not discovery of unseen transformations. An untrained +5 case predicted 117 instead of 120. Untrained negation predicted grow from seed/no/water, demonstrating unsafe suffix fallback. Frequency is not calibrated confidence. It memorizes local transition statistics and composes compatible suffixes; it cannot establish general understanding, semantic role discovery, or arbitrary tool actions. Checked prediction records are retained but are not yet a recursive prediction-pattern learner. There is no learned grouping, neural encoder, or independent meta-pattern adaptation in this prototype.

Next research question: can a learned context-selection criterion improve held-out predictions and reject misleading short-context matches without per-domain rules?

## Connected prediction-outcome self

`AdaptivePredictionSelf` wraps `PatternSelf`. Each observed suffix length proposes an outcome. Before feedback updates anything, it records the proposals and compares each with the checked actual outcome. Its meta state retains success/failure evidence separately for `(last observed token, context length)`.

With retention rho (default 0.95) and correctness indicator I:

    success <- rho * success + I
    failure <- rho * failure + (1-I)
    selection_score = (success+1)/(success+failure+2)

Choose the available candidate with highest score. Ties prefer a longer context. The symmetric prior, grouping by final token, context-length choices and retention constant are architectural assumptions. Retention changes the reliability evidence only; the underlying observation counts are kept. Scores are not calibrated confidence. Feedback is external checked truth, not the model endorsing its own output. The meta layer learns patterns in prediction correctness; it does not yet encode all relationships across prediction sequences or recursively create further levels.

```python
from pattern_self import PatternSelf, AdaptivePredictionSelf
base = PatternSelf().observe(['a','signal','wrong'])
for name in ['b','c','d']:
    base.observe([name,'signal','right'])
ai = AdaptivePredictionSelf(base)
print(ai.predict(['a','signal']))
ai.feedback(['a','signal'], 'right')
print(ai.predict(['a','signal']))
```

For an isolated selector experiment use `update_patterns=False` in feedback; normal feedback also updates the base learner with the checked outcome. `history` preserves pre-update predictions for inspection and currently grows without a bound. This remains a research API, not a public deployment component.

Run `python evaluate_adaptive_self.py` to reproduce the latest audit. After 40 feedback examples: 200/200 adaptive versus 0/200 longest-context predictions on deliberately misleading long contexts. All base contexts were known; evaluation prefixes differed from feedback prefixes and targets were excluded from feedback. The base learner was frozen to isolate the selector. Another context family retained 100/100 bindings. In an 80-step reversal test, accuracy was 0/10 in the first ten and 10/10 in the last ten (67/80 overall). Twelve unit checks pass, including earlier failure probes. The task was designed to benefit shorter-context selection, not sampled from general language. Final-token grouping can transfer incorrectly when two situations share a final token but require different selection strategies. The +5 and negation failures of the base mechanism are not solved by this work. No claim of improvements to the chatbot is made.
