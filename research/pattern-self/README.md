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

## Recursive numeric observations

`RecursiveNumericSelf` now learns local transitions within the supplied numeric observations at several finite-difference depths. No +5, square, or cube answer rule is installed. The DIFFERENCE OPERATOR and the maximum depth (default 3) are explicitly designed. This is recursion over transformed observations, not the unfinished general recursion over patterns of predictions.

    D^(0) = observed numbers
    D^(l+1)_i = D^(l)_(i+1) - D^(l)_i

At each available level, a fresh PatternSelf observes that level and predicts its next change. Reconstruct a candidate next number by adding that predicted value back to the last value at each earlier level, deepest first. Select a candidate using local empirical frequency multiplied by its checked level-reliability score. The multiplication and shallow tie preference are fixed design decisions. Feedback compares all candidates with the actual outcome and updates level success/failure evidence with the same retained-score formula. Prediction does not feed generated values into training. Local transition tables are rebuilt from each input; checked level reliability persists in the object.

```python
from pattern_self import RecursiveNumericSelf
ai = RecursiveNumericSelf()
print(ai.predict([100,105,110,115])['prediction']) # 120
print(ai.predict([1,4,9,16,25])['prediction'])     # 36
ai.feedback([1,4,9,16,25],36)
```

Run `python evaluate_recursive_numeric.py`. After 18 checked calibration cases with coefficients 1,2,3, held-out coefficients 5 through 14 and offsets 100 through 109 scored 100/100 for each of degrees 1,2,3. +5 can also be inferred with no calibration: 100,105,110,115 -> 120. Geometric 2,4,8,16,32 still failed (58 instead of 64 after calibration). All 18 unit checks pass, including failure probes. These are curated noiseless arithmetic families favored by the chosen representation. Finite evidence permits many possible continuations; success does not establish the unique correct rule. No improved negation, text semantics, tool use, learned transformation discovery, noise robustness or universal generalization is claimed. This is separate from the unchanged deployment.

## One callable experimental interface

`unified_self.py` provides `Tokenizer` and `UnifiedSelf`, bringing this research prototype's observation, response, numeric and checked-feedback mechanisms under one API. Text/code transition tables stay separate. Question-answer transitions use an additional response table with non-colliding boundary markers. Numeric processing calls the recursive numeric component. Task routing is explicitly chosen by the caller, not predicted from natural-language intent. This wrapper does not jointly train these learners, does not merge earlier sklearn experiments, and does not change the deployment. It is importable from this research folder, not installed as a top-level mikteeng_ai API.

```python
from unified_self import UnifiedSelf, Tokenizer
self = UnifiedSelf()
self.observe('hello world', modality='text')
self.train_responses([
    {'question':'Who holds the book?', 'answer':'Alice.'},
    {'question':'Who holds the cup?', 'answer':'Brian.'},
])
print(self.respond('Who holds the book?')['text'])
self.feedback('Who holds the book?', actual='Brian.')
print(self.respond(observations=[100,105,110,115],
                   task='numeric_next', modality='numeric')['prediction'])
self.save('my_self.pkl')
```

Tokenizer supports character and word levels. Word tokens retain whitespace and punctuation, allowing exact reconstruction of text/code. They are strings, not hardcoded semantic categories or learned token-ID embeddings. Audio and image input raise an explicit unsupported-modality error. Sentence/relationship grouping and cross-modality grounding are not added.

`observe` updates text/code transition counts. For numeric observations it validates and records the sequence; supply that sequence to `respond` for local pattern discovery. Observing a raw passage alone does NOT teach the response table how to answer questions. `train_responses` receives question/answer examples with optional observations. `feedback` compares proposals against checked target tokens with teacher-forced correct prefixes and updates response counts and reliability. Numeric feedback updates level reliability. Generating an answer does not train the model on it.

Checkpoint save/load uses pickle: never load untrusted files. Histories and context counts are not bounded for production. Responses include traces; frequencies and selector scores are not calibrated certainty. Supported operations are observe, train_responses, respond, feedback, save, load; these do not execute external tools.

All 24 unit checks pass. The two trained question forms produced Alice and Brian respectively; five checked corrections changed the first to Brian. Numeric +5 gave 120. Tokenizers reconstructed code exactly; text/code continuation tables and save/load worked. These are small functional/integration checks, not new generalization benchmarks. The unsupported notebook question returned Alice despite lacking evidence. A paraphrase also returned Alice but that alone cannot distinguish learned reasoning from short-suffix fallback. `unified_results.json` records these probes. Reliable abstention and evidence binding remain unresolved.
