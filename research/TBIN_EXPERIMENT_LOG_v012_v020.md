# TBIN v0.12-v0.20 experiment log

Status: experimental research. Most results below are synthetic. They are not
evidence of state-of-the-art performance, general intelligence, semantic
understanding, or real-language sample efficiency.

## v0.12 — upward invariant compression
Stable invariant + bounded residual bank. Synthetic best observed accuracy:
31.9% at 160 exposures/concept in the experiment. 1,920 observations were
represented with 36 retained residual observations (53.3:1 observation/residual
ratio). This is not a general compression benchmark.

## v0.13 — relational distances
First-order relational fingerprints added little: best same-run improvement was
about +0.3 percentage points. Conclusion: distance-to-concepts is not sufficient
for meaning.

## v0.14-v0.16 — transfer controls
v0.14 was invalid as evidence because nearest-source baseline also achieved
100%. v0.15 removed that shortcut and transferred a fixed synthetic relation.
v0.16 selected among a predefined operator library; therefore it demonstrated
operator selection, not discovery.

## v0.17 — operator induction
Removed the predefined global operator library. Within the explicit hypothesis
family y_i ~= sign*x_j+b, transfer rose from 14.5% with one example to 98.3%
with three and 100% with five or more in that synthetic setup. This is few-shot
structural operator induction inside a programmed hypothesis family, not
unrestricted rule discovery.

## v0.18 — composition crystallization
Recurring primitive transformation sequences were compressed into reusable
composition states. Frequency alone also created extra states, showing that
recurrence is not truth.

## v0.19-v0.20 — evidence gating
Crystallization was constrained by recurrence, predictive consistency,
compression gain, and cross-context stability. v0.19 was not discriminative.
v0.20 introduced adversarial recurrent false patterns; in that synthetic setup,
recurrence-only accepted all eight false patterns while evidence gating retained
six genuine patterns and accepted zero false patterns across 200 trials.

## First real phone-audio test
A user-provided ~54.6 s phone recording was analyzed without transcript labels.
The exploratory segmentation produced 62 candidate acoustic segments, 37
structural clusters, and six clusters meeting the recurrence criterion. These
are candidate recurring acoustic structures only. They have not been validated
as words, phonemes, meanings, or speaker-independent concepts.

## Next falsification
Use controlled real recordings with repeated words, multiple speakers, held-out
utterances, and ideally a completely held-out speaker. Measure same-unit versus
different-unit structural distance, cluster purity, false crystallization,
sample efficiency, CPU time, and memory growth.
