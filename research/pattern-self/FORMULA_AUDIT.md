# Audit against Mading's original formula

This audit corrects the scope of earlier progress claims. It is a code/document inspection, not a new training experiment. No chatbot or learner is changed by this audit.

## Intended mechanism

Observations -> learned independent patterns -> predictions -> self learns patterns across predictions -> self + question + current observations -> response -> checked outcomes adapt the patterns and their selection.

The intended scales include characters, words, sentences, relationships, states/actions, patterns across these systems, and meta-patterns. Pattern separation should preserve different actors, subjects and actions rather than average them into one indistinguishable pattern. Representations and learning algorithms must be specified, but domain answer rules should not replace discovery from observations. This formula is a conceptual specification; it does not uniquely determine one executable update algorithm.

## Which work is closest to the core?

The earlier PredictionPatternLearner in deploy/mikteeng-chatbot/src/mikteeng_ai/prediction_patterns.py is closer to the central predictions -> patterns of predictions requirement than the recent standalone prototype.

It fits a first Ridge predictor on observation windows. It freezes that predictor, forms sequences of its predictions on a second group of observations, and fits a higher Ridge predictor from prediction windows to checked observed targets. Its formulas are approximately:

    P_t = F_theta(O_(t-w:t))
    refined_P_t = G_phi(P_(t-k+1:t+1))
    train G_phi against the corresponding actual observation O_t

The higher target is the corresponding observed outcome, not necessarily a separately shifted future event. This is prediction-based refinement. Future rollout appends predictions as hypotheses; it does not verify them automatically.

build_self stores a versioned snapshot of observations, predictions and the final prediction window. That SNAPSHOT is not itself a learner that discovers patterns. The trained first/higher models hold learned weights. Calling both the snapshot and the weights 'self' without explanation blurred this distinction.

SentencePatternLearner connects sentence vectors to that learner. AdaptivePredictionPatternLearner adds separate learned experts and a gate. RecursivePatternSystem trains additional banks using previous bank outputs, expert predictions and routing scores. LearnedLevelSelector learns which depth fits checked targets. These are real partial implementations of the formula, not just class names, but bounded and based on existing statistical algorithms.

## Stage-by-stage audit

| Intended requirement | Actual implementation | Assessment |
|---|---|---|
| Observations become token representations | Existing text/audio/image encoders; recent lossless text/code tokenizer | Partial. Modality-specific encodings are designed; no shared sensory meaning demonstrated. |
| Character -> word -> sentence hierarchy | Separate character and role models; optional features in generation; sentence positional vectors | Partial connections only. No single learned bottom-up hierarchy discovers all groupings. |
| Independent patterns stay distinguishable | Earlier regression experts and learned gate; recent context-specific counts | Partial. Expert number and grouping assumptions are configured. Wrong routing/binding remains possible. |
| Patterns predict new observations | Earlier sequence predictors and newer transition/numeric components | Demonstrated on bounded tasks. Unrestricted unseen-pattern learning is unproven. |
| Self learns patterns across predictions | Earlier higher predictor consumes prediction windows and bank outputs | Closest existing core. Recent reliability statistics are a narrower meta signal, not the same full mechanism. |
| Question patterns condition self | Earlier PatternQuestionReader/QuestionConditionedPredictor receive self snapshots | Partial. Supervised questions, designed features/output heads; no demonstrated necessity of prediction history in all tasks. |
| Response derived from unified self | Earlier API routes to specific trained heads; recent UnifiedSelf wraps different tables/learners | Not complete. Common API is not one shared learned state. |
| Relationships/states/actions emerge from observation patterns | Earlier annotated role/transition heads; recent labeled evidence/span selection | Supervised support experiments. Autonomous category/grouping discovery has not been demonstrated. |
| Checked outcomes adapt predictive structure | Earlier reader retraining and depth selection; recent count updates and learned reliability | Partial. Updates are local to particular components, not coordinated across the hierarchy. |
| Recursion over pattern systems is adaptive | Earlier bounded learned levels; recent finite differences | Earlier learned levels are partial. Finite differences transform observations, not prediction systems. |
| Multimodal unification | Separate experimental encoders; recent wrapper rejects audio/images | Missing at the core. No cross-modal shared prediction-self demonstrated. |
| Self predicts tool actions | No learned executable-tool component | Missing. Infrastructure for calling a tool is not a learned action policy. |

## What the recent prototype actually does

PatternSelf: empirical next-token counts keyed by suffix contexts. It learns transition frequencies; longest-suffix fallback is programmed.

AdaptivePredictionSelf: learns recent success/failure evidence for (last token, context length). This learns which proposals succeed, but does not learn rich sequences of prediction patterns. History is stored; storing history alone does not learn it.

RecursiveNumericSelf: learns local transitions at a designed finite-difference hierarchy and learns reliability per depth. Useful controlled extrapolation, but the transform family and integration are programmed. It is not the general meta-pattern mechanism.

EvidenceSelf, RankedEvidenceSelf, SpanEvidenceSelf: supervised candidate scoring using designed question/position/boundary features. They test grounding, corrections and binding. Adding missing-evidence and span candidates is architectural engineering, not evidence that self discovered those concepts.

UnifiedSelf: dispatches between separate observation tables, response transitions, numeric predictors and evidence scorers. Its text/code observe tables do not automatically train the question-response table; question-answer examples and labels are separate. Evidence answers bypass the prediction-history learner. Numeric observations are recorded, then analyzed locally when explicitly supplied to respond. All these are interface integration, not full formula integration.

These experiments should be retained as baselines and capability probes. They should not be described as the completed core. They do not replace the earlier prediction-pattern implementation.

## Existing evidence that limits our claims

Inspected saved earlier experiment reports (not rerun here):

- Recursive sentence levels: base 156/192 correct; each added level 120/192. More recursion hurt this test.
- Separate depth-selection audit: base 72/96, learned selector 71/96. No improvement established there.
- Pattern-question reader: 384/384 with prediction history and 384/384 without it. That test does not demonstrate prediction history is necessary.

The recent prototype's numeric and supervised evidence successes are narrow. Negation/span repairs were developed using known task failures. A higher score on those tasks does not prove the original recursive architecture. A direct standard-library implementation is still a machine-learning algorithm: replacing sklearn does not by itself make it more faithful. Faithfulness depends on the information flow and learned update, not the library name.

## Next authorized core work, following this audit

Do not add another domain-specific repair as the next milestone. Start from the earlier prediction-window mechanism and specify a single trace contract that retains observations, distinct candidate predictions, provenance/context, question and checked outcome. Use that trace as actual input to a learned prediction-pattern stage and its response stage; do not merely log it.

Keep both earlier and recent components. Compare the same response path with prediction-pattern self enabled, disabled, and shuffled, while preserving question/observation access. On disjoint sequences, any claimed contribution of self must be supported by those comparisons. Test independent actors/actions, contradictions and outcome changes; check that feedback actually changes the relevant shared predictive state and retains old capabilities. If a new stage hurts, record it rather than silently replacing the test.

One core acceptance milestone: a traceable learned path from observations to distinct predictions, from those predictions to a learned higher representation, and from that representation plus question to response, with feedback updating the same path. This still permits explicit architecture choices. It does not promise that one test establishes universal intelligence or that supervised labels can be eliminated immediately.

Deployment remains separate. The public chatbot configuration does not currently load this research prototype. No completed live deployment was verified during this work.
