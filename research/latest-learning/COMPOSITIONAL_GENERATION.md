# Internal predictive composition in original Mikteeng

This extends the original MikteengAI instance and reuses its learned StateTransitionLearner. That original component already supported bounded planning. The new interface adds an immutable versioned Self, normalized vector snapshots of predicted states, ranked candidate paths, explicit constraints, prediction budgets and an inspectable internal trace. The original sentence checkpoint is loaded before adding transitions and remains in the delivered checkpoint.

Mechanism:
1. Structured observation and goal initialize Self.
2. Existing learned transition heads predict outcomes for candidate actions.
3. Predicted states become internal observations for subsequent predictions.
4. Generic goal equality and forbidden-state constraints evaluate candidates.
5. Bounded beam search composes an action sequence and emits it as the response.

No stored complete answers are selected. The response is a generated structured sequence, not natural-language prose. Actions are supplied as a finite vocabulary; the model does not invent new primitive actions. The transition heads are the original tree learners; no new pretrained model or neural architecture was added. Scalar-to-vector hashing is for Self inspection; learned transition heads operate on their original features. State vectors can collide; exact state dictionaries govern identity. Scores are not calibrated truth probabilities.

Test: 48 observed one-step transitions for three binary state fields. All 56 distinct start/goal pairs were solved, including 32 requiring multiple steps, with shortest paths in this tiny test. Complete solution paths were absent from fitting. All individual transitions/state values were familiar. This is compositional planning on a toy domain, not broad intelligence or human-like thought. Constraint rejection, prediction-budget stopping, stale Self rejection, normalized state vectors and checkpoint reload passed. The integrated original sentence model retained 156/192.

The same transition learner can receive observed feedback through learn_feedback. Feedback retrains that learner and invalidates earlier Self snapshots. Search does not access the teacher/environment simulator. The teacher in the test script only generates supervised examples and independently verifies responses. No external actions are executed.

Example:
```python
from mikteeng_ai import MikteengAI
ai = MikteengAI.load('models/compositional_generation.mkteeng', trusted=True)
self_state = ai.build_deliberation_self(
    {'a':0, 'b':0, 'c':0},
    {'a':1, 'b':1, 'c':1},
    [{'field':k, 'value':v} for k in ['a','b','c'] for v in [0,1]],
)
response = ai.generate_from_self(self_state, max_depth=3, max_predictions=256)
print(response['response'])
```

CPU-only. max_depth is capped at 16; max_predictions at 4096; beam width at 64; actions at 64. Search can fail because of its bounds or wrong predicted transitions. Intermediate candidates are not facts. Free-language question interpretation, generating spoken answers and grounding semantic goals remain unfinished. Do not claim the successful toy result demonstrates those abilities.
