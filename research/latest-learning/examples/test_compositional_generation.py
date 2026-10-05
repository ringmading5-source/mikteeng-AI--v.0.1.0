import itertools,json
from pathlib import Path
from mikteeng_ai import MikteengAI
root=Path(__file__).resolve().parents[1]
keys=['a','b','c'];states=[dict(zip(keys,bits)) for bits in itertools.product([0,1],repeat=3)]
actions=[{'field':k,'value':v} for k in keys for v in [0,1]]
# Teacher simulation is used only to construct examples and check responses.
# The runtime predictor does not call this oracle.
def environment(state,action):return {**state,action['field']:action['value']}
training=[{'state':s,'action':a,'next_state':environment(s,a)} for s in states for a in actions]
ai=MikteengAI.load_trusted(root/"models/passage_patterns_16.mkteeng").train_transitions(training)
rows=[]
for state in states:
 for goal in states:
  if state==goal:continue
  self_state=ai.build_deliberation_self(state,goal,actions)
  answer=ai.generate_from_self(self_state,max_depth=3,beam_width=16,max_predictions=256,min_score=.8)
  actual=dict(state)
  for action in answer['response']:actual=environment(actual,action)
  rows.append({'state':state,'goal':goal,'response':answer['response'],'status':answer['status'],'correct':actual==goal,
               'steps':len(answer['response']),'minimum_steps':sum(state[k]!=goal[k] for k in keys),'predictions_evaluated':answer['predictions_evaluated']})
# Constraint and exhausted-budget behavior.
state={'a':0,'b':0,'c':0};goal={'a':1,'b':1,'c':1}
blocked=ai.generate_from_self(ai.build_deliberation_self(state,goal,actions,constraints=[{'a':1}]),max_depth=3)
assert blocked['status']!='generated_path'
budget=ai.generate_from_self(ai.build_deliberation_self(state,goal,actions),max_predictions=1)
assert budget['status']=='prediction_budget'
report={'training':'48 single-step transitions only; no complete paths or answers provided','test_cases':len(rows),'correct':sum(r['correct'] for r in rows),'multi_step_cases':sum(r['minimum_steps']>1 for r in rows),'multi_step_correct':sum(r['correct'] and r['minimum_steps']>1 for r in rows),'shortest_correct':sum(r['correct'] and r['steps']==r['minimum_steps'] for r in rows),'constraint_test':blocked['status'],'budget_test':budget['status'],'cases':rows,'limits':['structured state/goal/action inputs, not free-language question understanding','all individual state values and one-step transitions familiar','generic beam search and goal equality are programmed','transition mappings are the existing learned original heads','not evidence of human-like thought or creative language generation']}
ai.save(root/'models/compositional_generation.mkteeng')
loaded=MikteengAI.load_trusted(root/'models/compositional_generation.mkteeng')
assert loaded.generate_from_self(loaded.build_deliberation_self(state,goal,actions))['response']==ai.generate_from_self(ai.build_deliberation_self(state,goal,actions))['response']
report['save_reload']='passed'
(root/'experiments/compositional_generation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
