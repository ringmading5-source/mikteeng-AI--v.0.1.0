import json,time
from pathlib import Path
from mikteeng_ai.character_prediction import CharacterPredictionModel
r=Path(__file__).resolve().parents[1]
roots=['walk','jump','look','talk','help','work','play','rain','call','clean','paint','cook']
endings=['s','ed','ing']
held=[root+endings[i%3] for i,root in enumerate(roots)]
training=roots+[root+ending for root in roots for ending in endings if root+ending not in held]
assert not set(held)&set(training)
def generate(model,seed,limit=20):
 current='^'+seed;trace=[]
 for _ in range(limit):
  prediction=model.predict_next(current);character=prediction['character'];trace.append(character)
  if character=='$':return current[1:],True,trace
  current+=character
 return current[1:],False,trace
reports=[]
for context in [16,3]:
 start=time.perf_counter();model=CharacterPredictionModel(context=context).fit(['^'+word+'$' for word in training]);fit=time.perf_counter()-start
 rows=[]
 for i,root in enumerate(roots):
  ending=endings[i%3];expected=root+ending
  # 's' is the full one-character suffix, so exclude it from partial-word novelty claims.
  seed=root+ending[0] if len(ending)>1 else root
  actual,stopped,trace=generate(model,seed)
  rows.append({'seed':seed,'expected':expected,'actual':actual,'stopped':stopped,'correct':actual==expected,'complete_word_absent_from_training':expected not in training,'suffix_characters_supplied':1 if len(ending)>1 else 0,'predicted_characters':trace})
 controls=[]
 for word in training:
  if len(word)>3:
   seed=word[:-2];actual,stopped,_=generate(model,seed)
   controls.append({'seed':seed,'expected':word,'actual':actual,'correct':actual==word})
 unprompted=[]
 for root in roots:
  actual,stopped,trace=generate(model,root)
  unprompted.append({'seed':root,'actual':actual,'stopped':stopped,'in_training':actual in training,'valid_test_formation':actual in {a+b for a in roots for b in ['',*endings]}})
 unseeded,unseeded_stopped,_=generate(model,'')
 reports.append({'start_only_generation':unseeded,'start_only_stopped':unseeded_stopped,'start_only_in_training':unseeded in training,'context':context,'fit_seconds':fit,'training_words':len(training),'heldout_correct':sum(x['correct'] for x in rows),'heldout_total':len(rows),'multi_character_suffix_correct':sum(x['correct'] and x['suffix_characters_supplied']==1 for x in rows),'multi_character_suffix_total':sum(x['suffix_characters_supplied']==1 for x in rows),'familiar_completion_correct':sum(x['correct'] for x in controls),'familiar_completion_total':len(controls),'heldout_cases':rows,'controls':controls,'root_only_generations':unprompted})
report={'mechanism':'existing CharacterPredictionModel, unchanged algorithm','training_words':training,'heldout_words':held,'boundary_markers':'^ start, $ stop, learned from examples','generation':'greedy next-character predictions; no full-word candidate lookup','test_not_used_for_fitting':True,'evaluations':reports,'limits':['English regular endings only; not African-language evidence','roots and ending fragments seen; full test words withheld','partial seed supplies one suffix character for ed/ing','s suffix cases are ambiguous with root-only seeds','character head is separate from prediction-history Self, though in original package','no meaning, pronunciation or creativity evaluation']}
(r/'experiments/word_formation_test.json').write_text(json.dumps(report,indent=2))
print(json.dumps([{k:v for k,v in rep.items() if k not in ['heldout_cases','controls','root_only_generations']} for rep in reports],indent=2))
for rep in reports:
 print('CONTEXT',rep['context']);print(json.dumps(rep['heldout_cases'],indent=2))
