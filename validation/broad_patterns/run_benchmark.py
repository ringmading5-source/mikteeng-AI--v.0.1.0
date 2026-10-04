"""Broad bounded audit and same-architecture training of existing saved model."""
import argparse,json,sys,time,hashlib
from pathlib import Path

def passage(actor,crop):
    return [f'{actor} {v} the {crop}.' for v in ['selects','plants','waters','checks','harvests','cleans','stores']]

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.project.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'src'))
    import numpy as np
    from mikteeng_ai import MikteengAI
    from threadpoolctl import threadpool_limits
    checkpoints=['prediction_patterns.mkteeng','adaptive_pattern_bank.mkteeng','question_conditioned_predictions.mkteeng']
    hashes={n:hashlib.sha256((root/'models'/n).read_bytes()).hexdigest() for n in checkpoints}
    models={n:MikteengAI.load(root/'models'/n) for n in checkpoints}
    report={'scope':'Finite synthetic pattern families, not all possible patterns; expected numeric continuation follows the specified generating rule.','original_checkpoint_hashes':hashes,'numeric':{},'language':{}}
    families={
      'arithmetic':lambda t,k: k+3*t,
      'constant':lambda t,k:k,
      'doubling':lambda t,k:k*2**t,
      'tripling':lambda t,k:k*3**t,
      'halving':lambda t,k:k/2**t,
      'quadratic':lambda t,k:k+t*t,
      'cubic':lambda t,k:k+t**3,
      'alternating':lambda t,k:k+(-1)**t,
      'period_three':lambda t,k:k+t%3,
      'triangular':lambda t,k:k+t*(t+1)/2,
      'fibonacci':lambda t,k:k+[1,1,2,3,5,8,13,21,34][t],
      'alternating_growth':lambda t,k:k+(-1)**t*t}
    for filename in checkpoints[:2]:
        groups={}
        for name,f in families.items():
            cases=[]
            for k in range(2,12):
                observations=[f(t,k) for t in range(6)];expected=[f(t,k) for t in range(6,9)]
                try:
                    s=models[filename].build_prediction_self(observations)
                    actual=np.asarray(models[filename].respond_prediction_self(s,steps=3)['response']).ravel().tolist()
                    correct=bool(np.allclose(actual,expected,atol=.01,rtol=0))
                    cases.append({'observations':observations,'expected':expected,'actual':actual,'correct':correct})
                except Exception as e:cases.append({'observations':observations,'expected':expected,'correct':False,'error':str(e)})
            groups[name]={'correct':sum(c['correct'] for c in cases),'total':len(cases),'cases':cases}
        report['numeric'][filename]=groups
    ai=models[checkpoints[2]];actors=['alice','brian','carol','david','eva','frank','grace','henry'];crops=['beans','maize','rice','wheat','peas','millet','sorghum','lentils']
    train=[];tests=[]
    for i,actor in enumerate(actors):
        for j,crop in enumerate(crops):
            obs=passage(actor,crop);other=actors[(i+1)%len(actors)]
            rows=[('paraphrase',f'Name the person who waters the {crop}.',actor),
                  ('missing_relation',f'Who eats the {crop}?','insufficient evidence.'),
                  ('why_absent',f'Why does {actor} water the {crop}?','insufficient evidence.'),
                  ('negative_question',f'Who does not water the {crop}?','insufficient evidence.')]
            for kind,q,answer in rows:
                row={'family':kind,'observations':obs,'question':q,'answer':answer}
                (tests if (i+j)%4==0 else train).append(row)
            # Existing sentence vocabulary/grammar, independently changing bindings.
            mixed=obs[:3]+passage(other,crops[(j+1)%8])[3:]
            tests.extend([{'family':'changing_actor','observations':mixed,'question':f'Who waters the {crop}?','answer':actor},
                          {'family':'changing_object','observations':mixed,'question':f'Which crop does {other} harvests?','answer':crops[(j+1)%8]}])
            if (i+j)%4==0:
                tests.extend([{'family':'unseen_grammar','observations':[f'The {crop} are watered by {actor}.']+obs[1:],'question':f'Who waters the {crop}?','answer':actor},
                  {'family':'negated_observation','observations':[f'{actor} never waters the {crop}.']+obs[1:],'question':f'Who waters the {crop}?','answer':'insufficient evidence.'}])
    for name in ['nia','omar','pia','ravi']:
        tests.append({'family':'new_vocabulary','observations':passage(name,'pencil'),'question':'Who waters the pencil?','answer':name})
    # These answer templates are absent from feedback, probing wording transfer.
    for i in range(8):
        actor=actors[i];crop=crops[i];obs=passage(actor,crop)
        tests.extend([{'family':'fresh_wording','observations':obs,'question':f'Tell me who waters the {crop}.','answer':actor},
          {'family':'fresh_missing_wording','observations':obs,'question':f'Name the person who eats the {crop}.','answer':'insufficient evidence.'}])
    def evaluate(model):
        groups={}
        for row in tests:
            try:
                s=model.build_sentence_prediction_self(row['observations']);answer=model.ask(row['question'],predicted_self=s)['text']
                case=dict(row,actual=answer,correct=answer==row['answer'])
            except ValueError as e:case=dict(row,actual=None,correct=False,unsupported=True,error=str(e))
            groups.setdefault(row['family'],[]).append(case)
        return {k:{'correct':sum(x['correct'] for x in v),'total':len(v),'unsupported':sum(x.get('unsupported',False) for x in v),'cases':v} for k,v in groups.items()}
    report['language']['before']=evaluate(ai)
    original=json.loads((root/'data/question_conditioned_training.json').read_text())
    train_rows=[{k:r[k] for k in ['observations','question','answer']} for r in train]
    # Verify exact full question/context tuples do not overlap new feedback and evaluation.
    keys=lambda rows:{(tuple(r['observations']),r['question']) for r in rows}
    assert not keys(train_rows)&keys(tests)
    (out/'training_additions.json').write_text(json.dumps(train_rows,indent=2))
    (out/'evaluation_cases.json').write_text(json.dumps(tests,indent=2))
    print('Before training:',json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='cases'} for k,v in report['language']['before'].items()}),flush=True)
    started=time.time()
    with threadpool_limits(limits=2):ai.train_question_conditioned_predictions(original+train_rows)
    ai.save(out/'broader_questions_candidate.mkteeng')
    report['training']={'original_examples_replayed':len(original),'new_examples':len(train_rows),'new_feedback_eval_overlap':0,'seconds':round(time.time()-started,3),'architecture_changed':False,'component_updated':'question_conditioned_predictor only; observation/first/higher predictors unchanged'}
    report['language']['after']=evaluate(ai)
    report['original_checkpoints_unchanged']=all(hashlib.sha256((root/'models'/n).read_bytes()).hexdigest()==hashes[n] for n in checkpoints)
    (out/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'numeric':{m:{k:{kk:vv for kk,vv in v.items() if kk!='cases'} for k,v in groups.items()} for m,groups in report['numeric'].items()},'after':{k:{kk:vv for kk,vv in v.items() if kk!='cases'} for k,v in report['language']['after'].items()},'training':report['training'],'original_checkpoints_unchanged':report['original_checkpoints_unchanged']},indent=2))
if __name__=='__main__':main()
