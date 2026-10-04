"""Connect existing role head to existing question-self path, with controls."""
import argparse,json,sys,hashlib,time,copy
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--broad',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.project.resolve();broad=a.broad.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root/'src'))
    from mikteeng_ai import MikteengAI
    from mikteeng_ai.question_prediction import QuestionConditionedPredictor
    from threadpoolctl import threadpool_limits
    source=root/'models/question_conditioned_predictions.mkteeng';role_source=root/'models/biology_physics.mkteeng'
    hashes={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,role_source]}
    base=MikteengAI.load(source);roles=MikteengAI.load(role_source).roles
    original=json.loads((root/'data/question_conditioned_training.json').read_text());extra=json.loads((broad/'training_additions.json').read_text());same=original+extra
    names=['alice','brian','carol','david','eva','frank','grace','henry'];crops=['beans','maize','rice','wheat','peas','millet','sorghum','lentils'];verbs=['selects','plants','waters','checks','harvests','cleans','stores']
    def passage(actor,crop):return [f'{actor} {v} the {crop}.' for v in verbs]
    mixed=[]
    for i,actor in enumerate(names):
        for j,crop in enumerate(crops):
            if (i+j)%4==0:continue
            for shift in [2,3]:
                other=names[(i+shift)%8];othercrop=crops[(j+shift)%8];obs=passage(actor,crop)[:3]+passage(other,othercrop)[3:]
                mixed.extend([{'observations':obs,'question':f'Who waters the {crop}?','answer':actor},{'observations':obs,'question':f'Which crop does {other} harvests?','answer':othercrop}])
    tests=json.loads((broad/'evaluation_cases.json').read_text())+json.loads((broad/'additional_evaluation_cases.json').read_text())
    key=lambda rows:{(tuple(r['observations']),r['question']) for r in rows}
    assert not key(mixed)&key(tests)
    fresh=[]
    for i,actor in enumerate(names):
        for j,crop in enumerate(crops):
            if (i+j)%4!=0:continue
            other=names[(i+4)%8];othercrop=crops[(j+4)%8];obs=passage(actor,crop)[:3]+passage(other,othercrop)[3:]
            fresh.append({'family':'fresh_actor_binding','observations':obs,'question':f'Who waters the {crop}?','answer':actor})
            fresh.append({'family':'fresh_object_binding','observations':obs,'question':f'Which crop does {other} harvests?','answer':othercrop})
    assert not key(same+mixed)&key(fresh)
    tests+=fresh
    def evalmodel(model):
        groups={}
        for row in tests:
            try:
                s=model.build_sentence_prediction_self(row['observations']);result=model.ask(row['question'],predicted_self=s)['text'];case=dict(row,actual=result,correct=result==row['answer'])
            except ValueError as exc:case=dict(row,actual=None,correct=False,unsupported=True,error=str(exc))
            groups.setdefault(row['family'],[]).append(case)
        return {k:{'correct':sum(r['correct'] for r in rows),'total':len(rows),'unsupported':sum(r.get('unsupported',False) for r in rows),'cases':rows} for k,rows in groups.items()}
    report={'source_checkpoint_hashes':hashes,'role_head_retrained':False,'first_and_higher_predictors_retrained':False,'original_training_rows':len(original),'prior_additions':len(extra),'mixed_training_rows':len(mixed),'mixed_eval_overlap':0,'experiments':{}}
    candidate=MikteengAI.load(broad/'broader_questions_candidate.mkteeng');report['experiments']['prediction_summary_baseline']=evalmodel(candidate)
    start=time.time()
    with threadpool_limits(limits=2):
        for label,rows,connected in [('ordered_observations_control',same,False),('roles_connected_same_training',same,True),('ordered_observations_mixed_training',same+mixed,False),('roles_connected_mixed_training',same+mixed,True)]:
            ai=MikteengAI.load(source)
            if connected:ai.train_subject_conditioned_questions(rows,role_model=roles)
            else:ai.question_conditioned_predictor=QuestionConditionedPredictor(ai.sentence_pattern_model,self_source='hybrid').fit(rows)
            report['experiments'][label]=evalmodel(ai)
            print(label,json.dumps({k:{kk:vv for kk,vv in r.items() if kk!='cases'} for k,r in report['experiments'][label].items()}),flush=True)
            if label=='roles_connected_mixed_training':
                ai.save(out/'subject_connected_candidate.mkteeng');candidate=ai
    report['seconds']=round(time.time()-start,3)
    # Independent load exercises actual dispatch and stored role component.
    loaded=MikteengAI.load(out/'subject_connected_candidate.mkteeng')
    report['roundtrip_same']=evalmodel(loaded)==report['experiments']['roles_connected_mixed_training']
    report['role_examples']={s:roles.predictions(base.sentence_pattern_model.tokens(s)) for s in ['alice waters the beans.','brian harvests the maize.']}
    report['original_models_unchanged']=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name] for p in [source,role_source])
    (out/'mixed_training_additions.json').write_text(json.dumps(mixed,indent=2));(out/'fresh_evaluation_cases.json').write_text(json.dumps(fresh,indent=2));(out/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'seconds':report['seconds'],'roundtrip_same':report['roundtrip_same'],'original_models_unchanged':report['original_models_unchanged']}),flush=True)
if __name__=='__main__':main()
