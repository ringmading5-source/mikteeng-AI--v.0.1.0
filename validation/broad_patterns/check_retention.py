"""Additional read-only retention and boundary tests; no fitting."""
import argparse,sys,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.project.resolve();out=a.output.resolve();sys.path.insert(0,str(root/'src'))
    from mikteeng_ai import MikteengAI
    models={'before':MikteengAI.load(root/'models/question_conditioned_predictions.mkteeng'),'after':MikteengAI.load(out/'broader_questions_candidate.mkteeng')}
    report=json.loads((out/'results.json').read_text());rows=[]
    for obs in json.loads((root/'data/passage_patterns_test.json').read_text()):
        actor=obs[0].split()[0];crop=obs[0].split()[-1].rstrip('.')
        for i,text in enumerate(obs):
            verb=text.split()[1]
            if sum(t.split()[1]==verb for t in obs)>1:continue
            for q,y in [(f'Who {verb} the {crop}?',actor),(f'Which crop does {actor} {verb}?',crop)]+([(f'What happens after {actor} {verb} the {crop}?',obs[i+1])] if i<len(obs)-1 else []):rows.append({'observations':obs,'question':q,'answer':y})
    extra=[]
    names=['alice','brian','carol','david'];crops=['beans','maize','rice','wheat']
    for i,actor in enumerate(names):
        other=names[(i+1)%4]
        for crop in crops:
            obs=[f'{actor} waters the {crop}.',f'{other} waters the {crop}.',f'{actor} checks the {crop}.']
            extra.append({'family':'multiple_actors','observations':obs,'question':f'Name everyone who waters the {crop}.','answer':f'{actor} and {other}'})
            extra.append({'family':'possession','observations':[f'{actor} owns the {crop}.',f'{other} waters the {crop}.',f'{actor} checks the {crop}.'],'question':f'Who owns the {crop}?','answer':actor})
            extra.append({'family':'explicit_cause','observations':[f'{actor} waters the {crop} because the {crop} need water.',f'{actor} checks the {crop}.',f'{actor} harvests the {crop}.'],'question':f'Why does {actor} water the {crop}?','answer':f'the {crop} need water'})
    def evalrows(model,rows):
        cases=[]
        for row in rows:
            try:
                result=model.ask(row['question'],predicted_self=model.build_sentence_prediction_self(row['observations']))['text'];cases.append(dict(row,actual=result,correct=result==row['answer']))
            except ValueError as e:cases.append(dict(row,actual=None,correct=False,unsupported=True,error=str(e)))
        return {'correct':sum(x['correct'] for x in cases),'total':len(cases),'unsupported':sum(x.get('unsupported',False) for x in cases),'cases':cases}
    report['original_544_retention']={k:evalrows(model,rows) for k,model in models.items()}
    for label,model in models.items():
        for kind in ['multiple_actors','possession','explicit_cause']:report['language'][label][kind]=evalrows(model,[r for r in extra if r['family']==kind])
    report['language_case_count']=sum(g['total'] for g in report['language']['before'].values())
    report['external_teacher_api_connected']=False
    (out/'additional_evaluation_cases.json').write_text(json.dumps(extra,indent=2));(out/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'retention':{k:{kk:vv for kk,vv in v.items() if kk!='cases'} for k,v in report['original_544_retention'].items()},'extra':{label:{kind:{kk:vv for kk,vv in g.items() if kk!='cases'} for kind,g in groups.items() if kind in ['multiple_actors','possession','explicit_cause']} for label,groups in report['language'].items()}},indent=2))
if __name__=='__main__':main()
