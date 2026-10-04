"""Read-only inference audit of existing Mikteeng checkpoints; no retraining."""
import argparse,json,sys,hashlib,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.project.resolve();sys.path.insert(0,str(root/'src'))
    import numpy as np
    from mikteeng_ai import MikteengAI
    paths=[root/'models'/n for n in ['prediction_patterns.mkteeng','adaptive_pattern_bank.mkteeng','sentence_prediction_patterns.mkteeng','question_prediction_self.mkteeng','question_conditioned_predictions.mkteeng']]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    start=time.time();report={'checkpoints_sha256':hashes,'retrained':False,'new_research_prototype_used':False}
    numeric=MikteengAI.load(paths[0]);cases=[]
    for a in [50,110,-80]:
        for delta in [2,7,-5,.5]:
            obs=[a+i*delta for i in range(4)];expected=[a+(4+i)*delta for i in range(3)]
            s=numeric.build_prediction_self(obs);before=s.observations
            actual=np.array(numeric.respond_prediction_self(s,steps=3)['response']).ravel().tolist()
            cases.append({'observations':obs,'expected':expected,'actual':actual,'correct':bool(np.allclose(actual,expected,atol=1e-5)),'snapshot_unchanged':s.observations==before})
    report['numeric']={'correct':sum(c['correct'] for c in cases),'total':len(cases),'cases':cases}
    bank=MikteengAI.load(paths[1]);cases=[]
    for obs,expected in [([4,6,8],[10,12,14]),([4,8,16],[32,64,128]),([4,12,36],[108,324,972])]:
        s=bank.build_prediction_self(obs);actual=np.array(bank.respond_prediction_self(s,steps=3)['response']).ravel().tolist()
        cases.append({'observations':obs,'expected':expected,'actual':actual,'correct':bool(np.allclose(actual,expected,atol=.01)),'routing':bank.prediction_pattern_model.inspect(s)})
    report['distinct_patterns']={'correct':sum(c['correct'] for c in cases),'total':len(cases),'cases':cases}
    sentence=MikteengAI.load(paths[2]);d=json.loads((root/'data/sentence_pattern_training.json').read_text());verbs=['plants','waters','checks','harvests'];cases=[]
    for name,crop in d['held_out_pairs']:
        for offset in range(4):
            obs=[f'{name} {verbs[(offset+i)%4]} {crop}.' for i in range(3)]
            expected=[f'{name} {verbs[(offset+i+3)%4]} {crop}.' for i in range(3)]
            s=sentence.build_sentence_prediction_self(obs);actual=sentence.continue_sentence_patterns(s,steps=3)['sentences']
            cases.append({'observations':obs,'expected':expected,'actual':actual,'correct_sentences':sum(a==b for a,b in zip(actual,expected)),'prediction_window_matches_self':bool(np.allclose(s.pattern,np.array(s.predictions)[-3:].ravel()))})
    report['sentence_continuation']={'correct':sum(c['correct_sentences'] for c in cases),'total':3*len(cases),'contexts':len(cases),'all_snapshots_derived_from_predictions':all(c['prediction_window_matches_self'] for c in cases),'cases':cases}
    passages=json.loads((root/'data/passage_patterns_test.json').read_text())
    reader=MikteengAI.load(paths[3]);generator=MikteengAI.load(paths[4]);reading=[];generating=[]
    for passage in passages:
        rs=reader.build_sentence_prediction_self(passage);gs=generator.build_sentence_prediction_self(passage)
        actor=passage[0].split()[0];crop=passage[0].split()[-1].rstrip('.')
        for i,text in enumerate(passage):
            verb=text.split()[1]
            if sum(t.split()[1]==verb for t in passage)>1:continue
            for question in [f'Who {verb} the {crop}?',f'What does {actor} {verb}?']:
                answer=reader.ask(question,predicted_self=rs)
                reading.append({'question':question,'expected':text,'actual':answer['text'],'correct':answer['text']==text})
            questions=[(f'Who {verb} the {crop}?',actor),(f'Which crop does {actor} {verb}?',crop)]
            if i<len(passage)-1:questions.append((f'What happens after {actor} {verb} the {crop}?',passage[i+1]))
            for question,expected in questions:
                answer=generator.ask(question,predicted_self=gs)
                generating.append({'question':question,'expected':expected,'actual':answer['text'],'correct':answer['text']==expected})
    report['question_evidence']={'correct':sum(c['correct'] for c in reading),'total':len(reading),'cases':reading}
    report['question_generation']={'correct':sum(c['correct'] for c in generating),'total':len(generating),'cases':generating}
    first=passages[0];snapshot=generator.build_sentence_prediction_self(first)
    report['unsupported_probes']=[{'question':q,'actual':generator.ask(q,predicted_self=snapshot)['text'],'expected':'No justified answer from these observations'} for q in ['Who eats the beans?','Why does alice water the beans?']]
    try:sentence.build_sentence_prediction_self(['zebra eats leaves.']*3)
    except ValueError as exc:report['unknown_vocabulary']={'behavior':'rejected','message':str(exc)}
    report['checkpoint_files_unchanged']=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name] for p in paths)
    report['seconds']=round(time.time()-start,3)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:({kk:vv for kk,vv in v.items() if kk!='cases'} if isinstance(v,dict) else v) for k,v in report.items()},indent=2))
if __name__=='__main__':main()
