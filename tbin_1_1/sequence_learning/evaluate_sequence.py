import ast,json,time
from pathlib import Path
from sequence import SequenceLearner

def dataset():
    train=[];test=[]
    for rows,names in [(train,['Ana','Ben','Dan','Eve','Ian','Joy','Kim','Lee']),(test,['Ada','Bob','Mia','Sam'])]:
        for n in names:rows.append({'task':'conversation','prompt':f'My name is {n}. Say hello.','response':f'Hello, {n}!'})
    for rows,names in [(train,['add','sum','plus','total','combine','join']),(test,['merge','compute'])]:
        for n in names:rows.append({'task':'coding','prompt':f'Write Python function {n} to add a and b.','response':f'def {n}(a, b):\n    return a + b'})
    for rows,names in [(train,['subtract','minus','difference','remove','deduct','reduce']),(test,['decrease','less'])]:
        for n in names:rows.append({'task':'coding','prompt':f'Write Python function {n} to subtract b from a.','response':f'def {n}(a, b):\n    return a - b'})
    return train,test

def evaluate(m,rows):
    result=[]
    for r in rows:
        pred=m.generate(r['prompt']);syntax=None
        if r['task']=='coding':
            try:ast.parse(pred['text']);syntax=True
            except SyntaxError:syntax=False
        result.append({**r,**{'predicted':pred['text'],'ended':pred['ended'],'exact':pred['ended'] and pred['text']==r['response'],'python_syntax_valid':syntax}})
    return {'correct':sum(x['exact'] for x in result),'total':len(result),'examples':result}
if __name__=='__main__':
    start=time.perf_counter();train,test=dataset();m=SequenceLearner();before=evaluate(m,test)
    history=m.train(train);m.save('sequence_pilot.npz');after=evaluate(m,test);seen=evaluate(m,train)
    assert evaluate(SequenceLearner.load('sequence_pilot.npz'),test)==after
    assert history[-1]<history[0]
    report={'architecture':'tanh recurrent byte decoder, BPTT and Adam; optional TBIN companion',
      'parameters':sum(x.size for x in m.p.values()),'weight_bytes':sum(x.nbytes for x in m.p.values()),
      'epochs':len(history),'first_loss':history[0],'last_loss':history[-1],
      'untrained_heldout':before,'trained_seen':seen,'trained_heldout':after,'reload_identical':True,
      'seconds':time.perf_counter()-start,'limits':['20 synthetic training pairs, one seed',
      'no real speech or conversation dataset','syntax is not functional correctness; generated code was not executed',
      'separate from TBIN modality projections; no end-to-end multimodal generation']}
    Path('sequence_results.json').write_text(json.dumps(report,indent=2))
    for name,rows in [('train',train),('heldout',test)]:Path(name+'.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    print('Loss',history[0],history[-1],'seen',seen['correct'],seen['total'],'heldout',after['correct'],after['total'],flush=True)
