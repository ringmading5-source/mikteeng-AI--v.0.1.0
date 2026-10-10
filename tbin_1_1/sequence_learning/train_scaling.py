"""Fixed-architecture synthetic data scaling and continued training.
Uses fresh test identifiers; no model selection on test results.
"""
import json,random,time
from pathlib import Path
from sequence import SequenceLearner
from evaluate_sequence import dataset,evaluate
from copy_conditioning import CopyConditioner

def build_data():
    train,_=dataset();rng=random.Random(421);used=set();extra=[];test=[]
    def identifier():
        while True:
            value=''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(rng.randint(3,8)))
            if value not in used:used.add(value);return value
    def row(i,name):
        if i%3==0:return {'task':'conversation','prompt':f'My name is {name}. Say hello.','response':f'Hello, {name}!'}
        op='add a and b' if i%3==1 else 'subtract b from a';symbol='+' if i%3==1 else '-'
        return {'task':'coding','prompt':f'Write Python function {name} to {op}.','response':f'def {name}(a, b):\n    return a {symbol} b'}
    # Deterministic synthetic variation in values; this does not add new language templates.
    for i in range(220):extra.append(row(i,identifier()))
    train+=extra
    for i in range(30):test.append(row(i,identifier()))
    assert not {r['prompt'] for r in train}&{r['prompt'] for r in test}
    return train,test

def measure(m,train,test):
    seen=evaluate(m,train);held=evaluate(m,test)
    return {'training_correct':seen['correct'],'training_total':seen['total'],
            'test_correct':held['correct'],'test_total':held['total'],'test_examples':held['examples']}

if __name__=='__main__':
    out=Path('scaling_run');out.mkdir(exist_ok=True);train,test=build_data();runs=[];start=time.perf_counter()
    for n in (20,80,240):
        m=SequenceLearner(seed=7);t=time.perf_counter();history=m.train(train[:n],epochs=120)
        result={'mode':'from_scratch','examples':n,'epochs':120,'seed':7,'first_loss':history[0],
                'last_loss':history[-1],**measure(m,train[:n],test),'seconds':time.perf_counter()-t}
        m.save(out/f'neural_{n}.npz');runs.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='test_examples'}),flush=True)
    # Continue the actual earlier checkpoint, not only new independently initialized models.
    checkpoint=Path('sequence_pilot.npz')
    if not checkpoint.exists():
        previous=SequenceLearner();previous.train(train[:20]);previous.save(checkpoint)
    m=SequenceLearner.load(checkpoint);before=measure(m,train,test);t=time.perf_counter();h=m.train(train,epochs=120)
    m.save(out/'continued_model.npz');result={'mode':'continued_prior_pilot','examples':240,'additional_epochs':120,
       'optimizer_reset':True,'first_loss':h[0],'last_loss':h[-1],**measure(m,train,test),'seconds':time.perf_counter()-t}
    restored=SequenceLearner.load(out/'continued_model.npz');assert evaluate(restored,test)==evaluate(m,test)
    runs.append(result);print(json.dumps({k:v for k,v in result.items() if k!='test_examples'}),flush=True)
    c=CopyConditioner();info=c.fit(train);copyresults=[{'prompt':r['prompt'],'expected':r['response'],**c.generate(r['prompt'])} for r in test]
    c.save(out/'copy_model.json')
    unknown=['Please greet Mading.','Write a sorting algorithm.','Translate this Bor sentence.','What is 19 times 23?','My name is Akol. Say goodbye.','Write Python function product to multiply a and b.']
    report={'runs':runs,'continued_before':before,'copy_rules':info['rules'],
      'copy_correct':sum(r['text']==r['expected'] for r in copyresults),'copy_total':len(copyresults),'copy_examples':copyresults,
      'out_of_template_abstentions':sum(c.generate(q)['route']=='abstain' for q in unknown),'out_of_template_total':len(unknown),
      'seconds':time.perf_counter()-start,'checkpoint_reload_identical':True,
      'limits':['synthetic data: varied names/identifiers, only three templates','one initialization seed',
      'fixed epochs means larger datasets also receive more optimization steps','no real speech or conversation corpus',
      'exact answer accuracy, not functional execution of generated code','neural architecture unchanged; copy results separate']}
    (out/'results.json').write_text(json.dumps(report,indent=2))
    for name,rows in [('train',train),('test',test)]:
        (out/f'{name}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    print('Copy:',report['copy_correct'],'/',report['copy_total'],'Total seconds:',report['seconds'],flush=True)
