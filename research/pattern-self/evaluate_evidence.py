import json
from pathlib import Path
from evidence_self import EvidenceSelf

def dataset(names,objects):
    rows=[]
    for i,name in enumerate(names):
        for j,obj in enumerate(objects):
            other=objects[(j+1)%len(objects)];actor=names[(i+1)%len(names)]
            q=f'Who holds the {obj}?'
            rows.extend([(q,f'{name} holds the {obj}.',name),
                (q,f'{actor} holds the {other}. {name} holds the {obj}.',name),
                (q,f'{name} holds the {obj}. {actor} holds the {other}.',name),
                (q,f'{name} holds the {other}.',None),
                (q,f'The {obj} is on the table.',None)])
    return rows

def run():
    train=dataset(['Alice','Brian','Carol','David'],['book','cup','bag','key'])
    ai=EvidenceSelf()
    for q,o,a in train:ai.learn(q,o,a)
    tests=dataset(['Nia','Omar','Pia','Ravi'],['pencil','basket','coin','bottle'])
    results=[]
    for q,o,a in tests:
        r=ai.predict(q,o);expected='' if a is None else a
        results.append({'question':q,'observations':o,'answer':a,'prediction':r['text'],'status':r['status'],'correct':r['text']==expected})
    probes=[]
    for q,o,a in [('Who holds the pencil?','The pencil is held by Nia.','Nia'),('Who carries the pencil?','Nia carries the pencil.','Nia'),('Who holds the pencil?','Nia does not hold the pencil.',None),('Who holds the pencil?','Nia never holds the pencil.',None)]:
        r=ai.predict(q,o);probes.append({'question':q,'observations':o,'answer':a,'prediction':r['text'],'status':r['status'],'correct':r['text']==('' if a is None else a)})
    report={'training_rows':len(train),'test_rows':len(tests),'correct':sum(r['correct'] for r in results),'missing_correct':sum(r['correct'] for r in results if r['answer'] is None),'missing_total':sum(r['answer'] is None for r in results),'scope':'All test names and object words are absent from training; grammar and question shape are familiar. Single-token supervised answer selection.','probes':probes,'results':results}
    Path(__file__).with_name('evidence_results.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
    return ai,report
if __name__=='__main__':run()
