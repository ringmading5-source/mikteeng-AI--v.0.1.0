"""Train generic scorer on passive and negative examples; audit retention."""
import json
from pathlib import Path
from evidence_self import EvidenceSelf,RankedEvidenceSelf
from evaluate_evidence import dataset

def additions(names,objects):
    rows=[]
    for i,name in enumerate(names):
        for j,obj in enumerate(objects):
            q=f'Who holds the {obj}?';other=objects[(j+1)%len(objects)];actor=names[(i+1)%len(names)]
            for passive in [f'The {obj} is held by {name}.',f'The {obj} was held by {name}.']:
                rows.append((q,passive,name))
                rows.append((q,f'{actor} holds the {other}. '+passive,name))
            for neg in [f'{name} never holds the {obj}.',f'{name} does not hold the {obj}.']:
                rows.append((q,neg,None))
                rows.append((q,neg+f' {actor} holds the {other}.',None))
                rows.append((q,neg+f' {actor} holds the {obj}.',actor))
                rows.append((q,f'{actor} holds the {obj}. '+neg,actor))
    return rows

def audit(ai,rows):
    outputs=[]
    for q,o,a in rows:
        r=ai.predict(q,o);outputs.append({'question':q,'observations':o,'expected':a,'prediction':r['text'],'correct':r['text']==('' if a is None else a)})
    return {'correct':sum(r['correct'] for r in outputs),'total':len(outputs),'outputs':outputs}

def run():
    names=['Alice','Brian','Carol','David'];objects=['book','cup','bag','key']
    old=dataset(names,objects);new=additions(names,objects)
    heldnames=['Nia','Omar','Pia','Ravi'];heldobjects=['pencil','basket','coin','bottle']
    baseline=EvidenceSelf();retrained=EvidenceSelf();enhanced=EvidenceSelf(contextual_missing=True)
    for q,o,a in old:baseline.learn(q,o,a)
    for q,o,a in old+new:retrained.learn(q,o,a);enhanced.learn(q,o,a)
    ranked=RankedEvidenceSelf()
    for _ in range(30):
        for q,o,a in old+new:ranked.learn(q,o,a)
    checks={'retention':dataset(heldnames,heldobjects),'passive_and_negation':additions(heldnames,heldobjects),
      'fresh_entities':dataset(['Elin','Farah','Goran'],['rope','stone','hat'])+additions(['Elin','Farah','Goran'],['rope','stone','hat']),
      'untrained_grammar':[('Who holds the pencil?','Nia is not holding the pencil.',None),('Who holds the pencil?','The pencil belongs to Nia.',None),('Who holds the pencil?','Mary Jane holds the pencil.','Mary Jane')]}
    report={'training_original':len(old),'training_added':len(new),'scope':'Disjoint entity vocabulary, familiar trained grammatical forms. Same original test set used for retention; new passive/negative test shapes guided by known failures.', 'models':{}}
    for label,model in [('original',baseline),('more_data',retrained),('data_and_context_features',enhanced),('error_driven_ranking',ranked)]:
        report['models'][label]={k:audit(model,rows) for k,rows in checks.items()}
    root=Path(__file__).parent
    (root/'repaired_evidence_training.json').write_text(json.dumps([{'question':q,'observations':o,'answer':a} for q,o,a in old+new],indent=2))
    (root/'repaired_evidence_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'training':len(old+new),'models':{m:{k:{'correct':r['correct'],'total':r['total']} for k,r in checks.items()} for m,checks in report['models'].items()}},indent=2))
    return ranked,report
if __name__=='__main__':run()
