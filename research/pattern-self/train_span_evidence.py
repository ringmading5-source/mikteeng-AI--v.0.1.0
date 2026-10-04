import json
from pathlib import Path
from evidence_self import SpanEvidenceSelf
from evaluate_evidence import dataset
from repair_evidence import additions,audit

def examples(names,objects):return dataset(names,objects)+additions(names,objects)
def run():
    train=examples(['Alice','Brian','Carol','David'],['book','cup','bag','key'])
    train+=examples(['Mary Jane','John Paul','Anna Maria Lee','James Arthur Cole'],['book','cup','bag','key'])
    ai=SpanEvidenceSelf()
    for _ in range(30):
        for q,o,a in train:ai.learn(q,o,a)
    checks={'prior_single_token':examples(['Nia','Omar','Pia','Ravi'],['pencil','basket','coin','bottle']),
            'unseen_multiword_names':examples(['Lina Rose','Omar Ali','Pia Sera Moon','Ravi Arun Dev'],['pencil','basket','coin','bottle']),
            'fresh_mixed_names':examples(['Elin','Farah Noor','Goran Ayo Kim'],['rope','stone','hat']),
            'outside_span_bound':[('Who holds the pencil?','Nia Rose Ayo Moon holds the pencil.','Nia Rose Ayo Moon')]}
    report={'training_rows':len(train),'passes':30,'max_span':3,'scope':'Unseen names/objects; familiar grammar, contiguous answers up to 3 words. Designed span enumeration and boundary features.','tests':{k:audit(ai,rows) for k,rows in checks.items()}}
    root=Path(__file__).parent
    (root/'span_evidence_training.json').write_text(json.dumps([{'question':q,'observations':o,'answer':a} for q,o,a in train],indent=2))
    (root/'span_evidence_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'training_rows':len(train),'tests':{k:{'correct':v['correct'],'total':v['total']} for k,v in report['tests'].items()}},indent=2))
    return ai,report
if __name__=='__main__':run()
