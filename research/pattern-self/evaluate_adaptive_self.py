"""Reproducible controlled context selection and distribution-shift audit."""
import json
from pathlib import Path
from pattern_self import PatternSelf,AdaptivePredictionSelf

def run():
    base=PatternSelf(2)
    for i in range(240):
        base.observe([f'rare{i}','signal','wrong'])
        for j in range(3):base.observe([f'common{i}_{j}','signal','right'])
    # A different context family requires keeping its two-token binding.
    for i in range(100):base.observe([f'actor{i}','holds',f'object{i}'])
    self=AdaptivePredictionSelf(base)
    for i in range(40):self.feedback([f'rare{i}','signal'],'right',update_patterns=False)
    held=[([f'rare{i}','signal'],'right') for i in range(40,240)]
    raw=sum(base.predict(c)['prediction']==y for c,y in held)
    adaptive=sum(self.predict(c)['prediction']==y for c,y in held)
    intact=[([f'actor{i}','holds'],f'object{i}') for i in range(100)]
    preserve=sum(self.predict(c)['prediction']==y for c,y in intact)
    shift=[]
    for _ in range(80):
        shift.append(int(self.predict(['rare0','signal'])['prediction']=='wrong'))
        self.feedback(['rare0','signal'],'wrong',update_patterns=False)
    report={'selection_test':{'baseline_correct':raw,'adaptive_correct':adaptive,'total':200,
      'feedback_examples':40,'base_patterns_frozen':True,'scope':'Known base contexts; checked reliability transfers to other prefixes with the same final token. Targets in evaluation are not used for training.'},
      'other_context_bindings_retained':{'correct':preserve,'total':100},
      'changed_outcome_prequential':{'first_10_correct':sum(shift[:10]),'last_10_correct':sum(shift[-10:]),'total_correct':sum(shift),'total':80},
      'limitations':['Designed suffix contexts and final-token reliability grouping','Fixed exponential retention 0.95','No unseen transformation discovery','No learned negation or semantic role discovery','Reliability scores are not calibrated correctness probabilities','Single meta level, no arbitrary recursive pattern learning']}
    Path(__file__).with_name('adaptive_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':run()
