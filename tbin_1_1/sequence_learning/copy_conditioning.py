"""Learn a single copied span per response from paired demonstrations.
Task-independent template induction, not neural attention or general reasoning.
"""
from dataclasses import dataclass,asdict
from itertools import combinations
import json
from pathlib import Path

@dataclass(frozen=True)
class CopyRule:
    input_prefix:str
    input_suffix:str
    output_prefix:str
    output_suffix:str
    support:int=0
    def apply(self,prompt,max_slot=128):
        a,b=self.input_prefix,self.input_suffix
        if not prompt.startswith(a) or not prompt.endswith(b):return None
        end=len(prompt)-len(b) if b else len(prompt)
        slot=prompt[len(a):end]
        if end<=len(a) or len(slot)>max_slot or '\n' in slot or '\r' in slot:return None
        return self.output_prefix+slot+self.output_suffix

def differences(a,b):
    """Common prefix/suffix, and the two differing spans (no overlap)."""
    p=0
    while p<min(len(a),len(b)) and a[p]==b[p]:p+=1
    s=0
    while s<min(len(a)-p,len(b)-p) and a[len(a)-s-1]==b[len(b)-s-1]:s+=1
    return a[:p],a[len(a)-s:] if s else '',a[p:len(a)-s if s else len(a)],b[p:len(b)-s if s else len(b)]

class CopyConditioner:
    def __init__(self):self.rules=[]
    def fit(self,rows,max_examples=256,max_rules=256):
        if not 2<=len(rows)<=max_examples:raise ValueError('Pilot requires 2..256 examples')
        for r in rows:
            if not isinstance(r['prompt'],str) or not isinstance(r['response'],str):raise ValueError('Text required')
            if len(r['prompt'])>512 or len(r['response'])>512:raise ValueError('Example too long')
        candidates=set()
        for x,y in combinations(rows,2):
            ip,is_,iv,jv=differences(x['prompt'],y['prompt'])
            op,os,ov,pv=differences(x['response'],y['response'])
            if iv and jv and iv!=jv and (iv,jv)==(ov,pv) and len(ip)+len(is_)>=3:
                candidates.add(CopyRule(ip,is_,op,os))
        kept=[]
        for rule in candidates:
            matched=[(r,rule.apply(r['prompt'])) for r in rows]
            matched=[(r,o) for r,o in matched if o is not None]
            if any(o!=r['response'] for r,o in matched):continue
            support=len({r['prompt'] for r,o in matched})
            if support>=2:kept.append(CopyRule(rule.input_prefix,rule.input_suffix,rule.output_prefix,rule.output_suffix,support))
        # Prefer rules supported by more distinct demonstrations. No task strings in the algorithm.
        kept.sort(key=lambda r:(-r.support,len(r.input_prefix)+len(r.input_suffix),r.input_prefix,r.input_suffix,r.output_prefix,r.output_suffix))
        if len(kept)>max_rules:raise ValueError('Too many rules for this pilot; divide tasks')
        self.rules=kept;return {'rules':len(kept),'examples':len(rows)}
    def generate(self,prompt):
        if not isinstance(prompt,str) or len(prompt)>512:raise ValueError('Invalid prompt')
        hits=[(r,r.apply(prompt)) for r in self.rules];hits=[(r,o) for r,o in hits if o is not None]
        answers={o for r,o in hits}
        if not answers:return {'text':None,'route':'abstain','reason':'unmatched_template'}
        if len(answers)>1:return {'text':None,'route':'abstain','reason':'conflicting_templates'}
        return {'text':next(iter(answers)),'route':'learned_copy','support':max(r.support for r,o in hits)}
    def save(self,path):Path(path).write_text(json.dumps({'version':1,'rules':[asdict(r) for r in self.rules]},ensure_ascii=False,indent=2))
    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text())
        if d.get('version')!=1 or len(d['rules'])>256:raise ValueError('Unsupported model')
        m=cls();m.rules=[CopyRule(**r) for r in d['rules']];return m

class CopyConditionedLearner:
    """Wrap a SequenceLearner. Fallback is opt-in because neural results were poor."""
    def __init__(self,neural=None):self.copy=CopyConditioner();self.neural=neural
    def fit_copy(self,rows):return self.copy.fit(rows)
    def generate(self,prompt,allow_neural_fallback=False):
        result=self.copy.generate(prompt)
        if result['route']=='abstain' and allow_neural_fallback and self.neural is not None:
            return {**self.neural.generate(prompt),'route':'neural_unverified'}
        return result
