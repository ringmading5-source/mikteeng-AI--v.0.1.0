"""Count-based supervised candidate scoring; no role-specific answer branches."""
import re
from collections import Counter
from math import log

class EvidenceSelf:
    def __init__(self):
        self.counts=[Counter(),Counter()];self.totals=[0,0]
    def views(self,question,observations):
        q=re.findall(r'\w+',question.lower())
        sentences=[re.findall(r'\w+',s) for s in re.split(r'[.!?]+',observations) if s.strip()]
        allwords={w.lower() for s in sentences for w in s}
        global_features={f'global:q{j}:{int(w in allwords)}' for j,w in enumerate(q)}
        yield None,global_features|{'kind:missing'}|{'missing:'+f for f in global_features}
        for sentence in sentences:
            lower=[w.lower() for w in sentence]
            for i,word in enumerate(sentence):
                f={'kind:token',f'position:{min(i,8)}'}|global_features
                for j,qword in enumerate(q):
                    indices=[k for k,w in enumerate(lower) if w==qword]
                    f.add(f'local:q{j}:{int(bool(indices))}')
                    if indices:
                        distance=min(indices,key=lambda k:abs(k-i))-i
                        f.add(f'distance:q{j}:{max(-8,min(8,distance))}')
                for offset in [-2,-1,0,1,2]:
                    if 0<=i+offset<len(lower):f.add(f'neighbor:{offset}:{lower[i+offset]}')
                f |= {'token:'+feature for feature in global_features}
                yield word,f
    def learn(self,question,observations,answer):
        views=list(self.views(question,observations))
        if answer is not None and not any(w is not None and w.lower()==answer.lower() for w,_ in views):
            raise ValueError('Answer must be an observed single token, or None')
        for word,features in views:
            correct=(word is None and answer is None) or (word is not None and answer is not None and word.lower()==answer.lower())
            label=int(correct);self.totals[label]+=1;self.counts[label].update(features)
        return self
    def score(self,features):
        # Present-feature likelihood ratio with Laplace smoothing. Unknown
        # categorical values contribute equally to both classes and are ignored.
        known=self.counts[0].keys()|self.counts[1].keys()
        result=log((self.totals[1]+1)/(self.totals[0]+1))
        for f in features & known:
            result+=log((self.counts[1][f]+1)/(self.totals[1]+2))-log((self.counts[0][f]+1)/(self.totals[0]+2))
        return result
    def predict(self,question,observations):
        if not self.totals[1]:return {'status':'untrained','text':'','candidates':[]}
        candidates=[{'token':word,'score':self.score(features)} for word,features in self.views(question,observations)]
        best=max(candidates,key=lambda c:c['score'])
        return {'status':'insufficient_evidence' if best['token'] is None else 'experimental_answer',
                'text':'' if best['token'] is None else best['token'],'candidates':candidates}
