"""Count-based supervised candidate scoring; no role-specific answer branches."""
import re
from collections import Counter
from math import log

class EvidenceSelf:
    def __init__(self, contextual_missing=False):
        self.contextual_missing=contextual_missing
        self.counts=[Counter(),Counter()];self.totals=[0,0]
    def views(self,question,observations):
        q=re.findall(r'\w+',question.lower())
        sentences=[re.findall(r'\w+',s) for s in re.split(r'[.!?]+',observations) if s.strip()]
        allwords={w.lower() for s in sentences for w in s}
        global_features={f'global:q{j}:{int(w in allwords)}' for j,w in enumerate(q)}
        missing=global_features|{'kind:missing'}|{'missing:'+f for f in global_features}
        if self.contextual_missing:
            for sentence in sentences:
                lower=[w.lower() for w in sentence]
                mask=''.join(str(int(w in lower)) for w in q)
                missing |= {f'missing:sentence:{mask}:word:{w}' for w in lower}
        yield None,missing
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

class RankedEvidenceSelf(EvidenceSelf):
    """Structured perceptron ranking, implemented with sparse Python counts."""
    def __init__(self):
        super().__init__(contextual_missing=True)
        self.weights=Counter();self.updates=0
    def score(self,features):return sum(self.weights[f] for f in features)
    def learn(self,question,observations,answer):
        views=list(self.views(question,observations))
        correct=[(word,f) for word,f in views if (word is None and answer is None) or (word is not None and answer is not None and word.lower()==answer.lower())]
        if not correct:raise ValueError('Answer must be an observed single token, or None')
        best=max(views,key=lambda v:self.score(v[1]));target=max(correct,key=lambda v:self.score(v[1]))
        if best[0]!=target[0]:
            self.weights.update(target[1]);self.weights.subtract(best[1]);self.updates+=1
        self.totals[1]+=1
        return self

class SpanEvidenceSelf(RankedEvidenceSelf):
    """Learn contiguous word-span selection with a bounded candidate window."""
    def __init__(self,max_span=3):
        if not isinstance(max_span,int) or max_span<1:raise ValueError('Positive span bound required')
        super().__init__();self.max_span=max_span
    def views(self,question,observations):
        original=iter(super().views(question,observations))
        yield next(original)
        sentences=[re.findall(r'\w+',s) for s in re.split(r'[.!?]+',observations) if s.strip()]
        for sentence in sentences:
            token_features=[next(original)[1] for _ in sentence]
            for start in range(len(sentence)):
                for length in range(1,min(self.max_span,len(sentence)-start)+1):
                    end=start+length-1
                    features={'span:length:'+str(length)}
                    features|={'start:'+f for f in token_features[start] if not f.startswith('neighbor:')}
                    features|={'end:'+f for f in token_features[end] if not f.startswith('neighbor:')}
                    features.add('span:sentence_start:'+str(int(start==0)))
                    features.add('span:sentence_end:'+str(int(end==len(sentence)-1)))
                    for offset in [1,2]:
                        if end+offset<len(sentence):features.add(f'after:{offset}:{sentence[end+offset].lower()}')
                        if start-offset>=0:features.add(f'before:{offset}:{sentence[start-offset].lower()}')
                    yield ' '.join(sentence[start:end+1]),features
