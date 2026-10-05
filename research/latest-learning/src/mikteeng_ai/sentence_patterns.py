"""Sentence vectors through the shared prediction-pattern learner."""
import re
import numpy as np
from .prediction_patterns import PredictionPatternLearner

class SentencePatternLearner:
    def __init__(self, adaptive=False, **kwargs):
        from .pattern_bank import AdaptivePredictionPatternLearner
        self.learner=(AdaptivePredictionPatternLearner if adaptive or "upward_depth" in kwargs else PredictionPatternLearner)(**kwargs)
    @staticmethod
    def tokens(text):
        if not isinstance(text,str) or not text.strip():raise ValueError('sentence must be nonempty text')
        return re.findall(r"\w+|[^\w\s]",text.lower())
    def fit(self, observations, *, self_observations):
        primary=[list(s) for s in observations];meta=[list(s) for s in self_observations]
        rows=[self.tokens(t) for s in primary+meta for t in s]
        if not rows:raise ValueError('empty observations')
        self.vocabulary=['<end>']+sorted(set(t for r in rows for t in r))
        self.index={t:i for i,t in enumerate(self.vocabulary)}
        self.width=max(map(len,rows))+1
        self.learner.fit([self.encode(s) for s in primary],self_observations=[self.encode(s) for s in meta])
        return self
    def encode(self, sentences):
        rows=[]
        for text in sentences:
            ts=self.tokens(text)
            if len(ts)>=self.width:raise ValueError('sentence exceeds trained width')
            if any(t not in self.index for t in ts):raise ValueError('word outside trained vocabulary')
            a=np.zeros((self.width,len(self.vocabulary)))
            for j in range(self.width):a[j,self.index[ts[j]] if j<len(ts) else 0]=1
            rows.append(a.ravel())
        return np.array(rows)
    def decode(self, vector):
        words=[]
        for i in np.asarray(vector).reshape(self.width,-1).argmax(axis=1):
            if i==0:break
            words.append(self.vocabulary[i])
        # Only punctuation spacing; no sentence template or semantic rule.
        return re.sub(r'\s+([.,!?;:])',r'\1',' '.join(words))
    def build_self(self, sentences):return self.learner.build_self(self.encode(sentences))
    def respond(self, learned_self, *, steps=1):
        r=self.learner.respond(learned_self,steps=steps)
        return {'sentences':[self.decode(v) for v in r['response']], 'vectors':r['response'],'confidence':None}
