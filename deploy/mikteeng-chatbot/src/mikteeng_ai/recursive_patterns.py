"""Bounded learned patterns of pattern-system outputs."""
import numpy as np
from .pattern_bank import LearnedPatternBank

class RecursivePatternSystem:
    def __init__(self, base, depth=2, seed=42, bounded=False):
        if type(depth) is not int or not 1<=depth<=3:raise ValueError('depth must be 1 to 3')
        self.base=base;self.depth=depth;self.seed=seed;self.levels=[];self.bounded=bounded
    @staticmethod
    def features(bank,x,anchor=None):
        # Describe the whole current pattern bank, not only its selected answer.
        experts=[np.asarray(m.predict(x)).reshape(len(x),-1) for m in bank.models]
        scores=bank.weights(x)
        chosen=bank.predict(x)
        return np.concatenate([x if anchor is None else anchor,*experts,scores,chosen],axis=1)
    def fit(self,x,y,groups):
        current=np.asarray(x);bank=self.base
        self.levels=[]
        for level in range(self.depth):
            current=self.features(bank,current,anchor=np.asarray(x) if getattr(self,"bounded",False) else None)
            bank=LearnedPatternBank(patterns=bank.patterns,seed=self.seed+level+1).fit(current,y,groups=groups)
            self.levels.append(bank)
        return self
    def predict(self,x,depth=None):
        depth=self.depth if depth is None else depth
        if type(depth) is not int or not 0<=depth<=self.depth:raise ValueError('invalid depth')
        bank=self.base;current=np.asarray(x)
        for level in self.levels[:depth]:
            current=self.features(bank,current,anchor=np.asarray(x) if getattr(self,"bounded",False) else None);bank=level
        return bank.predict(current)
    def weights(self,x):
        bank=self.base;current=np.asarray(x)
        for level in self.levels:
            current=self.features(bank,current,anchor=np.asarray(x) if getattr(self,"bounded",False) else None);bank=level
        return bank.weights(current)
