"""Latent regression experts: pattern identities inferred from prediction error."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesClassifier
from .prediction_patterns import PredictionPatternLearner, sequence

class LearnedPatternBank:
    def __init__(self, patterns=3, seed=42):
        if patterns<2:raise ValueError('at least two patterns required')
        self.patterns=patterns;self.seed=seed
    @staticmethod
    def context(x):
        x=np.asarray(x)
        return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    def fit(self,x,y,groups=None):
        x=np.asarray(x);y=np.asarray(y);best=None
        if len(x)<self.patterns*3:raise ValueError('insufficient pattern observations')
        rng=np.random.default_rng(self.seed)
        groups=np.arange(len(x)) if groups is None else np.asarray(groups)
        unique=np.unique(groups)
        # No pattern names, arithmetic rules, or class labels enter training.
        for restart in range(12):
            assignment=rng.integers(self.patterns,size=len(unique))
            labels=assignment[np.searchsorted(unique,groups)]
            for _ in range(35):
                models=[]
                for j in range(self.patterns):
                    ids=labels==j
                    if ids.sum()<2:ids[rng.choice(len(x),min(5,len(x)),replace=False)]=True
                    models.append(Ridge(alpha=1e-6).fit(x[ids],y[ids]))
                errors=np.stack([np.mean((np.asarray(m.predict(x)).reshape(y.shape)-y)**2,axis=1) for m in models],axis=1)
                group_error=np.array([errors[groups==g].mean(axis=0) for g in unique])
                updated=group_error.argmin(axis=1)[np.searchsorted(unique,groups)]
                if np.array_equal(updated,labels):break
                labels=updated
            loss=float(errors.min(axis=1).mean())
            if best is None or loss<best[0]:best=(loss,models,labels.copy())
        self.loss,self.models,labels=best
        self.gate=ExtraTreesClassifier(n_estimators=100,min_samples_leaf=2,random_state=self.seed).fit(self.context(x),labels)
        self.counts=np.bincount(labels,minlength=self.patterns).tolist()
        from .vector_nodes import PatternNodeSpace
        self.vector_nodes=PatternNodeSpace.from_assignments(x,labels,max_nodes=self.patterns)
        return self
    def predict(self,x):
        x=np.asarray(x);chosen=self.gate.predict(self.context(x))
        # Hard routing avoids averaging incompatible continuations.
        predictions=np.stack([np.asarray(m.predict(x)).reshape(len(x),-1) for m in self.models])
        return predictions[chosen,np.arange(len(x))]
    def weights(self,x):
        return self.gate.predict_proba(self.context(x))

class AdaptivePredictionPatternLearner(PredictionPatternLearner):
    def __init__(self, patterns=3, seed=42, recursive_depth=0, upward_depth=None, **kwargs):
        if upward_depth is not None:
            if recursive_depth!=0:raise ValueError('use upward_depth or recursive_depth, not both')
            recursive_depth=upward_depth
        self.bounded_upward=upward_depth is not None
        if type(recursive_depth) is not int or not 0<=recursive_depth<=3:raise ValueError('recursive_depth must be 0 to 3')
        super().__init__(**kwargs);self.patterns=patterns;self.seed=seed;self.recursive_depth=recursive_depth
    def fit(self, observations, *, self_observations):
        primary=[sequence(v) for v in observations];meta=[sequence(v) for v in self_observations]
        super().fit(primary,self_observations=meta)
        x=[];y=[];groups=[];w=self.observation_window;k=self.prediction_window
        for group,v in enumerate(meta):
            p=self._predictions(v)
            for j in range(k-1,len(p)-1):
                x.append(p[j-k+1:j+1].ravel());y.append(v[w+j]);groups.append(group)
        self.higher=LearnedPatternBank(self.patterns,self.seed).fit(x,y,groups=groups)
        if self.recursive_depth:
            from .recursive_patterns import RecursivePatternSystem
            self.higher=RecursivePatternSystem(self.higher,self.recursive_depth,self.seed,bounded=self.bounded_upward).fit(x,y,groups)
        return self
    def inspect(self,s):
        if s.version!=self.version:raise ValueError('stale self')
        bank=self.higher
        if hasattr(bank,'levels'):
            scores=bank.weights([s.pattern])[0]
            final=bank.levels[-1]
            return {'learned_pattern_ids':final.gate.classes_.tolist(),'routing_scores':scores.tolist(),
                    'selected_pattern':int(final.gate.classes_[scores.argmax()]),
                    'recursive_depth':bank.depth,'selected_depth':int(bank.choose([s.pattern])[0]) if hasattr(bank,'choose') else bank.depth,'scores_calibrated':False}
        scores=bank.weights([s.pattern])[0]
        return {'learned_pattern_ids':bank.gate.classes_.tolist(),'routing_scores':scores.tolist(),
                'selected_pattern':int(bank.gate.predict(bank.context([s.pattern]))[0]),
                'recursive_depth':0,'scores_calibrated':False}
