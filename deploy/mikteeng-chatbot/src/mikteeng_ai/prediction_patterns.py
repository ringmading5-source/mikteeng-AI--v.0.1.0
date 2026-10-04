"""Observation -> prediction -> learned prediction-pattern self experiment.
No arithmetic operation or semantic role is selected by an answer rule.
"""
from dataclasses import dataclass
from uuid import uuid4
import numpy as np
from sklearn.linear_model import Ridge


def sequence(values):
    a=np.asarray(values,dtype=float)
    if a.ndim==1:a=a[:,None]
    if a.ndim!=2 or not len(a) or not np.isfinite(a).all():
        raise ValueError('observations must be a finite, nonempty sequence of vectors')
    return a

@dataclass(frozen=True)
class PredictionPatternSelf:
    version: str
    observations: tuple
    predictions: tuple
    pattern: tuple

class PredictionPatternLearner:
    """Two learned temporal levels with sequence-disjoint fitting data.

    First level predicts observations. Second learns temporal relationships
    among first-level predictions, grounded by future observed targets.
    """
    def __init__(self, observation_window=1, prediction_window=3, alpha=1e-6):
        if observation_window<1 or prediction_window<1:raise ValueError('windows must be positive')
        self.observation_window=observation_window
        self.prediction_window=prediction_window
        self.alpha=alpha
        self.version=None

    def fit(self, observations, *, self_observations):
        primary=[sequence(v) for v in observations]
        meta=[sequence(v) for v in self_observations]
        if not primary or not meta:raise ValueError('both training groups are required')
        d=primary[0].shape[1];w=self.observation_window;k=self.prediction_window
        if any(v.shape[1]!=d or len(v)<w+k+1 for v in primary+meta):
            raise ValueError('inconsistent dimensions or sequences too short')
        # Separate sequence groups: self sees predictions from a frozen first learner.
        x=[];y=[]
        for v in primary:
            for t in range(w,len(v)):
                x.append(v[t-w:t].ravel());y.append(v[t])
        self.first=Ridge(alpha=self.alpha).fit(x,y)
        x=[];y=[]
        for v in meta:
            p=self._predictions(v)
            for j in range(k-1,len(p)-1):
                x.append(p[j-k+1:j+1].ravel())
                # p[j] predicts observation w+j, target is w+j.
                y.append(v[w+j])
        self.higher=Ridge(alpha=self.alpha).fit(x,y)
        self.dimensions=d;self.version=uuid4().hex
        return self

    def _predictions(self,v):
        w=self.observation_window
        # Includes prediction of the next unobserved element.
        return self.first.predict(np.array([v[t-w:t].ravel() for t in range(w,len(v)+1)])).reshape(-1,v.shape[1])

    def build_self(self, observations):
        if self.version is None:raise RuntimeError('train first')
        v=sequence(observations)
        if v.shape[1]!=self.dimensions or len(v)<self.observation_window+self.prediction_window-1:
            raise ValueError('wrong vector dimensions or insufficient context')
        p=self._predictions(v);pattern=p[-self.prediction_window:].ravel()
        return PredictionPatternSelf(self.version,tuple(map(tuple,v)),tuple(map(tuple,p)),tuple(pattern))

    def respond(self, learned_self, *, steps=1):
        if learned_self.version!=self.version:raise ValueError('self belongs to a different training version')
        if not isinstance(steps,int) or steps<1:raise ValueError('steps must be a positive integer')
        # Question here is a requested future horizon; no natural-language parser.
        v=np.array(learned_self.observations);out=[]
        for _ in range(steps):
            s=self.build_self(v)
            answer=np.asarray(self.higher.predict([s.pattern])).reshape(-1)
            out.append(answer);v=np.vstack([v,answer])
        return {'response':np.array(out).tolist(),'question':{'steps':steps},
                'mechanism':'learned patterns of predictions','confidence':None}
