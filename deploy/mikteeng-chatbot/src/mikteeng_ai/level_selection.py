"""Learn recursive-depth selection from separate observed-outcome feedback."""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

class LearnedLevelSelector:
    def __init__(self, system, seed=42):
        self.system=system;self.depth=system.depth
        self.base=system.base;self.levels=system.levels;self.seed=seed
    def features(self,x):
        x=np.asarray(x);outputs=[self.system.predict(x,depth=i) for i in range(self.depth+1)]
        return np.concatenate([x,*outputs,*[a-outputs[0] for a in outputs[1:]]],axis=1),outputs
    def fit(self,x,y):
        features,outputs=self.features(x);y=np.asarray(y)
        errors=np.stack([np.mean((v-y)**2,axis=1) for v in outputs],axis=1)
        targets=errors.argmin(axis=1)
        self.gate=ExtraTreesClassifier(n_estimators=150,min_samples_leaf=2,random_state=self.seed).fit(features,targets)
        self.training_level_counts=np.bincount(targets,minlength=self.depth+1).tolist()
        return self
    def choose(self,x):return self.gate.predict(self.features(x)[0])
    def predict(self,x):
        features,outputs=self.features(x);choice=self.gate.predict(features)
        return np.stack(outputs)[choice,np.arange(len(features))]
    def weights(self,x):return self.system.weights(x)
