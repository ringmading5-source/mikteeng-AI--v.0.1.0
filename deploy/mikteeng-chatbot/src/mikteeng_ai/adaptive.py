"""Experimental learned representation with no semantic-role schema.

A denoising bottleneck reconstructs observed text features. A separate learned
readout selects supporting observation sentences. Output is evidence copying,
not open-ended language generation. Numerical scores are uncalibrated.
"""
from dataclasses import dataclass
import re,uuid
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

@dataclass(frozen=True)
class AdaptiveSelf:
    identifier:str
    version:str
    observations:tuple
    sentence_vectors:tuple
    pooled_vector:tuple

class AdaptiveRepresentation:
    def __init__(self,dimensions=32,max_features=768,seed=42):
        if type(dimensions)!=int or dimensions<2:raise ValueError('dimensions must be >=2')
        if type(max_features)!=int or max_features<2:raise ValueError('max_features must be >=2')
        self.dimensions=dimensions;self.max_features=max_features;self.seed=seed
    @staticmethod
    def _texts(values):
        values=list(values)
        if not values or any(not isinstance(v,str) or not v.strip() for v in values):raise ValueError('texts must be nonempty strings')
        return values
    def fit(self,observations,*,epochs=100,corruption=.25,learning_rate=.01):
        texts=self._texts(observations)
        if type(epochs)!=int or epochs<1 or not 0<=corruption<1 or learning_rate<=0:raise ValueError('invalid training settings')
        vectorizer=TfidfVectorizer(ngram_range=(1,2),max_features=self.max_features,sublinear_tf=True)
        X=vectorizer.fit_transform(texts).toarray()
        rng=np.random.default_rng(self.seed);d=X.shape[1];h=min(self.dimensions,d)
        W1=rng.normal(0,1/np.sqrt(d),(d,h));b1=np.zeros(h)
        W2=rng.normal(0,1/np.sqrt(h),(h,d));b2=np.zeros(d)
        params=[W1,b1,W2,b2];m=[np.zeros_like(p) for p in params];v=[np.zeros_like(p) for p in params]
        losses=[];step=0
        # Fixed corruption sample tracks training loss consistently across epochs.
        validation_mask=rng.random(X.shape)>=corruption
        initial=np.tanh((X*validation_mask)@W1+b1)@W2+b2
        losses.append(float(np.mean(np.sum((initial-X)**2,axis=1))))
        for epoch in range(epochs):
            order=rng.permutation(len(X))
            for start in range(0,len(X),128):
                target=X[order[start:start+128]]
                observed=target*(rng.random(target.shape)>=corruption)
                hidden=np.tanh(observed@W1+b1);prediction=hidden@W2+b2
                dy=2*(prediction-target)/len(target)
                dW2=hidden.T@dy;db2=dy.sum(0)
                dh=(dy@W2.T)*(1-hidden*hidden)
                gradients=[observed.T@dh,dh.sum(0),dW2,db2]
                step+=1
                for i,(p,g) in enumerate(zip(params,gradients)):
                    g=np.clip(g,-5,5);m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g
                    p-=learning_rate*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
            predicted=np.tanh((X*validation_mask)@W1+b1)@W2+b2
            losses.append(float(np.mean(np.sum((predicted-X)**2,axis=1))))
        self.vectorizer=vectorizer;self.W1=W1;self.b1=b1;self.W2=W2;self.b2=b2
        self.training_mean=X.mean(axis=0)
        self.training_loss=losses;self.version=uuid.uuid4().hex
        self.reader=None
        return self
    def encode(self,texts):
        if not hasattr(self,'W1'):raise RuntimeError('train representation first')
        X=self.vectorizer.transform(self._texts(texts)).toarray()
        return np.tanh(X@self.W1+self.b1)
    def reconstruction_errors(self, texts, *, corruption=.25):
        if not 0<=corruption<1:raise ValueError('invalid corruption')
        X=self.vectorizer.transform(self._texts(texts)).toarray()
        rng=np.random.default_rng(123)
        prediction=np.tanh((X*(rng.random(X.shape)>=corruption))@self.W1+self.b1)@self.W2+self.b2
        return {'learned':float(np.mean(np.sum((prediction-X)**2,axis=1))),
                'mean_baseline':float(np.mean(np.sum((self.training_mean-X)**2,axis=1)))}
    @staticmethod
    def sentences(observations):
        observations=(observations,) if isinstance(observations,str) else tuple(observations)
        values=[]
        for text in AdaptiveRepresentation._texts(observations):
            values.extend(s.strip() for s in re.split(r'(?<=[.!?])\s+',text.strip()) if s.strip())
        return tuple(values)
    def build(self,observations):
        sentences=self.sentences(observations);vectors=self.encode(sentences)
        return AdaptiveSelf(uuid.uuid4().hex,self.version,sentences,
                            tuple(tuple(float(x) for x in v) for v in vectors),tuple(vectors.mean(axis=0)))
    @staticmethod
    def _pair_features(query,vectors):
        q=np.repeat(query[None,:],len(vectors),axis=0)
        return np.concatenate([q,vectors,q*vectors,np.abs(q-vectors)],axis=1)
    def fit_reader(self,examples):
        rows=list(examples)
        if not rows:raise ValueError('reader data is empty')
        features=[];labels=[]
        for row in rows:
            s=self.build(row['observations']);query=self.encode([row['question']])[0]
            target=row['answer_sentence']
            if target is not None and (type(target)!=int or not 0<=target<len(s.observations)):raise ValueError('invalid answer sentence')
            features.extend(self._pair_features(query,np.array(s.sentence_vectors)))
            labels.extend(int(i==target) for i in range(len(s.observations)))
        if len(set(labels))<2:raise ValueError('correct and incorrect candidates are needed')
        scaler=StandardScaler().fit(features);X=scaler.transform(features)
        reader=LogisticRegression(C=1,max_iter=1000,class_weight='balanced',random_state=self.seed).fit(X,labels)
        self.reader=reader;self.reader_scaler=scaler
        return self
    def respond(self,predicted_self,question,*,min_score=.65):
        if not isinstance(predicted_self,AdaptiveSelf) or predicted_self.version!=self.version:raise ValueError('adaptive self is stale or belongs to another model')
        if not 0<=min_score<=1:raise ValueError('min_score must be between zero and one')
        if self.reader is None:raise RuntimeError('train adaptive readout first')
        query=self.encode([question])[0]
        features=self._pair_features(query,np.array(predicted_self.sentence_vectors))
        scores=self.reader.predict_proba(self.reader_scaler.transform(features))[:,1];index=int(scores.argmax())
        accepted=float(scores[index])>=min_score
        return {'self_id':predicted_self.identifier,'text':predicted_self.observations[index] if accepted else '',
                'status':'selected_evidence' if accepted else 'insufficient_support',
                'candidate_index':index,'score':float(scores[index]),'calibrated':False,'candidate_scores':list(map(float,scores))}
