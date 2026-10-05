"""Learned question-conditioned token prediction from prediction-pattern self."""
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

class QuestionConditionedPredictor:
    def __init__(self,sentence_model,self_source="predictions"):
        self.sentence_model=sentence_model;self.self_source=self_source
    def features(self,s,q):
        question=self.vectorizer.transform([q])
        # Predictions carry learned observation transitions, rather than a role parser.
        history=np.asarray(s.observations if self.self_source=="observations" else s.predictions)
        z=np.concatenate([history.mean(axis=0),history[-1]])
        if self.self_source=="hybrid":
            observations=np.asarray(s.observations)
            if len(observations)>self.max_sentences:raise ValueError('observation exceeds trained sentence capacity')
            intact=np.zeros((self.max_sentences,observations.shape[1]))
            intact[:len(observations)]=observations
            z=np.concatenate([z,intact.ravel()])
        character_model=getattr(self.sentence_model,"character_model",None)
        if character_model is not None:
            observations=" ".join(self.sentence_model.decode(v) for v in s.observations)
            z=np.concatenate([z,character_model.summary(observations),character_model.summary(q)])
        z=z/max(np.linalg.norm(z),1e-12)
        if self.self_source=="none":z=np.zeros_like(z)
        return sparse.hstack([question,sparse.csr_matrix(z[None,:]),sparse.kron(question,sparse.csr_matrix(z[None,:]),format='csr')],format='csr')
    def fit(self,rows):
        rows=list(rows)
        if not rows:raise ValueError('empty training')
        self.vectorizer=CountVectorizer().fit([r['question'] for r in rows])
        self.max_sentences=max(len(r['observations']) for r in rows)
        targets=[self.sentence_model.tokens(r['answer']) for r in rows]
        self.vocabulary=['<end>']+sorted(set(t for ts in targets for t in ts))
        self.width=max(map(len,targets))+1;self.heads=[]
        x=sparse.vstack([self.features(self.sentence_model.build_self(r['observations']),r['question']) for r in rows])
        for pos in range(self.width):
            y=[ts[pos] if pos<len(ts) else '<end>' for ts in targets]
            head=DummyClassifier(strategy='most_frequent') if len(set(y))==1 else LogisticRegression(C=20,max_iter=700)
            self.heads.append(head.fit(x,y))
        self.version=self.sentence_model.learner.version
        return self
    def respond(self,s,question):
        if s.version!=self.version:raise ValueError('stale self')
        if not isinstance(question,str) or not question.strip():raise ValueError('question must be nonempty')
        x=self.features(s,question);words=[]
        for head in self.heads:
            token=head.predict(x)[0]
            if token=='<end>':break
            words.append(token)
        text=re.sub(r'\s+([.,!?;:])',r'\1',' '.join(words))
        return {'text':text,'mechanism':'question-conditioned answer-token prediction','calibrated':False}
