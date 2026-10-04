"""Supervised question conditioning over prediction-pattern self and evidence."""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from uuid import uuid4

class PatternQuestionReader:
    def __init__(self, sentence_model, use_prediction_history=True):
        self.sentence_model=sentence_model;self.use_prediction_history=use_prediction_history
    def features(self,s,question):
        model=self.sentence_model
        candidates=[model.decode(v) for v in s.observations]
        q=self.vectorizer.transform([question]);c=self.vectorizer.transform(candidates)
        # Learned cross-word associations, not programmed grammatical roles.
        cross=sparse.vstack([sparse.kron(q,c[i],format='csr') for i in range(len(candidates))])
        words=c.multiply(sparse.vstack([q]*len(candidates)))
        # Question-dependent scoring receives histories of first-level predictions.
        hist=np.asarray(s.pattern).reshape(model.learner.prediction_window,-1)
        evidence=np.asarray(s.observations)
        interactions=np.concatenate([evidence*h for h in hist],axis=1)
        interactions=interactions/np.maximum(np.linalg.norm(interactions,axis=1,keepdims=True),1e-12)
        parts=[cross,words]
        if self.use_prediction_history:parts.append(sparse.csr_matrix(interactions))
        return sparse.hstack(parts,format='csr'),candidates
    def fit(self,rows):
        rows=list(rows)
        if not rows:raise ValueError('empty question training')
        self.vectorizer=CountVectorizer(lowercase=True)
        self.vectorizer.fit([x for r in rows for x in r['observations']]+[r['question'] for r in rows])
        xx=[];yy=[]
        for r in rows:
            s=self.sentence_model.build_self(r['observations']);x,c=self.features(s,r['question'])
            if r['answer_sentence'] is not None and (type(r['answer_sentence']) is not int or not 0<=r['answer_sentence']<len(c)):raise ValueError('invalid answer index')
            xx.append(x);yy.extend(int(i==r['answer_sentence']) for i in range(len(c)))
        self.classifier=LogisticRegression(C=10,class_weight='balanced',max_iter=1000).fit(sparse.vstack(xx),yy)
        self.version=self.sentence_model.learner.version
        self.training_rows=[dict(r,observations=list(r["observations"])) for r in rows]
        self.response_version=uuid4().hex
        return self
    def respond(self,s,question):
        if s.version!=self.version or self.sentence_model.learner.version!=self.version:raise ValueError('stale self or reader; retrain question reader')
        if not isinstance(question,str) or not question.strip():raise ValueError('question must be nonempty text')
        x,c=self.features(s,question);scores=self.classifier.predict_proba(x)[:,1];i=int(scores.argmax())
        abstain=getattr(self,'allow_abstention',False) and scores[i]<.5
        return {'text':'Insufficient supporting evidence.' if abstain else c[i],'evidence_index':None if abstain else i,'status':'insufficient_evidence' if abstain else 'answered','selection_score':float(scores[i]),'score_calibrated':False,
                'response_version':getattr(self,'response_version',None),
                'mechanism':'question-conditioned prediction-pattern self with observation evidence'}
