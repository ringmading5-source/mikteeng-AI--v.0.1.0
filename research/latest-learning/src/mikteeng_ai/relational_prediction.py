"""Question-conditioned evidence spans with intact event representations.
Supervised answer strings; no grammatical-role parser or role labels.
"""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

class RelationalPredictor:
    def __init__(self,sentence_model):self.sentence_model=sentence_model
    def candidates(self,s):
        rows=[]
        for vector in s.observations:
            sentence=self.sentence_model.decode(vector);tokens=self.sentence_model.tokens(sentence)
            # Every token and entire sentence are eligible; no assigned subject/action slots.
            for pos,token in enumerate(tokens):rows.append((token,sentence,pos,1,vector))
            rows.append((sentence,sentence,len(tokens),len(tokens),vector))
        return rows
    def features(self,s,q):
        rows=self.candidates(s);question=self.vocab.transform([q])
        context=self.vocab.transform([r[1] for r in rows]);answer=self.vocab.transform([r[0] for r in rows])
        # Preserve each source sentence separately, never average events together.
        predictions=self.sentence_model.learner.first.predict(np.array([r[4] for r in rows])).reshape(len(rows),-1)
        predictions=predictions/np.maximum(np.linalg.norm(predictions,axis=1,keepdims=True),1e-12)
        structural=np.zeros((len(rows),self.sentence_model.width+2))
        for i,r in enumerate(rows):structural[i,min(r[2],self.sentence_model.width)]=1;structural[i,-1]=r[3]
        event=sparse.hstack([context,answer,sparse.csr_matrix(structural),sparse.csr_matrix(predictions)],format='csr')
        cross=sparse.vstack([sparse.kron(question,event[i],format='csr') for i in range(len(rows))])
        return sparse.hstack([event,cross],format='csr'),rows
    def fit(self,examples):
        examples=list(examples)
        if not examples:raise ValueError('empty examples')
        self.vocab=CountVectorizer().fit([r['question'] for r in examples]+[s for r in examples for s in r['observations']])
        xx=[];yy=[]
        for r in examples:
            s=self.sentence_model.build_self(r['observations']);x,c=self.features(s,r['question'])
            target=r['answer'].lower();labels=[int(a[0]==target) for a in c]
            if not any(labels):raise ValueError('answer must be a source token or sentence')
            xx.append(x);yy.extend(labels)
        self.head=LogisticRegression(C=10,class_weight='balanced',max_iter=800).fit(sparse.vstack(xx),yy)
        self.version=self.sentence_model.learner.version
        return self
    def respond(self,s,q):
        if s.version!=self.version:raise ValueError('stale self')
        if not isinstance(q,str) or not q.strip():raise ValueError('question must be nonempty')
        x,rows=self.features(s,q);scores=self.head.predict_proba(x)[:,1];i=int(scores.argmax())
        return {'text':rows[i][0],'source_sentence':rows[i][1],'selection_score':float(scores[i]),'calibrated':False,'mechanism':'intact event predictions conditioned by question'}
