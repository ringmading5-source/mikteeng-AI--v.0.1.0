"""Supervised code-token prediction preserving whitespace and punctuation."""
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from .character_prediction import CharacterPredictionModel
class CodePredictionModel:
    @staticmethod
    def tokens(code):return re.findall(r'\s+|[A-Za-z_]\w*|\d+|[^\w\s]',code)
    def features(self,question):
        q=self.words.transform([question]);z=sparse.csr_matrix(self.characters.summary(question)[None,:])
        return sparse.hstack([q,z,sparse.kron(q,z,format='csr')],format='csr')
    def fit(self,rows):
        rows=list(rows)
        if not rows:raise ValueError('empty code training')
        self.characters=CharacterPredictionModel().fit([r['question'] for r in rows]+[r['code'] for r in rows])
        self.words=CountVectorizer().fit([r['question'] for r in rows])
        targets=[self.tokens(r['code']) for r in rows];self.width=max(map(len,targets))+1
        x=sparse.vstack([self.features(r['question']) for r in rows]);self.heads=[]
        for pos in range(self.width):
            y=[ts[pos] if pos<len(ts) else '<end>' for ts in targets]
            head=DummyClassifier(strategy='most_frequent') if len(set(y))==1 else LogisticRegression(C=20,max_iter=500)
            self.heads.append(head.fit(x,y))
        return self
    def generate(self,question):
        if not isinstance(question,str) or not question.strip():raise ValueError('nonempty code request required')
        x=self.features(question);tokens=[]
        for head in self.heads:
            token=head.predict(x)[0]
            if token=='<end>':break
            tokens.append(token)
        return {'code':''.join(tokens),'mechanism':'learned question-conditioned code tokens','calibrated':False}
