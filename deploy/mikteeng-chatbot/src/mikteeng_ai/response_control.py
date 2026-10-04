"""Learn answer/withhold/silence decisions from checked response examples."""
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
class LearnedResponseControl:
    def features(self,observations,question,answer):
        q=self.words.transform([question]);a=self.words.transform([answer or ' ']);o=self.words.transform([' '.join(observations)])
        return sparse.hstack([q,a,o,sparse.kron(q,a,format='csr'),sparse.kron(a,o,format='csr'),sparse.kron(q,o,format='csr')],format='csr')
    def fit(self,rows):
        rows=list(rows)
        if not rows:raise ValueError('empty checked response examples')
        if any(r['decision'] not in ['answer','withhold','silence'] for r in rows):raise ValueError('unknown response decision')
        self.words=CountVectorizer().fit([s for r in rows for s in r['observations']]+[r['question'] for r in rows]+[r['answer'] or ' ' for r in rows])
        x=sparse.vstack([self.features(r['observations'],r['question'],r['answer']) for r in rows]);y=[r['decision'] for r in rows]
        if len(set(y))<2:raise ValueError('need multiple checked decisions')
        self.model=LogisticRegression(C=10,class_weight='balanced',max_iter=600).fit(x,y)
        return self
    def decide(self,observations,question,answer):
        return str(self.model.predict(self.features(observations,question,answer))[0])
