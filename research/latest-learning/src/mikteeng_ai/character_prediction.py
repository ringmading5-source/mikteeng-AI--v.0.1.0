"""Learn next-character distributions from raw text; fixed bounded context."""
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
class CharacterPredictionModel:
    def __init__(self,context=16):
        if type(context) is not int or context<1:raise ValueError('context must be positive')
        self.context=context
    def fit(self,texts):
        texts=list(dict.fromkeys(texts))
        if not texts or any(not isinstance(t,str) or not t for t in texts):raise ValueError('nonempty text observations required')
        x=[];y=[]
        for text in texts:
            for t in range(1,len(text)):
                x.append(text[max(0,t-self.context):t]);y.append(text[t])
        if len(set(y))<2:raise ValueError('need at least two target characters')
        self.vectorizer=CountVectorizer(analyzer='char',ngram_range=(1,3),lowercase=False).fit(x)
        self.model=LogisticRegression(C=3,max_iter=400).fit(self.vectorizer.transform(x),y)
        self.training_targets=len(y)
        return self
    def predict_next(self,text):
        if not isinstance(text,str) or not text:raise ValueError('nonempty character context required')
        p=self.model.predict_proba(self.vectorizer.transform([text[-self.context:]]))[0]
        return {'character':str(self.model.classes_[p.argmax()]),'characters':self.model.classes_.tolist(),'probabilities':p.tolist(),'calibrated':False}
    def summary(self,text):
        if not text:return np.zeros(len(self.model.classes_))
        contexts=[text[max(0,t-self.context):t] for t in range(1,len(text)+1)]
        return self.model.predict_proba(self.vectorizer.transform(contexts)).mean(axis=0)
