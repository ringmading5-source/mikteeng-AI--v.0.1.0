import re,numpy as np
from collections import Counter
from scipy.sparse import hstack
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
def tokens(text):return re.findall(r"\w+(?:['’]\w+)?|[^\w\s]",text.lower())
def render(words):
 text=' '.join(words);return re.sub(r'\s+([.,!?;:])',r'\1',text)
def state(words):
 f={'length':len(words)/25}
 for j,w in enumerate(words[-4:][::-1]):f[f'recent:{j}:{w}']=1
 for j,w in enumerate(words[:2]):f[f'first:{j}:{w}']=1
 for w,n in Counter(words).items():f['seen:'+w]=n/5
 f['recent_pair:'+ '|'.join(words[-2:])]=1
 return f
class Generator:
 def state(self,words):return state(words)
 def fit(self,rows):
  self.query=FeatureUnion([('words',TfidfVectorizer(ngram_range=(1,2))),('characters',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5)))])
  questions=[r['input'] for r in rows];Q=self.query.fit_transform(questions);features=[];labels=[];indices=[]
  for i,r in enumerate(rows):
   history=[]
   for w in tokens(r['answer'])+['<end>']:
    features.append(self.state(history));labels.append(w);indices.append(i);history.append(w)
  self.vectorizer=DictVectorizer();H=self.vectorizer.fit_transform(features);X=hstack([Q[indices]*3,H],format='csr');self.model=LogisticRegression(C=30,max_iter=600).fit(X,labels);self.training_steps=len(labels);return self
 def distribution(self,q,history):
  X=hstack([q*3,self.vectorizer.transform([self.state(history)])],format='csr');return self.model.predict_proba(X)[0]
 def generate(self,question,max_words=70):
  q=self.query.transform([question]);history=[];trace=[]
  for step in range(max_words):
   p=self.distribution(q,history);i=int(p.argmax());w=str(self.model.classes_[i]);trace.append(dict(step=step,word=w,probability=float(p[i])))
   if w=='<end>':return dict(text=render(history),status='completed',trace=trace)
   history.append(w)
  return dict(text=render(history),status='word_limit',trace=trace)
