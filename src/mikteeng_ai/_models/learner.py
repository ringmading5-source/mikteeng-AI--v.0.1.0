import re,math
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np
def tokenize(text):return re.findall(r'\w+|[^\w\s]',text.lower())
class PassageLearner:
 def __init__(self):self.vectorizer=DictVectorizer();self.heads={}
 def features(self,w,query,index,prepared=None):
  counts,anchors=prepared if prepared is not None else (Counter(w),[j for j,x in enumerate(w) if x==query]);f={'length_log':math.log1p(len(w)),'candidate:'+w[index]:1,'count':counts[w[index]],'position':index/max(1,len(w)-1)}
  for j,x in enumerate(w[:2]):f[f'first:{j}:{x}']=1
  for j,x in enumerate(w[-2:]):f[f'last:{j}:{x}']=1
  for d in range(-6,7):
   j=index+d
   if 0<=j<len(w):f[f'ordered:{d}:{w[j]}']=1
  for a in anchors:
   d=a-index;key=str(d) if abs(d)<=24 else ('far_before' if d<0 else 'far_after');f['query_offset:'+key]=f.get('query_offset:'+key,0)+1
  return f
 def fit(self,passages,rng):
  fs=[];labels={'actor':[],'receiver':[]}
  for row in passages:
   w=tokenize(row['passage'])
   for q in row['queries']:
    prepared=(Counter(w),[j for j,x in enumerate(w) if x==q['action']]);positive={q['actor_index'],q['receiver_index']};pool=[i for i in range(len(w)) if i not in positive]
    # Balance examples without choosing grammatical candidates.
    selected=sorted(positive|set(rng.sample(pool,min(14,len(pool)))))
    for i in selected:
     fs.append(self.features(w,q['action'],i,prepared))
     for role in labels:labels[role].append(int(i==q[role+'_index']))
  X=self.vectorizer.fit_transform(fs)
  for role,y in labels.items():self.heads[role]=SGDClassifier(loss='log_loss',alpha=.00001,max_iter=100,tol=.0001,random_state=21,class_weight='balanced').fit(X,y)
  self.training_candidates=len(fs);return self
 def predict(self,text,action):
  w=tokenize(text);prepared=(Counter(w),[j for j,x in enumerate(w) if x==action]);X=self.vectorizer.transform([self.features(w,action,i,prepared) for i in range(len(w))]);return {role:w[int(clf.predict_proba(X)[:,1].argmax())] for role,clf in self.heads.items()}
