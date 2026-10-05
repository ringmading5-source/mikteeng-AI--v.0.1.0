import re,numpy as np
from collections import Counter,OrderedDict
from .predictor import Predictor
from .features import tokens
class CharacterStream:
 def __init__(self,p):self.p=p;self.h=np.zeros(p['U'].shape[0]);self.text='';self.states=[];self.updates=0
 def append(self,text):
  p=self.p
  for c in text.lower():
   base=self.h@p['U']+p['bias']
   if c in p['ci']:base=base+p['W'][p['ci'][c]]
   self.h=np.tanh(base);self.states.append(self.h.copy());self.updates+=1
  self.text+=text.lower()
  return self.h.copy()
class OptimizedPredictor(Predictor):
 def __init__(self,original,cache_size=256):
  self.__dict__.update(original.__dict__);self.cache=OrderedDict();self.cache_size=cache_size;self.hits=self.misses=0
 def word_state(self,word):
  if word in self.cache:
   self.hits+=1;self.cache.move_to_end(word);return self.cache[word]
  self.misses+=1;stream=CharacterStream(self.character);state=stream.append(word);self.cache[word]=state
  if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
  return state
 def state_features(self,sentence,stream=None):
  # Tokenize and calculate shared features once, not once per candidate.
  w=tokens(sentence);n=len(w);counts=Counter(w);shared={'length':n}
  shared.update({'count:'+x:c for x,c in counts.items()})
  shared.update({f'first:{j}:{x}':1 for j,x in enumerate(w[:2])});shared.update({f'last:{j}:{x}':1 for j,x in enumerate(w[-2:])})
  fs=[];before=Counter();after=counts.copy()
  for k,word in enumerate(w):
   after[word]-=1
   f=shared.copy();f.update({'candidate_count':counts[word],'candidate:'+word:1,'position':k/max(1,n-1),'self:'+word:1})
   f.update({'before:'+x:c for x,c in before.items() if c});f.update({'after:'+x:c for x,c in after.items() if c})
   for j,x in enumerate(w):f[f'offset:{j-k}:{x}']=1
   fs.append(f);before[word]+=1
  if self.character is None:return fs
  if stream is None:stream=CharacterStream(self.character);stream.append(sentence)
  if stream.text!=sentence.lower():raise ValueError('Stream must contain exactly this sentence')
  for k,match in enumerate(re.finditer(r'\w+|[^\w\s]',sentence.lower())):
   for d,value in enumerate(self.word_state(match.group())):fs[k]['char_word:'+str(d)]=float(value)
   for d,value in enumerate(stream.states[match.end()-1]):fs[k]['char_prefix:'+str(d)]=float(value)
  return fs
 def predict_stream(self,stream):
  sentence=stream.text;w=tokens(sentence);X=self.vectorizer.transform(self.state_features(sentence,stream));return {role:w[int(clf.predict_proba(X)[:,1].argmax())] for role,clf in self.heads.items()}
class RoutedPredictor:
 def __init__(self,word,character,vocabulary):self.word=word;self.character=character;self.vocabulary=set(vocabulary);self.routes={}
 def bucket(self,text):return 'unfamiliar_words' if any(x not in self.vocabulary for x in tokens(text)) else 'familiar_words'
 def calibrate(self,validation):
  scores={}
  for row in validation:
   bucket=self.bucket(row['sentence']);score=scores.setdefault(bucket,{'word':0,'character':0,'questions':0});score['questions']+=2
   for name,model in [('word',self.word),('character',self.character)]:
    answer=model.predict(row['sentence']);score[name]+=sum(answer[r]==row[r] for r in answer)
  # Select higher validation accuracy; ties choose the cheaper word path.
  self.routes={b:'character' if s['character']>s['word'] else 'word' for b,s in scores.items()};return scores
 def predict(self,text):
  route=self.routes.get(self.bucket(text),'word');return (self.character if route=='character' else self.word).predict(text)
