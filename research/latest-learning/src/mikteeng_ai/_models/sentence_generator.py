from collections import Counter
from .generator import Generator,state

def sentence_state(words):
 completed=[];current=[]
 for word in words:
  current.append(word)
  if word in ['.','!','?']:
   completed.append(current);current=[]
 f=state(words)
 f['sentence_number']=len(completed)
 f['current_sentence_length']=len(current)/25
 f['sentence_start']=float(not current)
 for j,w in enumerate(current[:2]):f[f'current_first:{j}:{w}']=1
 for j,w in enumerate(current[-4:][::-1]):f[f'current_recent:{j}:{w}']=1
 # Sequence of previous sentence summaries. No answer IDs or sentence lookup.
 for back,sentence in enumerate(completed[-2:][::-1]):
  f[f'previous:{back}:length']=len(sentence)/25
  for j,w in enumerate(sentence[:3]):f[f'previous:{back}:first:{j}:{w}']=1
  for j,w in enumerate(sentence[-4:][::-1]):f[f'previous:{back}:last:{j}:{w}']=1
  for w,n in Counter(sentence).items():f[f'previous:{back}:word:{w}']=n
  for a,b in zip(sentence,sentence[1:]):f[f'previous:{back}:pair:{a}|{b}']=1
 for sentence in completed:
  for w in set(sentence):f['passage_sentences_containing:'+w]=f.get('passage_sentences_containing:'+w,0)+1
 return f,completed,current
class SentenceGenerator(Generator):
 def state(self,words):return sentence_state(words)[0]
 def sentence_trace(self,text):
  from .generator import tokens
  history=[];events=[]
  for word in tokens(text):
   history.append(word)
   if word in ['.','!','?']:
    _,completed,current=sentence_state(history);events.append({'sentence_number':len(completed),'completed_sentence':completed[-1],'passage_words':len(history)})
  return events
