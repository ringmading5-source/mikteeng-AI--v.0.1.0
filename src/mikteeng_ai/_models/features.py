import re
from collections import Counter
def tokens(s):return re.findall(r'\w+|[^\w\s]',s.lower())
def features(sentence,index,ordered):
 w=tokens(sentence);c=Counter(w);f={'length':len(w),'candidate_count':c[w[index]],'candidate:'+w[index]:1}
 # Word identity and global summary are shared by both models.
 for x,n in c.items():f['count:'+x]=n
 for j,x in enumerate(w[:2]):f[f'first:{j}:{x}']=1
 for j,x in enumerate(w[-2:]):f[f'last:{j}:{x}']=1
 if ordered:
  # Generic ordered view around each possible answer; no grammar/role parser.
  f['position']=index/max(1,len(w)-1)
  for j,x in enumerate(w):
   d=j-index
   f[f'offset:{d}:{x}']=1
   side='before' if d<0 else 'after' if d>0 else 'self'
   f[f'{side}:{x}']=f.get(f'{side}:{x}',0)+1
 return f
