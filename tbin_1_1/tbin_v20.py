"""TBIN v20: integrated CPU research prototype. Run: python tbin_v20.py
Requires numpy. Synthetic structured observations; not a general-purpose language model.
"""
import json, re, time, zlib
from collections import Counter, defaultdict
import numpy as np

def tok(s): return re.findall(r'[^\W_]+',s.casefold(),re.UNICODE)
def vec(items,d):
    v=np.zeros(d,np.float32)
    for item in items:
        h=zlib.crc32(item.encode()); v[h%d]+=1.0 if h&256 else -1.0
    return v/(np.linalg.norm(v)+1e-8)

class TBIN:
    def __init__(self,d=32,hidden=32,seed=3):
        self.d=d; self.levels={k:defaultdict(Counter) for k in ('characters','words','sentences','passages')}
        self.edges=defaultdict(list);self.concepts=defaultdict(Counter)
        self.programs=defaultdict(Counter);self.alias=defaultdict(Counter)
        self.max_passages=256; self.known_relations=set(); self.neural_trained=False
        self.passages=[];self.concept_vectors={};self.retrievals=Counter()
        rng=np.random.default_rng(seed)
        self.W=rng.normal(0,.12,(4*d,hidden)).astype(np.float32);self.b=np.zeros(hidden,np.float32)
        self.V=rng.normal(0,.12,(hidden,2)).astype(np.float32);self.c=np.zeros(2,np.float32)
    def observe(self,passage):
        ss=[tok(s) for s in re.split(r'[.!?\n]+',passage) if tok(s)]
        for s in ss:
            for w in s:
                for a,b in zip(w,w[1:]):self.levels['characters'][a][b]+=1
            for a,b in zip(s,s[1:]):self.levels['words'][a][b]+=1
            # Learn relation boundaries from triples; extend using known predicates.
            if len(s)==3:
                self.observe_fact(*s)
            else:
                hits=[i for i,w in enumerate(s) if w in self.known_relations and 0<i<len(s)-1]
                if len(hits)==1:
                    i=hits[0]; self.observe_fact(' '.join(s[:i]),s[i],' '.join(s[i+1:]))
        for a,b in zip(ss,ss[1:]):self.levels['sentences'][' '.join(a)][' '.join(b)]+=1
        if self.passages:self.levels['passages'][self.passages[-1]][passage]+=1
        self.passages.append(passage)
        self.passages=self.passages[-self.max_passages:]
        self.refresh_concepts()
    @staticmethod
    def entity(text):
        words=tok(text)
        # Optional English article normalization, not a learned universal parser.
        if words and words[0] in ('the','a','an'): words=words[1:]
        return ' '.join(words)
    def observe_fact(self,subject,relation,obj):
        subject=self.entity(subject);obj=self.entity(obj);relation=' '.join(tok(relation))
        if not subject or not relation or not obj:raise ValueError('Empty fact field')
        self.known_relations.add(relation)
        if (relation,obj) not in self.edges[subject]:self.edges[subject].append((relation,obj))
        self.concepts[relation][obj]+=1
        self.refresh_concepts()
    def refresh_concepts(self):
        self.concept_vectors={r:vec([r]+[x for x,n in c.items() for _ in range(min(n,3))],self.d) for r,c in self.concepts.items()}
    def paths(self,start,end,depth=3):
        frontier=[(start,(),frozenset([start]))];found=[]
        for _ in range(depth):
            next_frontier=[]
            for node,path,seen in frontier:
                for rel,target in self.edges[node]:
                    if target in seen:continue
                    p=path+(rel,)
                    if target==end:found.append(p)
                    else:next_frontier.append((target,p,seen|{target}))
            frontier=next_frontier
        return found
    def demonstrate(self,q,start,answer):
        start=self.entity(start);answer=self.entity(answer)
        ps=self.paths(start,answer)
        if not ps:return False
        path=min(ps,key=lambda p:(len(p),p))
        for w in set(tok(q)):self.programs[w][path]+=1
        return True
    def teach_alias(self,unknown,known):self.alias[unknown.casefold()][known.casefold()]+=1
    def score_programs(self,q,use_alias=True):
        scores=Counter()
        for term in set(tok(q)):
            terms=[(term,1.)]
            if use_alias:
                total=sum(self.alias[term].values())
                if total:terms.extend((known,n/total) for known,n in self.alias[term].items())
            for w,weight in terms:
                for path,n in self.programs[w].items():scores[path]+=n*weight
        return scores
    def candidates(self,q,start,use_alias=True,use_program=True):
        if not use_program:return {}
        scores=self.score_programs(q,use_alias);out=defaultdict(float)
        for path,weight in scores.items():
            nodes={start}
            for relation in path:
                nodes={nxt for node in nodes for r,nxt in self.edges[node] if r==relation}
                if not nodes:break
            for node in nodes:out[node]+=weight
        return out
    def concept_for(self,q):
        # Retrieve a learned relation representation associated with the highest-scoring program.
        scores=self.score_programs(q)
        if not scores:return np.zeros(self.d,np.float32)
        path=max(sorted(scores),key=scores.get)
        vectors=[self.concept_vectors[r] for r in path if r in self.concept_vectors]
        self.retrievals['concept']+=1
        return np.mean(vectors,axis=0) if vectors else np.zeros(self.d,np.float32)
    def feature(self,q,entity,concept=None):
        a=vec(tok(q),self.d);b=vec(tok(entity),self.d)
        c=self.concept_for(q) if concept is None else concept
        return np.concatenate((a,b,a*b,c)).astype(np.float32)
    def train_neural(self,examples,epochs=120,lr=.1):
        if not examples:return []
        X=np.stack([self.feature(q,e) for q,e,y in examples]);Y=np.array([y for q,e,y in examples],dtype=np.int64)
        self.neural_trained=True
        losses=[]
        for ep in range(epochs):
            H=np.tanh(X@self.W+self.b);logits=H@self.V+self.c
            p=np.exp(logits-logits.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
            if ep in (0,epochs-1):losses.append(float(-np.log(p[np.arange(len(Y)),Y]+1e-9).mean()))
            grad=p.copy();grad[np.arange(len(Y)),Y]-=1;grad/=len(Y)
            dV=H.T@grad;dc=grad.sum(axis=0);dH=(grad@self.V.T)*(1-H*H)
            dW=X.T@dH;db=dH.sum(axis=0)
            self.V-=lr*dV;self.c-=lr*dc;self.W-=lr*dW;self.b-=lr*db
        return losses
    def neural_score(self,q,e):
        h=np.tanh(self.feature(q,e)@self.W+self.b);z=h@self.V+self.c
        return float(z[1]-z[0])
    def answer(self,q,start,use_alias=True,use_program=True,use_neural=True):
        start=self.entity(start)
        candidates=self.candidates(q,start,use_alias,use_program)
        if not candidates:return None
        return max(sorted(candidates),key=lambda e:candidates[e]+(.05*self.neural_score(q,e) if use_neural and self.neural_trained else 0))
    def consolidate(self, top_k=1):
        """Lossy pruning that preserves counters and supports further observations."""
        if top_k<1:raise ValueError('top_k must be positive')
        for k in ('characters','words'):
            self.levels[k]=defaultdict(Counter,{a:Counter(dict(v.most_common(top_k)))
                                             for a,v in self.levels[k].items()})
        return {'retained_top1':{k:len(self.levels[k]) for k in ('characters','words')},'lossy':True}

def benchmark():
    t=time.perf_counter();m=TBIN();n=32
    for i in range(n):
        m.observe(f'person{i} owns box{i}. person{i} visits town{i}.')
        m.observe(f'box{i} contains gem{i}. town{i} borders region{i}.')
        m.observe(f'gem{i} signals color{i}.')
    tasks=[('find gem','gem'),('find color','color'),('find region','region')]
    for i in range(16):
        for q,target in tasks:assert m.demonstrate(q,f'person{i}',f'{target}{i}')
    train=[]
    for i in range(16):
        for q,target in tasks:
            train.append((q,f'{target}{i}',1));train.append((q,f'town{i}' if target!='region' else f'box{i}',0))
    losses=m.train_neural(train)
    def evaluate(queries,use_alias=True,use_program=True,use_neural=True):
        return sum(m.answer(q,f'person{i}',use_alias,use_program,use_neural)==f'{target}{i}' for i in range(16,32) for q,target in queries)
    baseline=evaluate(tasks,use_neural=False)
    full=evaluate(tasks)
    no_program=evaluate(tasks,use_program=False)
    paraphrases=[('locate jewel','gem'),('identify hue','color'),('discover territory','region')]
    before=evaluate(paraphrases)
    for a,b in [('locate','find'),('jewel','gem'),('identify','find'),('hue','color'),('discover','find'),('territory','region')]:m.teach_alias(a,b)
    after=evaluate(paraphrases)
    before_predictions=[m.answer(q,f'person{i}') for i in range(16,32) for q,_ in tasks]
    consolidation=m.consolidate()
    preserved=before_predictions==[m.answer(q,f'person{i}') for i in range(16,32) for q,_ in tasks]
    result={'version':'20','train_entities':16,'heldout_entities':16,'observed_passages':len(m.passages),'training_examples':len(train),
      'test_questions':48,'full_correct':full,'without_neural_correct':baseline,'without_program_correct':no_program,
      'unseen_paraphrases_before_feedback':before,'unseen_paraphrases_after_explicit_feedback':after,'paraphrase_test_count':48,
      'neural_parameters':int(sum(x.size for x in (m.W,m.b,m.V,m.c))),'neural_loss_first_last':losses,
      'concept_retrieval_calls':m.retrievals['concept'],'concept_count':len(m.concept_vectors),
      'consolidation':consolidation,'application_retained_after_consolidation':preserved,
      'seconds':round(time.perf_counter()-t,4),
      'limitations':'Synthetic three-token relationship extraction; paths induced by supervised demonstrations; explicit synonym feedback; no unseen-fact inference or proven autonomous concept formation; neural improvement must be established by ablation.'}
    return result
if __name__=='__main__':
    result=benchmark()
    with open('/mnt/data/tbin_v20_results.json','w') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
