"""Bounded supervised pattern selector around the unchanged RSPM operator learner."""
import copy
import numpy as np
try:
 from .original_engine import BoundedRSPM as Core,assert_exact
except ImportError:
 from original_engine import BoundedRSPM as Core,assert_exact
class BoundedRSPM(Core):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.routing_examples={};self.routing_ids=[];self.routing_W=np.zeros((0,self.config['dim']))
 def snapshot(self):
  s=super().snapshot();s.update(routing_examples=copy.deepcopy(self.routing_examples),routing_ids=list(self.routing_ids),routing_W=self.routing_W.copy());return s
 def restore(self,s):
  super().restore(s);self.routing_examples=copy.deepcopy(s.get('routing_examples',{}));self.routing_ids=list(s.get('routing_ids',[]));self.routing_W=s.get('routing_W',np.zeros((0,self.config['dim']))).copy()
 def train(self,history,trigger,target):
  result=super().train(history,trigger,target)
  self.routing_examples={i:xs for i,xs in self.routing_examples.items() if i in self.contexts or i==-1}
  if result["status"]=="TRAINED":self.teach_route(history,trigger)
  return result
 def teach_route(self,history,trigger):
  h=self.vector(self.L(self.vector(history)));t=self.vector(trigger)
  status,cid=self._route([(i,c['key']) for i,c in self.contexts.items()],h,self.config['tau_context'])
  if status!='MATCH':return {'status':status+'_CONTEXT'}
  self.routing_examples={i:xs for i,xs in self.routing_examples.items() if i in self.contexts or i==-1}
  xs=self.routing_examples.setdefault(cid,[])
  if any(np.array_equal(x,t) for x in xs) and self.routing_ids==sorted(self.routing_examples):return {'status':'ROUTE_KNOWN','context_id':cid}
  if not any(np.array_equal(x,t) for x in xs):
   if len(xs)>=self.config['max_triggers']:xs.pop(0)
   xs.append(t.copy())
  self._fit_routes()
  return {'status':'ROUTE_TAUGHT','context_id':cid}
 def teach_unmatched(self,trigger):
  t=self.vector(trigger);xs=self.routing_examples.setdefault(-1,[])
  if not any(np.array_equal(x,t) for x in xs):
   if len(xs)>=self.config['max_triggers']:xs.pop(0)
   xs.append(t.copy())
  self._fit_routes()
 def _fit_routes(self):
  self.routing_ids=sorted(self.routing_examples)
  X=np.array([x for i in self.routing_ids for x in self.routing_examples[i]])
  Y=np.array([[float(i==j) for j in self.routing_ids] for i in self.routing_ids for x in self.routing_examples[i]])
  self.routing_W=np.linalg.solve(X.T@X+.1*np.eye(self.config['dim']),X.T@Y).T
 def query(self,history,trigger):
  strict=super().query(history,trigger) if history is not None else dict(status='UNMATCHED_CONTEXT',hypotheses=[])
  if strict['status']=='MATCH':return strict
  t=self.vector(trigger);live=[(j,cid) for j,cid in enumerate(self.routing_ids) if cid in self.contexts or cid==-1]
  if not live:return strict
  scores=sorted([(float(self.routing_W[j]@t),cid) for j,cid in live],reverse=True)
  score,cid=scores[0];margin=score-(scores[1][0] if len(scores)>1 else 0.)
  if cid==-1:return dict(status='UNMATCHED_PATTERN',hypotheses=[],reason='Learned no-pattern class')
  coverage=max(float(x@t) for x in self.routing_examples[cid])
  if score<.6 or coverage<.4:return dict(status='UNMATCHED_PATTERN',hypotheses=[],route_score=score,coverage=coverage)
  if margin<.15:return dict(status='AMBIGUOUS_PATTERN',hypotheses=[],route_score=score,route_margin=margin)
  if strict['status']=='AMBIGUOUS_CONTEXT':return strict
  if 'context_id' in strict and strict['context_id']!=cid:return dict(status='CONTEXT_PATTERN_CONFLICT',hypotheses=[])
  c=self.contexts[cid]
  return dict(status='PATTERN_FORECAST',context_id=cid,parametric=self.L(t+c['R']@t),hypotheses=[],probabilities=np.array([]),confidence='UNVERIFIED',route_score=score,route_margin=margin,coverage=coverage)
 def query_pattern(self,trigger):return self.query(None,trigger)
 def query_strict(self,h,t):return super().query(h,t)

 def merge(self,a,b,max_operator_dist=.1):
  before=self.snapshot()
  try:
   result=super().merge(a,b,max_operator_dist)
   if result['status']!='MERGED':return result
   xs=self.routing_examples.setdefault(a,[])
   for x in self.routing_examples.pop(b,[]):
    if not any(np.array_equal(x,y) for y in xs):xs.append(x.copy())
   self.routing_examples[a]=xs[-self.config['max_triggers']:]
   if self.routing_examples:self._fit_routes()
   return result
  except Exception as exc:
   self.restore(before);return dict(status='FAILED',reason=str(exc))
