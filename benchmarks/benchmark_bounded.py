"""Synthetic sandwich benchmark; history/trigger/target vectors are supplied explicitly."""
import json
import numpy as np
from mikteeng_rspm.engine import BoundedRSPM,assert_exact

e=BoundedRSPM(dim=64,max_contexts=10,ttl=30)
rng=np.random.default_rng(101)
v=e.B.copy(); rules=[(v[0],v[1],v[2]),(v[3],v[1],v[4])]
for h,t,y in rules:
 for _ in range(25): e.train(h,t,y)
assert sum(c['persistent'] for c in e.contexts.values())==2
basis=e.B.copy(); allocations=0; drops=0
for _ in range(1000):
 h=e.L(rng.normal(size=64)); t=e.L(rng.normal(size=64)); y=e.L(rng.normal(size=64))
 old=e.next_context; res=e.train(h,t,y); allocations+=e.next_context-old
 drops+=res['status'].endswith('CAPACITY_REJECTED'); e.check_bounds()
s=e.snapshot(); correct=0; rejected=0
for i in range(100):
 h,t,y=rules[i%2]
 hn=e.L(h+.03*rng.normal(size=64)); tn=t+.03*rng.normal(size=64)
 q=e.query(hn,tn)
 rejected+=q['status']!='MATCH'
 correct+=int(q['status']=='MATCH' and len(q['hypotheses'])==1 and q['hypotheses'][0]@y>.99)
assert_exact(s,e.snapshot()); assert_exact(basis,e.B)
assert correct==100
result=dict(novelty_observations=1000,novel_context_allocations=allocations,capacity_rejections=drops,
 frozen_noisy_retrieval_correct=correct,frozen_noisy_retrieval_queries=100,rejected_queries=rejected,
 retained_persistent_contexts=sum(c['persistent'] for c in e.contexts.values()),
 final_contexts=len(e.contexts),basis_unchanged=True,query_complete_snapshot_unchanged=True,
 limitation='Synthetic anchored outcome retrieval; does not establish language generation or unseen-rule generalization.')
with open('bounded_benchmark_results.json','w') as f: json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
