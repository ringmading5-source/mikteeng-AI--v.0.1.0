"""Train unchanged learner; separate public routing from operator diagnostics."""
import base64,json
import numpy as np
from mikteeng_rspm.engine import BoundedRSPM,assert_exact

def pack(x):
 if isinstance(x,np.ndarray): return {'array':base64.b64encode(x.tobytes()).decode(),'dtype':x.dtype.str,'shape':list(x.shape)}
 if isinstance(x,dict): return {'dict':[[pack(k),pack(v)] for k,v in x.items()]}
 if isinstance(x,list): return {'list':[pack(v) for v in x]}
 if isinstance(x,np.generic): return x.item()
 return x

def unpack(x):
 if not isinstance(x,dict): return x
 if 'array' in x: return np.frombuffer(base64.b64decode(x['array']),dtype=x['dtype']).copy().reshape(x['shape'])
 if 'dict' in x: return {unpack(k):unpack(v) for k,v in x['dict']}
 return [unpack(v) for v in x['list']]

def err(a,b): return float(np.linalg.norm(a-b))

def run(seed,save=False):
 e=BoundedRSPM(dim=32,seed=seed,max_contexts=6,max_triggers=12,ttl=200)
 rng=np.random.default_rng(seed+1000); v=e.B.copy(); histories=[v[24],v[25]]; inputs=v[:8]; outputs=v[8:16]; maps=[np.arange(8),np.arange(7,-1,-1)]; curve=[]
 for domain,h in enumerate(histories):
  for epoch in range(60):
   for i in rng.permutation(8): e.train(h,inputs[i],outputs[maps[domain][i]])
   if epoch in (0,2,9,29,59):
    cid=e.query(h,inputs[0])['context_id']; R=e.contexts[cid]['R']
    curve.append(dict(domain=domain,exposures=(epoch+1)*8,error=float(np.mean([err(e.L(t+R@t),outputs[maps[domain][i]]) for i,t in enumerate(inputs)]))))
  if domain==0: cid_a=e.query(h,inputs[0])['context_id']; preserved=e.contexts[cid_a]['R'].copy()
 assert_exact(preserved,e.contexts[cid_a]['R'])
 snap=e.snapshot(); cases={name:[] for name in ('seen','unseen_combinations','nearby_unseen','noisy_combinations')}
 for domain,h in enumerate(histories):
  cid=e.query(h,inputs[0])['context_id']; R=e.contexts[cid]['R']
  for kind in cases:
   for j in range(100):
    if kind=='seen': coeff=np.eye(8)[j%8]
    elif kind=='nearby_unseen':
     coeff=np.zeros(8); coeff[j%8]=1.; coeff[(j+1)%8]=.12
    else:
     coeff=np.zeros(8); ids=rng.choice(8,size=int(rng.integers(2,5)),replace=False); coeff[ids]=rng.uniform(.5,1.,size=len(ids))
     if kind=='noisy_combinations': coeff+=rng.uniform(0,.03,size=8)
    coeff/=np.linalg.norm(coeff); t=coeff@inputs; target=coeff@outputs[maps[domain]]
    q=e.query(h,t); nearest=outputs[maps[domain][int(np.argmax(inputs@t))]]; mean=e.vector(np.mean(outputs,axis=0))
    public=err(q['parametric'],target) if q['status']=='MATCH' else None
    retrieved=err(q['hypotheses'][int(np.argmax(q['probabilities']))],target) if q['status']=='MATCH' and q['hypotheses'] else None
    cases[kind].append(dict(public=public,retrieved=retrieved,diagnostic=err(e.L(t+R@t),target),identity=err(e.L(t),target),nearest=err(nearest,target),mean=err(mean,target)))
 assert_exact(snap,e.snapshot()); summary={}
 for kind,rows in cases.items():
  summary[kind]={'queries':len(rows),'public_matches':sum(r['public'] is not None for r in rows)}
  for metric in rows[0]:
   vals=[r[metric] for r in rows if r[metric] is not None]; summary[kind][metric+'_mean_error']=float(np.mean(vals)) if vals else None
 for _ in range(6): e.train(histories[0],inputs[0],outputs[1])
 conflict=e.query(histories[0],inputs[0]); assert len(conflict['hypotheses'])==2
 if save:
  with open('trained_bounded_generalization.json','w') as f: json.dump(pack(e.snapshot()),f)
  loaded=BoundedRSPM(); loaded.restore(unpack(json.load(open('trained_bounded_generalization.json')))); assert_exact(e.snapshot(),loaded.snapshot())
 return dict(seed=seed,training_observations=960,curve=curve,cases=summary,cross_context_operator_unchanged=True,frozen_snapshot_unchanged=True,conflict_modes=len(conflict['hypotheses']))

if __name__=='__main__':
 results=dict(experiment='Two opposite coordinate sequence rules, eight single-symbol transitions per domain; no mixture training.',learner='Unchanged bounded_rspm.py',seeds=[run(s,s==42) for s in (42,43,44,45,46)],limitation='Operator diagnostic bypasses trigger gate for measurement only. Public metrics include rejections. Not language understanding.')
 with open('generalization_results.json','w') as f: json.dump(results,f,indent=2)
 print(json.dumps(results['seeds'][0],indent=2))
