import copy,json,unittest
import numpy as np
from mikteeng_rspm.engine import BoundedRSPM,assert_exact
class Audit(unittest.TestCase):
 def setUp(self):
  self.e=BoundedRSPM(dim=16,max_contexts=4,max_triggers=3,max_modes=2,ttl=15); self.v=self.e.B.copy(); self.h,self.t,self.y=self.v[:3]
 def establish(self,h=None,t=None,y=None):
  h=self.h if h is None else h; t=self.t if t is None else t; y=self.y if y is None else y
  for _ in range(20): self.e.train(h,t,y)
  return self.e.query(h,t)['context_id']
 def test_learning(self):
  cid=self.establish(); self.assertTrue(self.e.contexts[cid]['persistent']); self.assertGreater(self.e.query(self.h,self.t)['parametric']@self.y,.99)
 def test_readonly_and_alias(self):
  self.establish(); s=self.e.snapshot()
  for _ in range(30): self.e.query(self.h,self.t); self.e.query(self.v[5],self.t); self.e.query(self.h,self.v[6])
  self.e.query(self.h,self.t)['hypotheses'][0][:]=0; assert_exact(s,self.e.snapshot()); self.assertFalse(self.e.B.flags.writeable)
 def test_isolation_flood(self):
  a=self.establish(); old=copy.deepcopy(self.e.contexts[a]); self.establish(self.v[3],self.t,self.v[4])
  for i in range(100): self.e.train(self.e.L(np.random.default_rng(i).normal(size=16)),self.v[7],self.v[8]); self.e.check_bounds()
  assert_exact(old,self.e.contexts[a]); self.assertGreater(self.e.query(self.h,self.t)['hypotheses'][0]@self.y,.999)
 def test_multimodal_capacity(self):
  self.establish()
  for _ in range(8): self.e.train(self.h,self.t,self.v[3])
  q=self.e.query(self.h,self.t); self.assertEqual(q['forecast_type'],'MULTIMODAL'); old=self.e.contexts[q['context_id']]['R'].copy()
  events=[self.e.train(self.h,self.t,self.v[4])['event'] for _ in range(5)]
  self.assertIn('OUTCOME_CAPACITY_REJECTED',events); assert_exact(old,self.e.contexts[q['context_id']]['R'])
 def test_ambiguity(self):
  self.establish(); t2=np.cos(np.deg2rad(40))*self.t+np.sin(np.deg2rad(40))*self.v[5]
  for _ in range(3): self.e.train(self.h,t2,self.y)
  mid=(self.t+t2)/np.linalg.norm(self.t+t2); s=self.e.snapshot(); self.assertEqual(self.e.query(self.h,mid)['status'],'AMBIGUOUS_TRIGGER'); assert_exact(s,self.e.snapshot())
  old=copy.deepcopy(self.e.contexts); self.assertEqual(self.e.train(self.h,mid,self.y)['status'],'AMBIGUOUS_TRIGGER'); assert_exact(old,self.e.contexts)
 def test_ttl_ids(self):
  self.establish(); self.e.train(self.h,self.v[5],self.y); tid=self.e.next_trigger-1
  for _ in range(17): self.e.train(self.h,self.t,self.y)
  cid=self.e.query(self.h,self.t)['context_id']; self.assertNotIn(tid,self.e.contexts[cid]['stores']); self.e.train(self.h,self.v[6],self.y); self.assertGreater(self.e.next_trigger-1,tid)
 def test_merge(self):
  a=self.establish(); self.e.contexts[99]=copy.deepcopy(self.e.contexts[a]); self.e.contexts[99]['stores'][0]['modes'][0]['centroid']=self.v[9].copy()
  s=self.e.snapshot(); self.assertEqual(self.e.merge(a,99)['status'],'NO_OP'); assert_exact(s,self.e.snapshot())
  self.e.contexts[99]=copy.deepcopy(self.e.contexts[a]); self.assertEqual(self.e.merge(a,99)['status'],'MERGED'); self.assertEqual(len(self.e.contexts[a]['stores'][0]['modes']),1)
 def test_restore(self):
  a=self.establish(); self.e.contexts[a]['R'][0,0]=-0.; s=self.e.snapshot(); self.e.contexts[a]['R'][0,0]=0.; self.e.restore(s); assert_exact(s,self.e.snapshot())
 def test_trigger_capacity(self):
  self.establish()
  for t in (self.v[5],self.v[6]): self.establish(t=t)
  self.assertEqual(self.e.train(self.h,self.v[7],self.y)['status'],'TRIGGER_CAPACITY_REJECTED')
 def test_invalid(self):
  s=self.e.snapshot()
  with self.assertRaises(ValueError): self.e.train(np.zeros(16),self.t,self.y)
  assert_exact(s,self.e.snapshot())
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Audit))
 with open('bounded_results.json','w') as f: json.dump(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors)),f,indent=2)
 raise SystemExit(not result.wasSuccessful())
