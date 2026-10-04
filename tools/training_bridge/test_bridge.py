import json, copy, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import bridge

class BridgeTest(unittest.TestCase):
 def test_teacher_request(self):
  rows=[{'observations':['alice waters the beans.'],'question':'Who waters the beans?','answer':'alice'}]
  class Response:
   def __enter__(self): return self
   def __exit__(self,*args): pass
   def read(self,size): return json.dumps({'choices':[{'message':{'content':json.dumps(rows)}}]}).encode()
  with patch.object(bridge,'urlopen',return_value=Response()) as call:
   self.assertEqual(bridge.propose('https://example.test/chat/completions','teacher','actors',1,['alice'],'test-key'),rows)
   self.assertEqual(call.call_args.args[0].headers['Authorization'],'Bearer test-key')
  with self.assertRaises(ValueError): bridge.propose('https://example.test/chat/completions','teacher','actors',1,[],None)
 def test_overlap_and_regression(self):
  row=lambda answer,q:{'observations':['alice waters the beans.'],'question':q,'answer':answer}
  class Model:
   subject_conditioned_predictor=SimpleNamespace(role_model=object())
   answer='alice'
   def build_sentence_prediction_self(self,obs): return obs
   def ask(self,q,predicted_self): return {'text':self.answer}
   def train_subject_conditioned_questions(self,rows,role_model): self.answer='brian'
  ai=Model()
  with self.assertRaises(ValueError): bridge.train_candidate(ai,[row('alice','q')],[row('alice','x')],[row('alice','q')])
  candidate,report=bridge.train_candidate(ai,[row('alice','q')],[row('brian','x')],[row('alice','holdout')])
  self.assertFalse(report['passed']);self.assertEqual(report['regressions'],[0]);self.assertEqual(ai.answer,'alice')
 def test_real_learner_roundtrip(self):
  from mikteeng_ai import MikteengAI
  from threadpoolctl import threadpool_limits
  ai=MikteengAI.load(bridge.ROOT/'validation/subject_connection/subject_connected_candidate.mkteeng')
  rows=json.loads((bridge.ROOT/'validation/subject_connection/mixed_training_additions.json').read_text())[:4]
  candidate=copy.deepcopy(ai)
  with threadpool_limits(limits=2): candidate.train_subject_conditioned_questions(rows,role_model=ai.subject_conditioned_predictor.role_model)
  expected=bridge.evaluate(candidate,rows)
  self.assertEqual(sum(r['correct'] for r in expected),4)
  with tempfile.TemporaryDirectory() as d:
   path=candidate.save(Path(d)/'candidate.mkteeng')
   self.assertEqual(bridge.evaluate(MikteengAI.load(path),rows),expected)
if __name__=='__main__': unittest.main()
