import unittest,copy,tempfile
from pathlib import Path
import numpy as np
from mikteeng_ai import MikteengAI
class ZeroRoles:
 def __init__(self,roles):self.roles=roles
 def predictions(self,words):
  result=copy.deepcopy(self.roles.predictions(words))
  for row in result:
   for key in row['scores']:row['scores'][key]=0.0
  return result
class SubjectConnectionTest(unittest.TestCase):
 def test_connected_features_dispatch_and_roundtrip(self):
  ai=MikteengAI.load('models/question_conditioned_predictions.mkteeng');roles=MikteengAI.load('models/biology_physics.mkteeng').roles
  a=['alice selects the beans.','alice plants the beans.','alice waters the beans.']
  b=['brian selects the maize.','brian plants the maize.','brian waters the maize.']
  rows=[{'observations':a,'question':'Who waters the beans?','answer':'alice'},{'observations':b,'question':'Who waters the maize?','answer':'brian'}]
  before=copy.deepcopy(roles.heads['subject'].coef_)
  ai.train_subject_conditioned_questions(rows,role_model=roles);p=ai.subject_conditioned_predictor
  self.assertIsNot(p.role_model,roles);np.testing.assert_equal(before,roles.heads['subject'].coef_)
  s=ai.build_sentence_prediction_self(a);x=p.features(s,'Who waters the beans?')
  original=p.role_model;p.role_model=ZeroRoles(original);zero=p.features(s,'Who waters the beans?');p.role_model=original
  self.assertEqual(x.shape,zero.shape);self.assertGreater((x-zero).nnz,0)
  self.assertEqual(ai.ask('Who waters the beans?',predicted_self=s)['text'],'alice')
  with tempfile.TemporaryDirectory() as d:
   path=ai.save(Path(d)/'connected.mkteeng');loaded=MikteengAI.load(path)
   self.assertEqual(loaded.ask('Who waters the beans?',predicted_self=loaded.build_sentence_prediction_self(a))['text'],'alice')
 def test_role_model_required(self):
  ai=MikteengAI.load('models/question_conditioned_predictions.mkteeng')
  with self.assertRaises(RuntimeError):ai.train_subject_conditioned_questions([])
if __name__=='__main__':unittest.main()
