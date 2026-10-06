import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import pipeline
from demo import make_demo
class PipelineTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.manifest=make_demo(self.root/'data');self.rows=[json.loads(x) for x in self.manifest.read_text().splitlines()]
  for r in self.rows:
   j=int(r['audio'].split('_')[-1].split('.')[0]);r['split']='train' if j<4 else 'validation' if j<6 else 'test';r['speaker_id']=r['split']+'_generator'
  self.write()
 def tearDown(self):self.tmp.cleanup()
 def write(self):self.manifest.write_text('\n'.join(json.dumps(r) for r in self.rows)+'\n')
 def test_select_validation_test_once(self):
  real=pipeline.evaluate;calls=[]
  def tracked(m,rows):calls.append(rows[0]['split']);return real(m,rows)
  with patch.object(pipeline,'evaluate',side_effect=tracked):r=pipeline.run(self.manifest,self.root/'run',epochs=10,patience=3)
  self.assertEqual(calls.count('test'),1);self.assertEqual(calls[-1],'test');self.assertEqual(r['test']['metrics']['correct'],3);self.assertLess(r['completed_epochs'],10)
 def test_speaker_leakage(self):
  self.rows[-1]['speaker_id']='train_generator';self.write()
  with self.assertRaises(ValueError):pipeline.run(self.manifest,self.root/'run')
  self.assertFalse((self.root/'run').exists())
 def test_missing_validation(self):
  for r in self.rows:
   if r['split']=='validation':r.update(split='test',speaker_id='test_generator')
  self.write()
  with self.assertRaises(ValueError):pipeline.prepare_manifest(self.manifest,'paired')
 def test_voice_only_no_accuracy_claim(self):
  for r in self.rows:r.pop('transcript')
  self.write();r=pipeline.run(self.manifest,self.root/'run',mode='voice-only',epochs=8,patience=3);self.assertNotIn('correct',r['test']['metrics']);self.assertIn('not semantic',r['selection_metric'])
 def test_output_not_overwritten(self):
  (self.root/'run').mkdir()
  with self.assertRaises(FileExistsError):pipeline.run(self.manifest,self.root/'run')
if __name__=='__main__':unittest.main()
