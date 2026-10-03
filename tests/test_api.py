import unittest,tempfile,json,zipfile
from pathlib import Path
from mikteeng_ai import MikteengAI
ROOT=Path(__file__).resolve().parents[1]
class ApiTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.demo=MikteengAI.load(ROOT/'models/biology_physics.mkteeng')
 def test_preserved_generation_and_roles(self):
  text=self.demo.generate('Explain cells and velocity.')['text']
  self.assertEqual(text,'cells are the basic structural and functional units of living organisms. velocity describes the rate of change of position and includes direction.')
  session=self.demo.role_session();session.push('the');before=session.push('cat')[1]['scores'];after=session.push('was')[1]['scores']
  self.assertLess(after['actor'],before['actor']);self.assertGreater(after['receiver'],before['receiver'])
  self.assertEqual(session.words,['the','cat','was'])
 def test_legacy_role_heads_and_character_state_preserved(self):
  self.assertEqual(self.demo.predict('Today the cat was chased by the dog outside.'),{'actor':'dog','receiver':'cat'})
  self.assertEqual(self.demo.answer_passage('The cat chased the dog.','chased'),{'actor':'cat','receiver':'dog'})
  stream=self.demo.new_character_stream();stream.append('the ');stream.append('cat');self.assertEqual(stream.updates,7)
 def test_training_and_save_load_roundtrip(self):
  rows=[{'input':'Explain speed.','answer':'Speed is distance divided by time.'},{'input':'What is speed?','answer':'Speed is distance divided by time.'},{'input':'Explain cells.','answer':'Cells are units of life.'}]
  model=MikteengAI().train(rows);before=model.generate('Explain speed.')
  with tempfile.TemporaryDirectory() as temp:
   path=model.save(Path(temp)/'new.mkteeng');loaded=MikteengAI.load(path)
   self.assertEqual(loaded.generate('Explain speed.'),before)
 def test_annotated_training_connects_to_generation(self):
  rows=[json.loads(x) for x in (ROOT/'experiments/annotated_role_training.jsonl').read_text().splitlines()[:6]]
  model=MikteengAI().train_roles(rows)
  self.assertEqual(len(model.predict_roles('the cat was chased by the dog.')),8)
  model.train_generation([{'input':'Explain cats.','answer':'Cats chase mice.'},{'input':'Explain dogs.','answer':'Dogs chase cats.'}],use_subjects=True)
  self.assertIs(model.generator.roles,model.roles)
 def test_validation_and_unknown_schema(self):
  with self.assertRaises(RuntimeError):MikteengAI().generate('hello')
  with self.assertRaises(ValueError):self.demo.generate('')
  with self.assertRaises(ValueError):self.demo.generate('hello',max_words=0)
  with self.assertRaises(ValueError):MikteengAI().train([],task='roles')
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'bad.mkteeng'
   with zipfile.ZipFile(path,'w') as z:z.writestr('metadata.json',json.dumps({'format':'mikteeng-ai','schema_version':99}));z.writestr('state.pkl',b'bad')
   with self.assertRaises(ValueError):MikteengAI.load(path)
if __name__=='__main__':unittest.main()
