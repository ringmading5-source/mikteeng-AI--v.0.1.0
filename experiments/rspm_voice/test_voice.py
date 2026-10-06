import unittest,tempfile,json,copy
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from voice_model import VoiceModel
from engine import assert_exact
from demo import make_demo
from cli import train_run,read_manifest
class VoiceTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.manifest=make_demo(self.root/'data')
 def tearDown(self):self.tmp.cleanup()
 def test_paired_end_to_end(self):
  r=train_run(self.manifest,self.root/'out','paired',10);self.assertEqual(r['after']['metrics']['correct'],9);self.assertTrue(r['speaker_split_verified']);m=VoiceModel.load(self.root/'out/model.json');self.assertIsNone(m.predict(self.root/'data/unknown.wav').get('transcript'))
 def test_voice_only_patterns_and_counts(self):
  r=train_run(self.manifest,self.root/'out','voice-only',5);m=VoiceModel.load(self.root/'out/model.json');self.assertEqual(r['after']['metrics']['statuses']['ACOUSTIC_PATTERN_MATCH'],9);self.assertEqual(m.predict(self.root/'data/unknown.wav')['status'],'UNKNOWN_ACOUSTIC_PATTERN');modes=[x for c in m.engine.contexts.values() for s in c['stores'].values() for x in s['modes']];self.assertEqual(len(modes),3);self.assertEqual(sum(x['count'] for x in modes),60)
 def test_unicode_transcript_and_read_only(self):
  m=VoiceModel();f=self.root/'data/tone_0_0.wav'
  for _ in range(6):m.train(f,'ŋ ɛ ɔ ä')
  snap=m.snapshot();self.assertEqual(m.predict(f)['transcript'],'ŋ ɛ ɔ ä');assert_exact(snap,m.snapshot());m.save(self.root/'unicode.json');assert_exact(snap,VoiceModel.load(self.root/'unicode.json').snapshot())
 def test_resume_bitwise_continuation(self):
  m=VoiceModel();f=self.root/'data/tone_0_0.wav'
  for _ in range(5):m.train(f,'low tone')
  m.save(self.root/'model.json');n=VoiceModel.load(self.root/'model.json')
  for _ in range(5):m.train(f,'low tone');n.train(f,'low tone')
  assert_exact(m.snapshot(),n.snapshot())
 def test_invalid_audio_and_missing_label_do_not_mutate(self):
  m=VoiceModel();snap=m.snapshot();wavfile.write(self.root/'silent.wav',16000,np.zeros(1000,dtype=np.int16))
  with self.assertRaises(ValueError):m.train(self.root/'silent.wav','silence')
  with self.assertRaises(ValueError):m.train(self.root/'data/tone_0_0.wav')
  assert_exact(snap,m.snapshot())
 def test_audio_formats_and_amplitude(self):
  m=VoiceModel();sr,x=wavfile.read(self.root/'data/tone_0_0.wav');x=x.astype(float)/32768;wavfile.write(self.root/'stereo.wav',sr,np.stack([x*.5,x*.5],axis=1).astype(np.float32));a=m.encoder.encode(self.root/'data/tone_0_0.wav');b=m.encoder.encode(self.root/'stereo.wav');self.assertGreater(float(a@b),.999999)
 def test_split_leakage_rejected(self):
  rows=[json.loads(s) for s in self.manifest.read_text().splitlines()];rows[-1]['speaker_id']='training_signal_generator';self.manifest.write_text('\n'.join(json.dumps(r) for r in rows))
  with self.assertRaises(ValueError):read_manifest(self.manifest,'paired')
  rows[-1]['speaker_id']='heldout_signal_generator';rows[-1]['audio']=rows[0]['audio'];self.manifest.write_text('\n'.join(json.dumps(r) for r in rows))
  with self.assertRaises(ValueError):read_manifest(self.manifest,'paired')
 def test_label_capacity_explicit_and_bounds(self):
  m=VoiceModel();m.transcripts={str(i):m.engine.B[0].copy() for i in range(64)};snap=m.snapshot();r=m.train(self.root/'data/tone_0_0.wav','new label');self.assertEqual(r['status'],'TRANSCRIPT_CAPACITY_REJECTED');assert_exact(snap,m.snapshot());m.engine.check_bounds()
if __name__=='__main__':unittest.main()
