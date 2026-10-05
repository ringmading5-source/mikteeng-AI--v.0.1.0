import unittest
import numpy as np
from mikteeng_ai import AcousticEncoder,SpeechPredictionModel
class SpeechTests(unittest.TestCase):
 def test_reconstruction(self):
  e=AcousticEncoder(frame_size=128);a=np.random.default_rng(5).uniform(-1,1,513)
  v,n=e.encode(a,16000);np.testing.assert_allclose(e.decode(v,n),a,atol=1e-14)
 def test_silence(self):
  e=AcousticEncoder();v,n=e.encode(np.zeros(1000),16000)
  np.testing.assert_array_equal(e.decode(v,n),np.zeros(1000))
 def test_pcm_scaling(self):
  e=AcousticEncoder();v,n=e.encode(np.array([-32768,0,32767],dtype=np.int16),16000)
  np.testing.assert_allclose(e.decode(v,n),[-1,0,32767/32768],atol=1e-14)
 def test_invalid_input(self):
  e=AcousticEncoder()
  for a,rate in [(np.zeros(10),8000),(np.zeros((10,2)),16000),(np.array([np.nan]),16000),(np.array([2.]),16000)]:
   with self.assertRaises(ValueError):e.encode(a,rate)
 def test_training_duplicate_rejection(self):
  m=SpeechPredictionModel(frame_size=32);a=np.sin(np.arange(400))*.2
  with self.assertRaises(ValueError):m.fit([(a,16000)],self_recordings=[(a,16000)])
