"""Bounded WAV -> spectral summary -> fixed-basis RSPM vector. Not a speech recognizer."""
from pathlib import Path
from math import gcd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
class AudioEncoder:
 def __init__(self,basis,sample_rate=16000,max_seconds=60):
  if basis.shape!=(256,256):raise ValueError('256-dimensional basis required')
  self.basis=basis;self.sample_rate=sample_rate;self.max_seconds=max_seconds
 def read(self,path):
  p=Path(path)
  if p.stat().st_size>64*1024*1024:raise ValueError('WAV exceeds 64 MiB')
  sr,x=wavfile.read(p)
  if not 8000<=sr<=96000:raise ValueError('WAV sample rate must be 8–96 kHz')
  if x.ndim not in [1,2] or not len(x) or len(x)>sr*self.max_seconds:raise ValueError('nonempty WAV no longer than 60 seconds required')
  if x.ndim==2 and x.shape[1]>8:raise ValueError('at most eight audio channels')
  if np.issubdtype(x.dtype,np.integer):
   if x.dtype==np.uint8:x=(x.astype(float)-128)/128
   else:x=x.astype(float)/max(abs(np.iinfo(x.dtype).min),np.iinfo(x.dtype).max)
  elif np.issubdtype(x.dtype,np.floating):x=x.astype(float)
  else:raise ValueError('unsupported WAV sample type')
  if not np.isfinite(x).all():raise ValueError('audio must be finite')
  if x.ndim==2:x=x.mean(axis=1)
  x=x-x.mean()
  if np.sqrt(np.mean(x*x))<1e-8:raise ValueError('silent or constant audio')
  g=gcd(sr,self.sample_rate);x=resample_poly(x,self.sample_rate//g,sr//g)
  return x
 def encode(self,path):
  x=self.read(path);x/=max(np.max(np.abs(x)),1e-12)
  frame=640;hop=320
  if len(x)<frame:x=np.pad(x,(0,frame-len(x)))
  frames=np.lib.stride_tricks.sliding_window_view(x,frame)[::hop]
  spectrum=np.abs(np.fft.rfft(frames*np.hanning(frame),n=1024,axis=1))**2
  # Fixed 64 linear frequency bands. Summary discards detailed phoneme order.
  edges=np.linspace(0,spectrum.shape[1],65,dtype=int)
  power=np.stack([spectrum[:,edges[i]:edges[i+1]].sum(axis=1) for i in range(64)],axis=1)
  power/=np.maximum(power.sum(axis=1,keepdims=True),1e-12);power=np.sqrt(power)
  half=max(1,len(power)//2)
  raw=np.concatenate([power.mean(axis=0),power.std(axis=0),power[:half].mean(axis=0),power[half:].mean(axis=0) if half<len(power) else power.mean(axis=0)])
  vector=raw@self.basis
  return vector/np.linalg.norm(vector)
 def metadata(self):return dict(version='spectral-summary-v1',sample_rate=self.sample_rate,max_seconds=self.max_seconds,bands=64,frame_samples=640,hop_samples=320,fft_size=1024)
