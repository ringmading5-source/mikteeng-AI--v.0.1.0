import json,os,tempfile
from pathlib import Path
import numpy as np
try:
 from .engine import BoundedRSPM,assert_exact
 from .checkpoint import pack,unpack
 from .text_adapter import encode_text,metadata
 from .audio_encoder import AudioEncoder
except ImportError:
 from engine import BoundedRSPM,assert_exact
 from checkpoint import pack,unpack
 from text_adapter import encode_text,metadata
 from audio_encoder import AudioEncoder
class VoiceModel:
 def __init__(self,mode='paired',seed=42):
  if mode not in ['paired','voice-only']:raise ValueError('paired or voice-only mode required')
  self.mode=mode;self.engine=BoundedRSPM(dim=256,seed=seed,max_contexts=2,max_triggers=64,max_modes=16,max_candidates=16,evidence=3,ttl=100000)
  self.engine.B=np.ascontiguousarray(self.engine.B);self.engine.B.flags.writeable=False
  self.encoder=AudioEncoder(self.engine.B);self.transcripts={}
 def train(self,audio,transcript=None):
  x=self.encoder.encode(audio)
  if self.mode=='voice-only':return self.engine.train(self.engine.B[244],self.engine.B[245],x)
  if not isinstance(transcript,str) or not transcript.strip():raise ValueError('nonempty transcript required for paired training')
  y=encode_text(transcript,self.engine.B)
  if transcript not in self.transcripts and len(self.transcripts)>=64:return dict(status='TRANSCRIPT_CAPACITY_REJECTED')
  result=self.engine.train(self.engine.B[243],x,y)
  if result['status']=='TRAINED':self.transcripts[transcript]=y.copy()
  return result
 def predict(self,audio):
  x=self.encoder.encode(audio)
  if self.mode=='voice-only':
   q=self.engine.query_strict(self.engine.B[244],self.engine.B[245]);modes=q.get('hypotheses',[])
   if not modes:return dict(status='NO_CONFIRMED_PATTERNS',mode=self.mode)
   scores=np.array([v@x for v in modes]);i=int(np.argmax(scores));matched=1-scores[i]<=self.engine.config['tol']
   # Pattern index is checkpoint-local, not a word, meaning, or stable global ID.
   return dict(status='ACOUSTIC_PATTERN_MATCH' if matched else 'UNKNOWN_ACOUSTIC_PATTERN',mode=self.mode,pattern_index=i if matched else None,similarity=float(scores[i]),confirmed_patterns=len(modes),confidence='UNVERIFIED')
  q=self.engine.query(self.engine.B[243],x)
  if 'parametric' not in q or not self.transcripts:return dict(status=q['status'],mode=self.mode,transcript=None)
  scores=sorted([(float(y@q['parametric']),text) for text,y in self.transcripts.items()],reverse=True)
  score,text=scores[0];margin=score-(scores[1][0] if len(scores)>1 else 0.)
  if score<.6 or margin<.1:return dict(status='UNCERTAIN_TRANSCRIPT',mode=self.mode,transcript=None,score=score,margin=margin)
  return dict(status='KNOWN_TRANSCRIPT_PREDICTION',mode=self.mode,transcript=text,score=score,margin=margin,confidence='UNVERIFIED',decoding='nearest known transcript vector',routing_status=q['status'])
 def snapshot(self):return dict(mode=self.mode,engine=self.engine.snapshot(),transcripts={k:v.copy() for k,v in self.transcripts.items()},audio=self.encoder.metadata(),text=metadata())
 def save(self,path):
  p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);name=None
  try:
   with tempfile.NamedTemporaryFile('w',dir=p.parent,delete=False,encoding='utf-8') as f:
    name=f.name;json.dump(dict(format='mikteeng-rspm-voice-v1',state=pack(self.snapshot())),f,allow_nan=False,ensure_ascii=False);f.flush();os.fsync(f.fileno())
   os.replace(name,p)
  finally:
   if name and os.path.exists(name):os.unlink(name)
 @classmethod
 def load(cls,path):
  p=Path(path)
  if p.stat().st_size>64*1024*1024:raise ValueError('checkpoint exceeds 64 MiB')
  data=json.loads(p.read_text(encoding='utf-8'))
  if data.get('format')!='mikteeng-rspm-voice-v1':raise ValueError('invalid voice checkpoint format')
  s=unpack(data['state']);m=cls(s['mode']);m.engine.restore(s['engine']);m.engine.check_bounds()
  if m.engine.config['dim']!=256 or m.engine.B.flags.writeable or m.engine.B.shape!=(256,256) or not np.isfinite(m.engine.B).all() or not np.allclose(m.engine.B@m.engine.B.T,np.eye(256),rtol=0,atol=1e-8):raise ValueError('invalid fixed basis')
  if len(s['transcripts'])>64:raise ValueError('too many transcripts')
  for text,y in s['transcripts'].items():
   if not isinstance(text,str) or y.shape!=(256,) or not np.isfinite(y).all():raise ValueError('invalid transcript target')
  if s['audio']!=m.encoder.metadata() or s['text']!=metadata():raise ValueError('incompatible audio/text encoder version')
  m.transcripts={k:v.copy() for k,v in s['transcripts'].items()};m.encoder=AudioEncoder(m.engine.B);return m
