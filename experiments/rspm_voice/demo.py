"""Generated tones exercise I/O and learning; not human voice or a Dinka benchmark."""
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
try:
 from .cli import train_run
 from .voice_model import VoiceModel
except ImportError:
 from cli import train_run
 from voice_model import VoiceModel
P=Path(__file__).resolve().parent

def make_demo(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);rows=[]
 for k,(freq,label) in enumerate([(300,'low tone'),(900,'middle tone'),(1800,'high tone')]):
  for j in range(7):
   sr=16000 if j<4 else 22050;duration=.5+.04*j;t=np.arange(int(sr*duration))/sr
   x=.6*np.sin(2*np.pi*(freq+(j-3)*2)*t+j*.3)
   name=f'tone_{k}_{j}.wav';wavfile.write(root/name,sr,(x*32767).astype(np.int16));rows.append(dict(audio=name,transcript=label,split='train' if j<4 else 'test',speaker_id='training_signal_generator' if j<4 else 'heldout_signal_generator'))
 (root/'manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
 t=np.arange(8000)/16000;wavfile.write(root/'unknown.wav',16000,(.5*np.sin(2*np.pi*4200*t)*32767).astype(np.int16))
 return root/'manifest.jsonl'
def main():
 manifest=make_demo(P/'demo_data');summary={}
 for mode in ['paired','voice-only']:
  report=train_run(manifest,P/f'demo_{mode}',mode,20);model=VoiceModel.load(P/f'demo_{mode}'/'model.json');summary[mode]=dict(before=report['before']['metrics'],after=report['after']['metrics'],unknown=model.predict(P/'demo_data'/'unknown.wav'))
 (P/'demo_results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
