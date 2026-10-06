import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
try:
 from .voice_model import VoiceModel
 from .engine import assert_exact
except ImportError:
 from voice_model import VoiceModel
 from engine import assert_exact

def read_manifest(path,mode):
 p=Path(path).resolve();rows=[]
 for line_no,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
  if not line.strip():continue
  r=json.loads(line);audio=(p.parent/r['audio']).resolve()
  if not audio.is_relative_to(p.parent) or not audio.is_file():raise ValueError(f'line {line_no}: WAV must exist beneath manifest directory')
  if r.get('split') not in ['train','test']:raise ValueError('each row needs train/test split')
  if mode=='paired' and (not isinstance(r.get('transcript'),str) or not r['transcript'].strip()):raise ValueError('paired rows require transcripts')
  rows.append(dict(r,path=str(audio),digest=hashlib.sha256(audio.read_bytes()).hexdigest()))
 if not rows or len(rows)>10000:raise ValueError('manifest requires 1–10000 rows')
 train=[r for r in rows if r['split']=='train'];test=[r for r in rows if r['split']=='test']
 if not train:raise ValueError('at least one training row required')
 if {r['digest'] for r in train}&{r['digest'] for r in test}:raise ValueError('identical audio present in training and test')
 if test and any('speaker_id' in r for r in rows):
  if not all(isinstance(r.get('speaker_id'),str) and r['speaker_id'] for r in rows):raise ValueError('speaker_id required for every row when speaker split is supplied')
  if {r['speaker_id'] for r in train}&{r['speaker_id'] for r in test}:raise ValueError('test speakers must be excluded from training')
 return train,test

def evaluate(model,rows):
 snap=model.snapshot();preds=[dict(audio=r['audio'],expected=r.get('transcript'),prediction=model.predict(r['path'])) for r in rows];assert_exact(snap,model.snapshot());metrics=dict(cases=len(rows),statuses=dict(Counter(r['prediction']['status'] for r in preds)))
 if model.mode=='paired':metrics.update(correct=sum(r['prediction'].get('transcript')==r['expected'] for r in preds),wrong=sum(r['prediction'].get('transcript') is not None and r['prediction']['transcript']!=r['expected'] for r in preds),rejected=sum(r['prediction'].get('transcript') is None for r in preds))
 return dict(metrics=metrics,predictions=preds)

def train_run(manifest,out,mode,epochs,resume=None):
 if not 1<=epochs<=1000:raise ValueError('epochs must be1–1000')
 train,test=read_manifest(manifest,mode);model=VoiceModel.load(resume) if resume else VoiceModel(mode)
 if model.mode!=mode:raise ValueError('resume mode must match requested mode')
 # Validate all waveforms before making any training updates.
 for r in train+test:model.encoder.encode(r['path'])
 before=evaluate(model,test);events=Counter();rng=np.random.default_rng(42)
 for _ in range(epochs):
  for i in rng.permutation(len(train)):
   r=train[i];result=model.train(r['path'],r.get('transcript'));events[result.get('event',result['status'])]+=1
 model.engine.check_bounds();after=evaluate(model,test);dest=Path(out);dest.mkdir(parents=True,exist_ok=True);model.save(dest/'model.json');assert_exact(model.snapshot(),VoiceModel.load(dest/'model.json').snapshot())
 report=dict(mode=mode,training_recordings=len(train),test_recordings=len(test),epochs=epochs,speaker_split_verified=bool(test and all('speaker_id' in r for r in train+test)),before=before,after=after,events=dict(events),limitations='Known-transcript classification in paired mode; acoustic pattern clustering in voice-only mode. Not free-form ASR or Dinka speech validation.')
 (dest/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report

def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
 t=sub.add_parser('train');t.add_argument('--manifest',required=True);t.add_argument('--out',required=True);t.add_argument('--mode',choices=['paired','voice-only'],default='paired');t.add_argument('--epochs',type=int,default=20);t.add_argument('--resume')
 p=sub.add_parser('predict');p.add_argument('--checkpoint',required=True);p.add_argument('--audio',required=True)
 args=ap.parse_args()
 if args.command=='train':result=train_run(args.manifest,args.out,args.mode,args.epochs,args.resume)
 else:result=VoiceModel.load(args.checkpoint).predict(args.audio)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
