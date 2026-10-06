"""Validate -> train -> select by validation -> evaluate test once. No auto deployment."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
try:
 from .voice_model import VoiceModel
 from .cli import evaluate
 from .text_adapter import encode_text
except ImportError:
 from voice_model import VoiceModel
 from cli import evaluate
 from text_adapter import encode_text

def prepare_manifest(path,mode):
 p=Path(path).resolve();splits={k:[] for k in ['train','validation','test']};digests={};speakers={}
 if p.stat().st_size>8*1024*1024:raise ValueError('manifest exceeds8 MiB')
 for line in p.read_text(encoding='utf-8').splitlines():
  if not line.strip():continue
  r=json.loads(line);split=r.get('split');f=(p.parent/r['audio']).resolve()
  if split not in splits or not f.is_file() or not f.is_relative_to(p.parent):raise ValueError('valid split and contained WAV path required')
  if f.stat().st_size>64*1024*1024:raise ValueError('WAV too large')
  digest=hashlib.sha256(f.read_bytes()).hexdigest()
  if digest in digests:raise ValueError('duplicate audio in dataset')
  digests[digest]=split
  if mode=='paired' and (not isinstance(r.get('transcript'),str) or not r['transcript'].strip() or len(r['transcript'])>4096):raise ValueError('nonempty transcript <=4096 characters required')
  if 'speaker_id' in r:
   s=r['speaker_id']
   if not isinstance(s,str) or not s or (s in speakers and speakers[s]!=split):raise ValueError('speakers must be disjoint across splits')
   speakers[s]=split
  splits[split].append(dict(r,path=str(f),digest=digest))
  if len(digests)>10000:raise ValueError('at most10000 rows')
 rows=sum(splits.values(),[])
 if not all(splits.values()):raise ValueError('all train/validation/test splits required')
 if speakers and not all('speaker_id' in r for r in rows):raise ValueError('speaker_id required on every row when supplied')
 return splits,rows,bool(speakers)

def run(manifest,out,mode='paired',epochs=20,patience=5,resume=None):
 if mode not in ['paired','voice-only'] or not 1<=epochs<=1000 or not 1<=patience<=1000:raise ValueError('invalid mode/epochs/patience')
 splits,rows,speaker_verified=prepare_manifest(manifest,mode);m=VoiceModel.load(resume) if resume else VoiceModel(mode)
 if m.mode!=mode:raise ValueError('resume mode mismatch')
 for r in rows:
  m.encoder.encode(r['path'])
  if mode=='paired':encode_text(r['transcript'],m.engine.B)
 labels={r['transcript'] for r in splits['train']} if mode=='paired' else set()
 if len(labels|set(m.transcripts))>64:raise ValueError('transcript capacity64 exceeded')
 dest=Path(out);dest.mkdir(parents=True,exist_ok=False);(dest/'dataset_audit.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='path'} for r in rows],ensure_ascii=False,indent=2))
 def score(v):
  q=v['metrics'];return (q['correct'] if mode=='paired' else q['statuses'].get('ACOUSTIC_PATTERN_MATCH',0))/q['cases']
 baseline=evaluate(m,splits['validation']);best=score(baseline);best_epoch=0;stale=0;history=[dict(epoch=0,score=best,validation=baseline)];m.save(dest/'best_model.json');rng=np.random.default_rng(42);events=Counter()
 for epoch in range(1,epochs+1):
  for i in rng.permutation(len(splits['train'])):
   r=splits['train'][i];z=m.train(r['path'],r.get('transcript'));events[z.get('event',z['status'])]+=1
  m.engine.check_bounds();v=evaluate(m,splits['validation']);s=score(v);history.append(dict(epoch=epoch,score=s,validation=v));m.save(dest/'last_model.json')
  if s>best:best=s;best_epoch=epoch;stale=0;m.save(dest/'best_model.json')
  else:stale+=1
  (dest/'validation_history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2))
  if stale>=patience:break
 selected=VoiceModel.load(dest/'best_model.json');test=evaluate(selected,splits['test'])
 report=dict(mode=mode,split_counts={k:len(v) for k,v in splits.items()},speaker_split_verified=speaker_verified,selected_epoch=best_epoch,completed_epochs=history[-1]['epoch'],validation_score=best,test_evaluation_calls=1,test=test,training_events=dict(events),selection_metric='known-transcript accuracy' if mode=='paired' else 'acoustic match coverage, not semantic accuracy',checkpoint='best_model.json',limitations='Known-transcript classifier/acoustic clustering. No free-form ASR or auto deployment.')
 (dest/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--mode',choices=['paired','voice-only'],default='paired');p.add_argument('--epochs',type=int,default=20);p.add_argument('--patience',type=int,default=5);p.add_argument('--resume');a=p.parse_args();print(json.dumps(run(a.manifest,a.out,a.mode,a.epochs,a.patience,a.resume),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
