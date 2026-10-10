"""TBIN v26: train/evaluate v25 on real paired image/audio files.
Manifest JSONL: {"image":"path.jpg","audio":"path.wav","group":"cow_01"}
Each group must occur in only one split. CPU-only; WAV PCM input.
Usage: python tbin_v26.py demo
       python tbin_v26.py train pairs.jsonl --out model.npz --report report.json
"""
import argparse, json, os, tempfile, time, wave
from pathlib import Path
import numpy as np
from PIL import Image
from tbin_v25 import TBIN25

def load_audio(path, max_seconds=3.0):
    with wave.open(str(path),'rb') as f:
        sr=f.getframerate(); channels=f.getnchannels(); sw=f.getsampwidth()
        if sw!=2: raise ValueError('Only 16-bit PCM WAV supported: '+str(path))
        n=min(f.getnframes(),int(sr*max_seconds))
        raw=np.frombuffer(f.readframes(n),dtype='<i2').astype(np.float32)/32768
        if channels>1: raw=raw.reshape(-1,channels).mean(axis=1)
        if raw.size==0:raise ValueError('Empty audio '+str(path))
        # Preserve time: resample to 8 kHz, never squeeze clips to a fixed length.
        from scipy.signal import resample_poly
        from math import gcd
        g=gcd(sr,8000)
        result=resample_poly(raw,8000//g,sr//g).astype(np.float32)
        return np.pad(result,(0,max(0,64-len(result))))

def load_manifest(path):
    base=Path(path).resolve().parent; rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if not all(k in row for k in ('image','audio','group')):raise ValueError('Each row requires image,audio,group')
        image=Path(row['image']);audio=Path(row['audio'])
        if not image.is_absolute():image=base/image
        if not audio.is_absolute():audio=base/audio
        with Image.open(image) as im:
            rgb=np.asarray(im.convert('RGB').resize((32,32)),dtype=np.uint8)
        rows.append({'image':rgb,'audio':load_audio(audio),'group':str(row['group'])})
    return rows

def split_groups(rows,seed=42,train_fraction=.7):
    groups=sorted({r['group'] for r in rows})
    if len(groups)<4:raise ValueError('Need at least 4 distinct groups for group-held-out evaluation')
    rng=np.random.default_rng(seed);rng.shuffle(groups)
    n=max(2,min(len(groups)-2,int(len(groups)*train_fraction)))
    train_groups=set(groups[:n]);train=[r for r in rows if r['group'] in train_groups]
    test=[r for r in rows if r['group'] not in train_groups]
    if len(train)<3 or len(test)<2:raise ValueError('Not enough training or test pairs')
    return train,test

def evaluate(model,rows):
    # One candidate per group, no duplicate-group ambiguity; gallery uses same held-out
    # examples as queries: tests cross-modal matching on unseen groups, not unseen recordings.
    first={}
    for r in rows:first.setdefault(r['group'],r)
    samples=list(first.values()); correct=0;total=0
    for i,r in enumerate(samples):
        for src,dst in [('image','audio'),('audio','image')]:
            gallery=[x[dst] for x in samples]
            result=model.cross_retrieve(src,r[src],dst,gallery)
            correct+=int(result['index']==i);total+=1
    return {'correct':correct,'total':total,'accuracy':round(correct/total,4),'gallery_size':len(samples)}

def run(rows,epochs=100,seed=42):
    train,test=split_groups(rows,seed)
    before=TBIN25(seed=seed);after=TBIN25(seed=seed)
    baseline=evaluate(before,test)
    t=time.perf_counter();history=after.train_unlabeled_pairs([{k:r[k] for k in ('image','audio')} for r in train],epochs=epochs,lr=.025)
    duration=time.perf_counter()-t
    result=evaluate(after,test)
    return after,{'train_pairs':len(train),'test_pairs':len(test),'train_groups':len({r['group'] for r in train}),'test_groups':len({r['group'] for r in test}),'baseline':baseline,'trained':result,'loss_history':history,'training_seconds':round(duration,3),'parameter_bytes':sum(w.nbytes for w in after.shared.weights.values()),'warning':'Held-out groups, but gallery uses matched examples; real-world robustness and semantics unproven.'}

def demo():
    rng=np.random.default_rng(4); rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        for i in range(8):
            for j in range(3):
                color=np.array([(i*41)%240+10,(i*67)%240+10,(i*91)%240+10])
                pixels=np.clip(color+rng.normal(0,3,(32,32,3)),0,255).astype('uint8')
                image=root/f'im_{i}_{j}.png';Image.fromarray(pixels).save(image)
                sr=8000;t=np.arange(4096)/sr
                samples=(np.sin(2*np.pi*(180+i*75+j*.5)*t)*10000).astype('<i2')
                audio=root/f'au_{i}_{j}.wav'
                with wave.open(str(audio),'wb') as w:
                    w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes(samples.tobytes())
                rows.append({'image':image.name,'audio':audio.name,'group':f'group_{i}'})
        manifest=root/'pairs.jsonl';manifest.write_text('\n'.join(json.dumps(r) for r in rows))
        data=load_manifest(manifest)
        model,report=run(data,epochs=80)
        path=root/'weights.npz';model.save_model(path)
        restored=TBIN25(seed=42);restored.load_model(path)
        _,test=split_groups(data,42)
        report['reload_equal']=evaluate(restored,test)==report['trained']
        report['test_type']='synthetic generated images/tones; grouped holdout'
        Path('/mnt/data/tbin_v26_results.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        assert report['reload_equal']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['demo','train']);p.add_argument('manifest',nargs='?');p.add_argument('--out',default='tbin_weights.npz');p.add_argument('--report',default='tbin_report.json');p.add_argument('--epochs',type=int,default=100);args=p.parse_args()
    if args.command=='demo':demo()
    else:
        if not args.manifest:p.error('train requires manifest')
        model,report=run(load_manifest(args.manifest),epochs=args.epochs)
        model.save_model(args.out);Path(args.report).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
