"""TBIN v25: CPU paired-observation contrastive multimodal learner.
No semantic labels required; co-occurrence/paired IDs ARE supervision.
Requires numpy, pillow, and sibling tbin_v21.py/tbin_v23_unified.py/tbin_v22_audio.py/tbin_v20.py.
Usage: python tbin_v25.py demo | image PHOTO | audio SONG.wav
"""
import argparse, json, tempfile, wave, time
from pathlib import Path
import numpy as np
from PIL import Image
from tbin_v24 import TBIN24

class TBIN25(TBIN24):
    def __init__(self, seed=7, dimension=24):
        super().__init__(seed)
        if dimension!=self.shared.d:
            raise ValueError('Only dimension=24 supported by this integrated version')
        self.contrastive_history=[]

    def train_unlabeled_pairs(self, pairs, epochs=100, lr=.04, temperature=.15,
                              batch_size=64):
        """Use a restartable callable for streamed data; lists remain supported.
        Optional group IDs mark multiple positives, avoiding same-class negatives.
        Memory for comparisons is O(batch_size**2), not O(dataset_size**2).
        """
        if epochs<1 or batch_size<6 or lr<=0 or temperature<=0:raise ValueError('Invalid training settings')
        if not callable(pairs) and iter(pairs) is pairs:raise ValueError('Use a restartable iterator factory')
        factory=pairs if callable(pairs) else lambda:iter(pairs)
        # Bounded validation pass before changing parameters.
        mods=None;count=0;grouped=None
        for row in factory():
            current=sorted(set(row)&{'image','audio','text'})
            if len(current)<2 or (mods is not None and current!=mods):raise ValueError('Rows must have identical modalities')
            has_group='group' in row
            if grouped is not None and grouped!=has_group:raise ValueError('Use group IDs for all rows or none')
            grouped=has_group;mods=current;count+=1
        if count<3:raise ValueError('At least three observations required')
        if not self.visual_fitted and 'image' in mods:
            for i,row in enumerate(factory()):
                self.shared.visual.observe(row['image'])
                if i>=255:break
            self.visual_fitted=True
        rng=np.random.default_rng(self.shared.rng.integers(2**32))
        history=[];self.training_stats={'pairs':count,'max_batch':0,'max_score_elements':0,'updates':0}
        # Reserving three rows avoids dropping or duplicating the tail.
        def batches():
            buf=[]
            for row in factory():
                buf.append(row)
                if len(buf)==batch_size+3:
                    yield buf[:batch_size];buf=buf[batch_size:]
            if len(buf)>batch_size:
                yield buf[:-3];buf=buf[-3:]
            if buf:yield buf
        for epoch in range(epochs):
            total_loss=0.;steps=0
            for batch in batches():
                rng.shuffle(batch);n=len(batch)
                self.training_stats['max_batch']=max(n,self.training_stats['max_batch'])
                self.training_stats['max_score_elements']=max(n*n,self.training_stats['max_score_elements'])
                features={m:np.stack([self.shared.feature(m,p[m]) for p in batch]) for m in mods}
                labels=[p['group'] for p in batch] if grouped else list(range(n))
                positive=np.array([[x==y for y in labels] for x in labels],np.float32)
                targets=positive/positive.sum(axis=1,keepdims=True)
                if np.all(positive):raise ValueError('Batch has no negative groups; interleave different groups')
                grads={m:np.zeros_like(self.shared.weights[m]) for m in mods}
                for ia,a in enumerate(mods):
                    for b in mods[ia+1:]:
                        xa,xb=features[a],features[b]
                        ya=xa@self.shared.weights[a];yb=xb@self.shared.weights[b]
                        na=np.maximum(np.linalg.norm(ya,axis=1,keepdims=True),1e-8)
                        nb=np.maximum(np.linalg.norm(yb,axis=1,keepdims=True),1e-8)
                        za,zb=ya/na,yb/nb;scores=za@zb.T/temperature
                        def softmax(x):
                            e=np.exp(x-x.max(axis=1,keepdims=True));return e/e.sum(axis=1,keepdims=True)
                        pa,pb=softmax(scores),softmax(scores.T)
                        total_loss+=float(-(targets*np.log(pa+1e-12)).sum()/n/2-(targets*np.log(pb+1e-12)).sum()/n/2);steps+=1
                        g=(pa-targets+(pb-targets).T)/(2*n*temperature)
                        ga=g@zb;gb=g.T@za
                        ga=(ga-za*(ga*za).sum(axis=1,keepdims=True))/na
                        gb=(gb-zb*(gb*zb).sum(axis=1,keepdims=True))/nb
                        grads[a]+=xa.T@ga;grads[b]+=xb.T@gb
                for m in mods:self.shared.weights[m]-=lr*np.clip(grads[m],-5,5)
                self.training_stats['updates']+=1
            history.append({'epoch':epoch+1,'loss':float(total_loss/max(1,steps))})
        self.contrastive_history=history
        return history

    def cross_retrieve(self, source_modality, observation, target_modality, gallery):
        """Retrieve an index in a supplied raw target-modality gallery."""
        if not gallery:raise ValueError('Gallery must not be empty')
        z=self.shared.encode(source_modality,observation)
        scores=[float(z@self.shared.encode(target_modality,item)) for item in gallery]
        return {'index':int(np.argmax(scores)),'scores':scores}

    def save_model(self, filename):
        payload={f'W_{m}':w for m,w in self.shared.weights.items()}
        payload['visual_centers']=np.asarray(self.shared.visual.centers,dtype=np.float32).reshape(-1,8)
        payload['visual_fitted']=np.array(self.visual_fitted)
        np.savez_compressed(filename,**payload)
    def load_model(self,filename):
        with np.load(filename,allow_pickle=False) as data:
            self.shared.visual.centers=list(data['visual_centers'])
            self.shared.visual.counts=[1]*len(self.shared.visual.centers)
            self.visual_fitted=bool(data['visual_fitted'])
            for m in self.shared.weights:
                w=data[f'W_{m}']
                if w.shape!=self.shared.weights[m].shape:raise ValueError('Wrong model shape')
                self.shared.weights[m]=w.astype(np.float32)

def demo():
    start=time.perf_counter();rng=np.random.default_rng(18)
    # Synthetic training: 6 paired concepts, with multiple variations; no labels given to learner.
    colors=np.array([[220,30,30],[30,220,30],[30,30,220],[220,220,30],[220,30,220],[30,220,220]])
    freqs=np.array([220,320,440,570,720,900]);sr=8000
    def img(i,v):
        a=np.broadcast_to(colors[i],(32,32,3)).astype(np.float32).copy()
        a+=rng.normal(0,v,(32,32,3));return np.clip(a,0,255).astype('uint8')
    def aud(i,shift):
        t=np.arange(2048)/sr
        return (.5*np.sin(2*np.pi*(freqs[i]+shift)*t+.1*shift)).astype('float32')
    pairs=[{'image':img(i,3),'audio':aud(i,float(j))} for j in range(3) for i in range(6)]
    model=TBIN25();before=TBIN25()
    held=[(i,img(i,7),aud(i,3.5)) for i in range(6)]
    gallery_img=[x[1] for x in held];gallery_aud=[x[2] for x in held]
    def evaluate(m):
        hits=[]
        for i,im,au in held:
            hits.append(m.cross_retrieve('image',im,'audio',gallery_aud)['index']==i)
            hits.append(m.cross_retrieve('audio',au,'image',gallery_img)['index']==i)
        return sum(hits),len(hits)
    initial=evaluate(before)
    history=model.train_unlabeled_pairs(pairs,epochs=150,lr=.025)
    trained=evaluate(model)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'model.npz';model.save_model(path)
        restored=TBIN25();restored.load_model(path);reload_score=evaluate(restored)
        wav=Path(tmp)/'song.wav';t=np.arange(sr*4)/sr
        with wave.open(str(wav),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr)
            w.writeframes((np.sin(2*np.pi*440*t)*12000).astype('<i2').tobytes())
        song=model.observe_audio(wav)
    image_result=model.observe_image(held[0][1])
    report={'version':25,'method':'paired symmetric contrastive InfoNCE, no class labels','training_pairs':len(pairs),'heldout_queries':trained[1],'baseline_correct':initial[0],'trained_correct':trained[0],'reloaded_correct':reload_score[0], 'loss_history':history,'projection_parameters':int(sum(w.size for w in model.shared.weights.values())),'projection_weight_bytes':int(sum(w.nbytes for w in model.shared.weights.values())),'audio_frames':song['frames'],'image_patches':image_result['patches'],'elapsed_seconds':round(time.perf_counter()-start,3),'limitations':['synthetic tones and flat-color images; not real speech/photos','pairing is supervisory signal, even without semantic labels','no object detection, ASR, singing separation, or semantic reasoning demonstrated','contrastive loss can exploit shortcuts; test with real held-out subjects and environments']}
    Path(__file__).with_name('tbin_v25_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    assert trained[0]>=initial[0], 'training regressed on this synthetic test'
    assert trained==reload_score, 'model reload changed predictions'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['demo','image','audio']);parser.add_argument('path',nargs='?');args=parser.parse_args()
    if args.command=='demo':demo()
    elif not args.path:parser.error('path required')
    elif args.command=='image':
        with Image.open(args.path) as im:print(json.dumps(TBIN25().observe_image(np.asarray(im.convert('RGB'))),indent=2))
    else:
        result=TBIN25().observe_audio(args.path)
        print(json.dumps({k:v for k,v in result.items() if k not in ('phrase_timeline','sections','pattern_transitions')},indent=2))
