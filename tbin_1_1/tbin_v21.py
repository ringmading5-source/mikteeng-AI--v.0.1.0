"""TBIN v21: CPU multimodal research prototype. pip install numpy; python tbin_v21.py
Text: UTF-8 hierarchy; audio: waveform frames/spectral features; image: patch RGB features.
Paired observations supervise a shared trainable concept space. Not ASR/TTS or image generation.
"""
import json, re, time, tracemalloc
from collections import Counter, defaultdict
import numpy as np

class TBINMultimodal:
    def __init__(self, dimension=24, seed=7):
        self.d=dimension; self.rng=np.random.default_rng(seed)
        self.stats={k:defaultdict(Counter) for k in ('characters','words','sentences','passages')}
        self.previous_passage=None
        self.inputs={'text':48,'audio':32,'image':48}
        self.weights={m:self.rng.normal(0,.04,(n,dimension)).astype('float32') for m,n in self.inputs.items()}
        self.prototypes={}; self.counts=Counter();self.feedback=[]
    def observe_text(self, passage):
        sentences=[re.findall(r'[^\W_]+',s.casefold(),re.UNICODE) for s in re.split(r'[.!?\n]+',passage)]
        sentences=[s for s in sentences if s]
        for s in sentences:
            for w in s:
                for a,b in zip(w,w[1:]):self.stats['characters'][a][b]+=1
            for a,b in zip(s,s[1:]):self.stats['words'][a][b]+=1
        for a,b in zip(sentences,sentences[1:]):self.stats['sentences'][' '.join(a)][' '.join(b)]+=1
        if self.previous_passage is not None:self.stats['passages'][self.previous_passage][passage]+=1
        self.previous_passage=passage
    def text_features(self, s):
        """Hash byte ngrams + word ngrams, with no language-specific vocabulary."""
        import zlib
        v=np.zeros(48,np.float32); raw=s.casefold().encode('utf-8')
        for n in (1,2,3):
            for i in range(max(0,len(raw)-n+1)):
                h=zlib.crc32(bytes([n])+raw[i:i+n]);v[h%48]+=1 if h&0x100 else -1
        return v/(np.linalg.norm(v)+1e-8)
    def audio_features(self, waveform, sample_rate=8000):
        """Variable-length mono waveform to 32 pooled spectral/temporal features."""
        a=np.asarray(waveform,dtype=np.float32).reshape(-1)
        if len(a)<64:raise ValueError('Audio needs >=64 samples')
        if not np.all(np.isfinite(a)):raise ValueError('Nonfinite audio')
        a=a/(max(1.,float(np.max(np.abs(a)))))
        frame=256;hop=128; frames=[]
        for i in range(0,max(1,len(a)-frame+1),hop):
            chunk=a[i:i+frame]
            if len(chunk)<frame:chunk=np.pad(chunk,(0,frame-len(chunk)))
            spec=np.abs(np.fft.rfft(chunk*np.hanning(frame)))
            bins=np.array([np.mean(x) for x in np.array_split(spec,24)],np.float32)
            frames.append(np.log1p(bins))
        z=np.asarray(frames);v=np.r_[z.mean(axis=0),z.std(axis=0)[:4],float(np.sqrt(np.mean(a*a))),float(np.mean(np.abs(np.diff(a)))),float(np.mean(a>0)),float(np.mean(a<0))].astype('float32')
        return v/(np.linalg.norm(v)+1e-8)
    def image_features(self, image):
        """RGB or grayscale image to 4x4 spatial mean-color patches (48 values)."""
        a=np.asarray(image,dtype=np.float32)
        if a.ndim==2:a=np.repeat(a[:,:,None],3,axis=2)
        if a.ndim!=3 or a.shape[2] not in (3,4) or min(a.shape[:2])<4:raise ValueError('Expected image HxWx3, >=4x4')
        a=a[:,:,:3];a=a/(255. if a.max()>1 else 1.)
        patches=[a[y0:y1,x0:x1].mean((0,1)) for y0,y1 in zip(np.linspace(0,a.shape[0],5,dtype=int)[:-1],np.linspace(0,a.shape[0],5,dtype=int)[1:]) for x0,x1 in zip(np.linspace(0,a.shape[1],5,dtype=int)[:-1],np.linspace(0,a.shape[1],5,dtype=int)[1:])]
        v=np.asarray(patches,dtype=np.float32).flatten();v=v/(np.linalg.norm(v)+1e-8)
        if hasattr(self,'visual'):
            hist=np.zeros(self.visual.max_patterns,np.float32)
            for y in range(0,a.shape[0],self.visual.patch):
                for x in range(0,a.shape[1],self.visual.patch):
                    tile=a[y:y+self.visual.patch,x:x+self.visual.patch]
                    descriptor=self.visual.descriptor(tile)
                    if self.visual.centers:
                        k=int(np.argmin([np.linalg.norm(descriptor-c) for c in self.visual.centers]))
                        hist[k]+=1
            hist/=np.linalg.norm(hist)+1e-8
            v=np.r_[v,hist].astype(np.float32)
        return v/(np.linalg.norm(v)+1e-8)
    def feature(self,modality,value):
        return {'text':self.text_features,'audio':self.audio_features,'image':self.image_features}[modality](value)
    def encode(self,modality,value):
        x=self.feature(modality,value);z=x@self.weights[modality];return z/(np.linalg.norm(z)+1e-8)
    def fit(self, examples, epochs=150, lr=.16):
        """Paired/triplet supervised alignment. Each example: (concept_id, {modality:raw}).
        Updates all modality projections toward the mean of their paired concept vectors.
        """
        samples=[(label,{m:self.feature(m,v) for m,v in obs.items()}) for label,obs in examples]
        for _ in range(epochs):
            sums=defaultdict(lambda:np.zeros(self.d,np.float32));counts=Counter()
            for label,obs in samples:
                for m,x in obs.items():
                    z=x@self.weights[m];z=z/(np.linalg.norm(z)+1e-8)
                    sums[label]+=z;counts[label]+=1
            targets={k:v/(np.linalg.norm(v)+1e-8) for k,v in sums.items()}
            for label,obs in samples:
                for m,x in obs.items():
                    z=x@self.weights[m];delta=targets[label]-z
                    self.weights[m]+=lr*np.outer(x,delta).astype('float32')
        # Recompute prototypes from precomputed training features (no raw examples retained).
        accum=defaultdict(lambda:np.zeros(self.d,np.float32));count=Counter()
        for label,obs in samples:
            for m,x in obs.items():
                z=x@self.weights[m];z=z/(np.linalg.norm(z)+1e-8);accum[label]+=z;count[label]+=1
        self.prototypes={k:v/(np.linalg.norm(v)+1e-8) for k,v in accum.items()};self.counts=count
    def predict(self, modality, value):
        z=self.encode(modality,value)
        scores=sorted(((float(z@p),label) for label,p in self.prototypes.items()),reverse=True)
        return {'concept':scores[0][1],'similarity':round(scores[0][0],4),'scores':{k:round(s,4) for s,k in scores}}
    def apply(self, modality,value,knowledge):
        p=self.predict(modality,value);return {'concept':p['concept'],'answer':knowledge.get(p['concept']),'similarity':p['similarity']}
    def correct(self,modality,value,label,lr=.3):
        """Online supervised feedback on concept representation."""
        x=self.feature(modality,value)
        if label not in self.prototypes:raise KeyError('Unknown concept label')
        self.weights[modality]+=lr*np.outer(x,self.prototypes[label]-x@self.weights[modality]).astype('float32')
        self.feedback.append((modality,label))
    def size_bytes(self):
        return sum(w.nbytes for w in self.weights.values())+sum(v.nbytes for v in self.prototypes.values())

def demo():
    t=time.perf_counter();tracemalloc.start();model=TBINMultimodal()
    # Three distinct toy classes; signals are correlated by construction, not real-world semantics.
    labels=['red','green','blue'];colors=[(255,30,30),(30,255,30),(30,30,255)];freqs=[240,460,720]
    def make_image(i,variant=0):
        a=np.zeros((16,16,3),np.float32);a[:]=colors[i];a+=variant*2;return np.clip(a,0,255).astype('uint8')
    def make_audio(i,variant=0):
        x=np.arange(2048)/8000;return np.sin(2*np.pi*(freqs[i]+variant)*x).astype('float32')
    examples=[]
    for i,label in enumerate(labels):
        for variant in range(4):
            phrase=f'{label} object sample {variant}'
            model.observe_text(phrase+'. '+label+' object appears.')
            examples.append((label,{'text':phrase,'image':make_image(i,variant),'audio':make_audio(i,variant)}))
    model.fit(examples)
    held=[]
    for i,label in enumerate(labels):
        for modality,value in [('text',f'{label} object sample 7'),('image',make_image(i,7)),('audio',make_audio(i,7))]:
            p=model.predict(modality,value);held.append({'modality':modality,'expected':label,'predicted':p['concept'],'correct':p['concept']==label})
    knowledge={'red':'warm color example','green':'middle color example','blue':'cool color example'}
    applications=[model.apply('image',make_image(i,7),knowledge) for i in range(3)]
    current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    result={'name':'TBIN v21 multimodal proof of concept','training_triplets':len(examples),'held_out':held,'held_out_accuracy':sum(x['correct'] for x in held)/len(held),'modalities':['text','audio','image'],'neural_projection_parameters':sum(w.size for w in model.weights.values()),'weights_and_concepts_bytes':model.size_bytes(),'peak_python_traced_bytes':peak,'duration_seconds':round(time.perf_counter()-t,4),'applications':applications,'limitations':['synthetic paired data with predefined labels','simple engineered features; no speech transcription or image generation','same labels and narrow data distribution in training and testing','not evidence of real-world multimodal understanding']}
    with open('/mnt/data/tbin_v21_results.json','w') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':demo()
