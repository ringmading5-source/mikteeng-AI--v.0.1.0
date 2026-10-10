"""TBIN v23: unified CPU research prototype: text, raw image pixels, streaming PCM WAV,
shared supervised cross-modal concepts, and optional v20 relational neural learner.
Dependencies: numpy, pillow. No claim of autonomous semantic discovery or ASR.
Usage: python tbin_v23_unified.py demo
       python tbin_v23_unified.py image photo.jpg
       python tbin_v23_unified.py audio song.wav
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from tbin_v21 import TBINMultimodal
from tbin_v20 import TBIN as RelationalNeuralTBIN
from tbin_v22_audio import analyze_wav

class PixelHierarchy:
    """Online learned patch prototypes -> patch grid -> coarse regional summaries.
    Patch prototypes represent appearances, not automatically recognized objects.
    """
    def __init__(self, patch=8, max_patterns=64, threshold=.16):
        self.patch=patch; self.max_patterns=max_patterns;self.threshold=threshold
        self.centers=[];self.counts=[];self.transitions=Counter()
    @staticmethod
    def descriptor(tile):
        gx=np.abs(np.diff(tile,axis=1)).mean() if tile.shape[1]>1 else 0.
        gy=np.abs(np.diff(tile,axis=0)).mean() if tile.shape[0]>1 else 0.
        return np.r_[tile.mean((0,1)),tile.std((0,1)),gx,gy].astype(np.float32)
    def _assign(self,v):
        if not self.centers:
            self.centers.append(v.copy());self.counts.append(1);return 0
        distances=[float(np.linalg.norm(v-c)) for c in self.centers]
        j=int(np.argmin(distances))
        if distances[j]>self.threshold and len(self.centers)<self.max_patterns:
            j=len(self.centers);self.centers.append(v.copy());self.counts.append(1)
        else:
            self.counts[j]+=1
            rate=min(.05,1/self.counts[j]);self.centers[j]=(1-rate)*self.centers[j]+rate*v
        return j
    def observe(self,image):
        a=np.asarray(image)
        if a.ndim==2:a=np.repeat(a[:,:,None],3,axis=2)
        if a.ndim!=3 or a.shape[2] not in (3,4):raise ValueError('Expected RGB or grayscale image')
        a=a[:,:,:3].astype(np.float32)
        if a.max()>1:a/=255.
        h,w=a.shape[:2];p=self.patch
        grid=[];last=None
        for y in range(0,h,p):
            row=[]
            for x in range(0,w,p):
                tile=a[y:min(y+p,h),x:min(x+p,w)]
                # Mean RGB, channel variation, and local horizontal/vertical gradients.
                mean=tile.mean((0,1));std=tile.std((0,1))
                gx=np.abs(np.diff(tile,axis=1)).mean() if tile.shape[1]>1 else 0.
                gy=np.abs(np.diff(tile,axis=0)).mean() if tile.shape[0]>1 else 0.
                v=self.descriptor(tile)
                k=self._assign(v);row.append(k)
                if last is not None:self.transitions[(last,k)]+=1
                last=k
            grid.append(row)
        # Group patch IDs into coarse regions; retains spatial hierarchy without claiming object detection.
        regions=[]
        for y in range(0,len(grid),4):
            for x in range(0,len(grid[0]),4):
                ids=[n for row in grid[y:y+4] for n in row[x:x+4]]
                regions.append({'dominant_pattern':Counter(ids).most_common(1)[0][0], 'patches':len(ids)})
        return {'width':w,'height':h,'patches':sum(map(len,grid)),'learned_pixel_patterns':len(self.centers),
                'regions':regions,'pattern_grid':grid,'note':'Appearance prototypes and spatial grouping, not semantic object recognition.'}

class UnifiedTBIN:
    def __init__(self,seed=7):
        self.visual=PixelHierarchy();self.shared=TBINMultimodal(seed=seed)
        # Retrieval codebook is frozen after warmup; observation UI can keep learning.
        self.shared.visual=PixelHierarchy(); self.visual_fitted=False
        self.shared.inputs['image']=48+self.shared.visual.max_patterns
        self.shared.weights['image']=self.shared.rng.normal(0,.04,(self.shared.inputs['image'],self.shared.d)).astype('float32')
        self.reasoner=RelationalNeuralTBIN(seed=seed)
    def observe_text(self,text):
        self.shared.observe_text(text);self.reasoner.observe(text)
    def observe_image(self,image):
        return self.visual.observe(image)
    def observe_audio(self,wav_path):
        return analyze_wav(wav_path)
    def fit_paired(self,examples,epochs=100):
        """examples = [(concept_id, {'text':str,'image':ndarray,'audio':waveform}), ...].
        These are explicitly paired examples, not unsupervised alignment.
        """
        self.shared.fit(examples,epochs=epochs)
    def predict(self,modality,raw):return self.shared.predict(modality,raw)

def demo():
    import math, tempfile, wave
    rng=np.random.default_rng(12)
    m=UnifiedTBIN(); training=[]
    for label, color, freq in [('red',(235,30,30),320),('green',(30,235,30),530),('blue',(30,30,235),780)]:
        for i in range(5):
            img=np.clip(np.full((32,32,3),color,dtype=np.float32)+rng.normal(0,5,(32,32,3)),0,255).astype('uint8')
            t=np.arange(2048)/8000
            audio=(.6*np.sin(2*np.pi*freq*t+i*.03)).astype('float32')
            training.append((label,{'text':label+' color','image':img,'audio':audio}))
            m.observe_image(img)
    m.fit_paired(training,epochs=100)
    tests=[]
    for label,color,freq in [('red',(235,30,30),320),('green',(30,235,30),530),('blue',(30,30,235),780)]:
        img=np.clip(np.full((32,32,3),color,dtype=np.float32)+rng.normal(0,9,(32,32,3)),0,255).astype('uint8')
        t=np.arange(2048)/8000
        audio=(.6*np.sin(2*np.pi*freq*t+.2)).astype('float32')
        for modality,value in [('image',img),('audio',audio),('text',label+' color')]:
            pred=m.predict(modality,value)
            tests.append({'modality':modality,'expected':label,'predicted':pred['concept'],'correct':pred['concept']==label})
    with tempfile.NamedTemporaryFile(suffix='.wav') as f:
        with wave.open(f.name,'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(8000)
            t=np.arange(8000*4)/8000
            s=(.5*np.sin(2*np.pi*440*t)*32767).astype('<i2');w.writeframes(s.tobytes())
        audio_result=m.observe_audio(f.name)
    image_result=m.observe_image(np.full((65,67,3),[150,80,20],dtype='uint8'))
    result={'version':'23','training_triplets':len(training),'test_cases':len(tests),
            'correct':sum(x['correct'] for x in tests),'tests':tests,
            'audio_streaming':{'duration_s':audio_result['duration_s'],'frames':audio_result['frames'], 'phrases':audio_result['phrases']},
            'pixel_hierarchy':{'patches':image_result['patches'],'regions':len(image_result['regions']), 'learned_patterns':image_result['learned_pixel_patterns']},
            'shared_concept_count':len(m.shared.prototypes),
            'neural_reasoner_parameters':int(sum(x.size for x in (m.reasoner.W,m.reasoner.b,m.reasoner.V,m.reasoner.c))),
            'limitations':'Synthetic paired labels; neural reasoner included but not trained on this demo; no ASR, object detection, or self-supervised cross-modal discovery.'}
    output=Path(__file__).with_name('tbin_v23_results.json');output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='tests'},indent=2))
    assert all(x['correct'] for x in tests), 'Cross-modal synthetic regression failed'
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['demo','image','audio']);p.add_argument('file',nargs='?');p.add_argument('--json')
    a=p.parse_args()
    if a.command=='demo':demo();return
    if not a.file:p.error('file is required')
    if a.command=='image':
        with Image.open(a.file) as im:result=UnifiedTBIN().observe_image(np.asarray(im.convert('RGB')))
    else:result=UnifiedTBIN().observe_audio(a.file)
    if a.json:Path(a.json).write_text(json.dumps(result,indent=2))
    print(json.dumps(result if a.command=='image' else {k:v for k,v in result.items() if k not in ('phrase_timeline','sections','pattern_transitions')},indent=2))
if __name__=='__main__':main()
