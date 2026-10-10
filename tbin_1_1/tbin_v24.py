"""TBIN v24: CPU multimodal observation + unlabeled paired episodic alignment.
Not ASR, image recognition, or autonomous semantic concept discovery.
Requires numpy, pillow and sibling tbin_v20.py/tbin_v21.py/tbin_v22_audio.py/tbin_v23_unified.py.
Usage: python tbin_v24.py demo | image PHOTO | audio PCM_WAV
"""
import argparse, json, wave, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from tbin_v23_unified import UnifiedTBIN

class TBIN24(UnifiedTBIN):
    def __init__(self, seed=7, max_episodes=2048):
        super().__init__(seed)
        self.episodes=[]; self.max_episodes=max_episodes
    def observe(self, modality, value):
        if modality=='image': return self.observe_image(np.asarray(value))
        if modality=='audio': return self.observe_audio(str(value))
        if modality=='text':
            self.observe_text(str(value));return {'characters':len(str(value))}
        raise ValueError('modality must be text, image, or audio')
    def pair(self, **observations):
        """Co-occurrence learning: no semantic label needed, but pairing is supervision.
        Store compact features, not full raw media. Supports audio as numpy waveform.
        """
        valid={'text','image','audio'}
        if len(observations)<2 or set(observations)-valid:raise ValueError('Provide 2+ modalities')
        features={k:self.shared.feature(k,v).astype(np.float32) for k,v in observations.items()}
        self.episodes.append(features)
        if len(self.episodes)>self.max_episodes:self.episodes.pop(0)
    def retrieve(self, modality, observation, target_modality, top_k=3):
        """Nearest stored cross-modal paired episode; no hallucinated semantic labels."""
        q=self.shared.feature(modality,observation).astype(np.float32)
        hits=[]
        for i,e in enumerate(self.episodes):
            if modality not in e or target_modality not in e:continue
            x=e[modality];score=float(np.dot(q,x)/(np.linalg.norm(q)*np.linalg.norm(x)+1e-9))
            hits.append((score,i))
        hits.sort(reverse=True)
        return [{'episode':i,'similarity':round(score,5),'target_feature':self.episodes[i][target_modality].tolist()} for score,i in hits[:top_k]]
    def save_episodes(self,path):
        # No pickle or arbitrary object deserialization.
        Path(path).write_text(json.dumps({'version':24,'episodes':[{k:v.tolist() for k,v in e.items()} for e in self.episodes]}))
    def load_episodes(self,path):
        d=json.loads(Path(path).read_text())
        if d.get('version')!=24:raise ValueError('Wrong version')
        self.episodes=[{k:np.asarray(v,dtype=np.float32) for k,v in e.items()} for e in d['episodes'][-self.max_episodes:]]

def demo():
    rng=np.random.default_rng(25);m=TBIN24();sr=8000
    colors=[(220,20,20),(20,220,20),(20,20,220)]
    freqs=[270,470,730]
    for color,f in zip(colors,freqs):
        img=np.full((32,32,3),color,dtype=np.uint8)
        t=np.arange(4096)/sr;audio=(.5*np.sin(2*np.pi*f*t)).astype(np.float32)
        m.pair(image=img,audio=audio)
    tests=[]
    for idx,(color,f) in enumerate(zip(colors,freqs)):
        img=np.clip(np.asarray(color)[None,None,:]+rng.normal(0,4,(32,32,3)),0,255).astype(np.uint8)
        t=np.arange(4096)/sr;audio=(.5*np.sin(2*np.pi*f*t+.13)).astype(np.float32)
        for modality,value,target in [('image',img,'audio'),('audio',audio,'image')]:
            hits=m.retrieve(modality,value,target,1)
            tests.append({'modality':modality,'expected_episode':idx,'retrieved_episode':hits[0]['episode'],'correct':hits[0]['episode']==idx})
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'episodes.json';m.save_episodes(path)
        loaded=TBIN24();loaded.load_episodes(path)
        persistence_ok=len(loaded.episodes)==len(m.episodes)
        wav=Path(d)/'song.wav'; t=np.arange(sr*3)/sr
        with wave.open(str(wav),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr)
            w.writeframes((np.sin(2*np.pi*440*t)*16000).astype('<i2').tobytes())
        song=m.observe('audio',wav)
    image=m.observe('image',np.full((33,34,3),[20,30,40],dtype=np.uint8))
    result={'version':24,'paired_episodes':len(m.episodes),'test_cases':len(tests),'correct':sum(x['correct'] for x in tests),'tests':tests,'persistence_ok':persistence_ok,'streaming_audio_frames':song['frames'],'pixel_patches':image['patches'],'gpu_required':False,'limitations':'Toy paired synthetic images and tones; paired observations are supervisory signal. No speech transcription, music source separation, object identification, learned cross-modal projection from unlabeled data, or semantic reasoning proof.'}
    Path(__file__).with_name('tbin_v24_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='tests'},indent=2))
    assert all(x['correct'] for x in tests) and persistence_ok

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['demo','image','audio']);p.add_argument('path',nargs='?');a=p.parse_args()
    if a.command=='demo':demo()
    elif not a.path:p.error('path required')
    elif a.command=='image':
        with Image.open(a.path) as im:print(json.dumps(TBIN24().observe('image',im.convert('RGB')),indent=2))
    else:
        r=TBIN24().observe('audio',a.path)
        print(json.dumps({k:v for k,v in r.items() if k not in ('phrase_timeline','sections','pattern_transitions')},indent=2))
