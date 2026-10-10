"""TBIN v22 streaming audio hierarchy (CPU, NumPy). WAV PCM 8/16/24/32-bit.
Usage: python tbin_v22_audio.py path/to/song.wav --json song_features.json
Optional MP3/M4A conversion: ffmpeg -i song.mp3 -ac 1 -ar 16000 song.wav
This is feature extraction and unsupervised pattern clustering, NOT speech transcription,
source separation, semantic understanding or lossless waveform reconstruction.
"""
import argparse, json, math, wave
from collections import Counter
import numpy as np

class StreamingAudioTBIN:
    def __init__(self, sample_rate=16000, frame=1024, hop=512, phrase_seconds=3.,
                 section_seconds=20., clusters=24, seed=11):
        self.sr=sample_rate; self.frame=frame; self.hop=hop
        self.phrase_frames=max(1,round(phrase_seconds*sample_rate/hop))
        self.section_phrases=max(1,round(section_seconds/phrase_seconds))
        self.clusters=clusters; self.rng=np.random.default_rng(seed)
        self.centers=[];self.cluster_counts=[];self.pattern_counts=Counter()
        self.phrases=[];self.sections=[];self.transitions=Counter()
        self.buffer=np.zeros(0,dtype=np.float32);self.frame_acc=[];self.phrase_acc=[]
        self.last_cluster=None;self.total_samples=0;self.frame_count=0
        self.window=np.hanning(frame).astype(np.float32)
    def _features(self, frame):
        f=frame*self.window
        spectrum=np.abs(np.fft.rfft(f)).astype(np.float32)
        power=spectrum*spectrum
        edges=np.geomspace(1,len(power),17).astype(int)
        bands=np.array([np.log1p(np.mean(power[edges[i]:max(edges[i]+1,edges[i+1])])) for i in range(16)],np.float32)
        rms=float(np.sqrt(np.mean(f*f)))
        zcr=float(np.mean(np.signbit(f[1:])!=np.signbit(f[:-1])))
        centroid=float(np.dot(np.arange(len(power)),power)/(power.sum()+1e-10))/len(power)
        return np.r_[bands,rms,zcr,centroid].astype(np.float32)
    def _assign(self, vector):
        # Adaptive prototypes: new patterns can form until capacity; afterward online centroid updates.
        v=vector/(np.linalg.norm(vector)+1e-9)
        if not self.centers:
            self.centers.append(v.copy());self.cluster_counts.append(1);return 0
        distances=np.array([np.linalg.norm(v-c) for c in self.centers])
        idx=int(distances.argmin())
        if len(self.centers)<self.clusters and distances[idx]>.18:
            idx=len(self.centers);self.centers.append(v.copy());self.cluster_counts.append(1)
        else:
            self.cluster_counts[idx]+=1
            rate=min(.08,1/self.cluster_counts[idx]);self.centers[idx]=(1-rate)*self.centers[idx]+rate*v
            self.centers[idx]/=np.linalg.norm(self.centers[idx])+1e-9
        return idx
    def _finish_phrase(self):
        if not self.frame_acc:return
        matrix=np.stack(self.frame_acc)
        summary=np.r_[matrix.mean(axis=0),matrix.std(axis=0)].astype(np.float32)
        idx=self._assign(summary)
        self.pattern_counts[idx]+=1
        if self.last_cluster is not None:self.transitions[(self.last_cluster,idx)]+=1
        self.last_cluster=idx
        self.phrases.append({'pattern':idx,'start_s':round((self.frame_count-len(self.frame_acc))*self.hop/self.sr,2),
                             'rms':round(float(matrix[:,-3].mean()),5),
                             'spectral_centroid':round(float(matrix[:,-1].mean()),5)})
        self.phrase_acc.append(idx);self.frame_acc=[]
        if len(self.phrase_acc)>=self.section_phrases:self._finish_section()
    def _finish_section(self):
        if not self.phrase_acc:return
        hist=Counter(self.phrase_acc)
        self.sections.append({'section':len(self.sections),'patterns':dict(hist),
                              'dominant_pattern':hist.most_common(1)[0][0]})
        self.phrase_acc=[]
    def feed(self, mono_float_samples):
        a=np.asarray(mono_float_samples,dtype=np.float32).reshape(-1)
        if not np.isfinite(a).all():raise ValueError('nonfinite audio')
        self.total_samples+=len(a)
        self.buffer=np.concatenate((self.buffer,a))
        while len(self.buffer)>=self.frame:
            self.frame_acc.append(self._features(self.buffer[:self.frame]))
            self.frame_count+=1
            self.buffer=self.buffer[self.hop:]
            if len(self.frame_acc)>=self.phrase_frames:self._finish_phrase()
    def finish(self):
        if self.frame_acc:self._finish_phrase()
        if self.phrase_acc:self._finish_section()
        return {'duration_s':round(self.total_samples/self.sr,2),'sample_rate':self.sr,
                'frames':self.frame_count,'phrases':len(self.phrases),'sections':self.sections,
                'phrase_timeline':self.phrases,'learned_patterns':len(self.centers),
                'pattern_frequencies':dict(self.pattern_counts),
                'pattern_transitions':[{'from':a,'to':b,'count':n} for (a,b),n in self.transitions.most_common()],
                'note':'Unsupervised acoustic patterns; not lyrics, instruments or semantic concepts.'}

def _decode_pcm(raw,width,channels):
    if width==1:values=(np.frombuffer(raw,dtype=np.uint8).astype(np.float32)-128)/128
    elif width==2:values=np.frombuffer(raw,dtype='<i2').astype(np.float32)/32768
    elif width==3:
        b=np.frombuffer(raw,dtype=np.uint8).reshape(-1,3).astype(np.int32)
        n=b[:,0]|(b[:,1]<<8)|(b[:,2]<<16)
        n=np.where(n&0x800000,n-0x1000000,n)
        values=n.astype(np.float32)/8388608
    elif width==4:values=np.frombuffer(raw,dtype='<i4').astype(np.float32)/2147483648
    else:raise ValueError('Unsupported PCM sample width')
    return values.reshape(-1,channels).mean(axis=1)

def analyze_wav(path,chunk_seconds=1.,**kwargs):
    with wave.open(str(path),'rb') as w:
        if w.getcomptype()!='NONE':raise ValueError('Uncompressed PCM WAV required')
        sr=w.getframerate();engine=StreamingAudioTBIN(sample_rate=sr,**kwargs)
        chunk=max(1,int(sr*chunk_seconds))
        while True:
            raw=w.readframes(chunk)
            if not raw:break
            engine.feed(_decode_pcm(raw,w.getsampwidth(),w.getnchannels()))
    return engine.finish()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wav');parser.add_argument('--json',default='tbin_song_features.json')
    args=parser.parse_args()
    result=analyze_wav(args.wav)
    with open(args.json,'w',encoding='utf-8') as out:json.dump(result,out,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('phrase_timeline','sections','pattern_transitions')},indent=2))
