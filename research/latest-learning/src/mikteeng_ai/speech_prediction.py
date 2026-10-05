"""CPU acoustic observations feeding Mikteeng's original prediction/Self loop.
No transcription or pretrained speech model. Waveform prediction is not TTS.
"""
import numpy as np
from scipy.fft import dct,idct
from scipy.io import wavfile
from .prediction_patterns import PredictionPatternLearner
from .vector_nodes import PatternNodeSpace

class AcousticEncoder:
    def __init__(self,sample_rate=16000,frame_size=256,max_seconds=60):
        if type(sample_rate)!=int or not 8000<=sample_rate<=48000:raise ValueError('sample rate must be 8000..48000')
        if type(frame_size)!=int or not 32<=frame_size<=1024:raise ValueError('frame size must be 32..1024')
        if type(max_seconds)!=int or not 1<=max_seconds<=60:raise ValueError('duration must be 1..60 seconds')
        self.sample_rate=sample_rate;self.frame_size=frame_size;self.max_seconds=max_seconds
    def samples(self,audio,rate):
        if rate!=self.sample_rate:raise ValueError('sample rate mismatch; convert explicitly before training')
        a=np.asarray(audio)
        if a.ndim!=1 or not 0<a.size<=self.sample_rate*self.max_seconds:raise ValueError('bounded mono audio required')
        if a.dtype.kind=='i':a=a.astype(float)/float(2**(a.dtype.itemsize*8-1))
        elif a.dtype==np.uint8:a=(a.astype(float)-128)/128
        elif a.dtype.kind=='f':a=a.astype(float)
        else:raise ValueError('signed PCM, uint8 PCM or float audio required')
        if not np.isfinite(a).all() or np.any(np.abs(a)>1):raise ValueError('audio must be finite in [-1,1]')
        return a
    def encode(self,audio,rate):
        a=self.samples(audio,rate);n=len(a)
        padded=np.pad(a,(0,(-n)%self.frame_size))
        frames=padded.reshape(-1,self.frame_size)
        # Full orthonormal DCT: preserves phase/time structure and amplitude.
        return dct(frames,type=2,norm='ortho',axis=1),n
    def decode(self,vectors,length=None):
        vectors=np.asarray(vectors,dtype=float)
        if vectors.ndim!=2 or vectors.shape[1]!=self.frame_size or not np.isfinite(vectors).all():raise ValueError('invalid acoustic vectors')
        if len(vectors)>int(np.ceil(self.sample_rate*self.max_seconds/self.frame_size)):raise ValueError('acoustic output exceeds duration budget')
        a=idct(vectors,type=2,norm='ortho',axis=1).ravel()
        if length is not None:
            if type(length)!=int or not 0<=length<=len(a):raise ValueError('invalid decoded length')
            a=a[:length]
        return a

class SpeechPredictionModel:
    def __init__(self,sample_rate=16000,frame_size=256,observation_window=3,prediction_window=3,alpha=1e-4,max_sound_nodes=256):
        self.encoder=AcousticEncoder(sample_rate,frame_size)
        self.core=PredictionPatternLearner(observation_window,prediction_window,alpha)
        self.sound_nodes=PatternNodeSpace(frame_size,max_sound_nodes,threshold=.95)
    def fit(self,recordings,*,self_recordings):
        # Each recording is (mono samples, rate). Groups must be separate recordings.
        primary=[self.encoder.encode(a,rate)[0] for a,rate in recordings]
        secondary=[self.encoder.encode(a,rate)[0] for a,rate in self_recordings]
        if {v.tobytes() for v in primary}&{v.tobytes() for v in secondary}:raise ValueError('primary and Self groups must not repeat exact encoded recordings')
        self.core.fit(primary,self_observations=secondary)
        nodes=PatternNodeSpace(self.encoder.frame_size,self.sound_nodes.max_nodes,threshold=self.sound_nodes.threshold)
        silence=0
        for clip in primary+secondary:
            for frame in clip:
                if np.linalg.norm(frame)<1e-8:silence+=1;continue
                candidates=nodes.route(frame)
                if candidates and candidates[0]['status']=='ACTIVE':identifier=candidates[0]['node_id']
                else:identifier='sound-'+str(len(nodes.ids))
                nodes.observe(identifier,frame)
        self.sound_nodes=nodes;self.silent_frames=silence;return self
    def build_self(self,audio,rate):
        vectors,_=self.encoder.encode(audio,rate)
        bound=self.core.observation_window+self.core.prediction_window-1
        return self.core.build_self(vectors[-bound:])
    def predict(self,learned_self,frames=1):
        response=self.core.respond(learned_self,steps=frames)
        audio=self.encoder.decode(response['response'])
        clipped=np.clip(audio,-1,1)
        return {'samples':clipped,'sample_rate':self.encoder.sample_rate,
                'clipped_samples':int(np.count_nonzero(audio!=clipped)),
                'mechanism':'original observation -> prediction-history Self -> acoustic response',
                'status':'experimental_waveform_prediction','intelligible_speech_verified':False}
    def write_wav(self,path,response):
        if response['sample_rate']!=self.encoder.sample_rate:raise ValueError('sample rate mismatch')
        samples=np.asarray(response['samples'])
        wavfile.write(path,self.encoder.sample_rate,np.rint(np.clip(samples,-1,1)*32767).astype(np.int16))
