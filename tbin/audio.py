"""Dependency-light WAV frontend for TBIN real-speech falsification.

Uses Python's standard library only. This is not ASR. It converts PCM WAV audio
into deterministic frame-level acoustic trajectories for structural testing.
"""
import math, wave
from array import array
from pathlib import Path

def read_wav_mono(path):
    path=Path(path)
    with wave.open(str(path),"rb") as wf:
        channels=wf.getnchannels()
        width=wf.getsampwidth()
        rate=wf.getframerate()
        frames=wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError("Prototype currently requires 16-bit PCM WAV")
    samples=array("h"); samples.frombytes(frames)
    if channels > 1:
        samples=array("h",(sum(samples[i:i+channels])//channels
                           for i in range(0,len(samples),channels)))
    scale=float(1 << 15)
    return rate,[x/scale for x in samples]

def acoustic_trajectory(path, frame_ms=25, hop_ms=10):
    """Return an amplitude/energy/zero-crossing trajectory scaled to 0..255."""
    rate,x=read_wav_mono(path)
    frame=max(1,int(rate*frame_ms/1000))
    hop=max(1,int(rate*hop_ms/1000))
    raw=[]
    for start in range(0,max(1,len(x)-frame+1),hop):
        f=x[start:start+frame]
        if not f: continue
        mean_abs=sum(abs(v) for v in f)/len(f)
        rms=math.sqrt(sum(v*v for v in f)/len(f))
        zc=sum((f[i]>=0)!=(f[i-1]>=0) for i in range(1,len(f)))/max(1,len(f)-1)
        raw.append((mean_abs,rms,zc))
    if not raw:
        return []
    # Per-utterance normalization deliberately suppresses microphone gain.
    cols=list(zip(*raw))
    norm=[]
    for col in cols:
        lo,hi=min(col),max(col); span=max(hi-lo,1e-12)
        norm.append([(v-lo)/span for v in col])
    # Interleave descriptors; BraidEncoder consumes a one-dimensional trajectory.
    out=[]
    for i in range(len(raw)):
        out.extend(round(255*norm[j][i]) for j in range(3))
    return out
