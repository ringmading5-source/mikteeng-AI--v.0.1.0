"""Bounded pattern prototypes and reversible byte transport.
Routing vectors are lossy features; BytePacket is the exact data path.
"""
from dataclasses import dataclass
import numpy as np
import json
from pathlib import Path
import zipfile
from threading import RLock
from functools import wraps

def locked(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        with self._lock:return method(self,*args,**kwargs)
    return call


def unit(value):
    a=np.asarray(value,dtype=np.float64)
    if a.ndim!=1 or not np.isfinite(a).all():
        raise ValueError("expected a finite vector")
    norm=np.linalg.norm(a)
    if not np.isfinite(norm) or norm==0:raise ValueError("zero or overflowing vector")
    return a/norm

@dataclass(frozen=True)
class BytePacket:
    payload: bytes
    modality: str = "binary"
    def __post_init__(self):
        if not isinstance(self.payload,bytes) or len(self.payload)>16*1024*1024:raise ValueError("byte packet exceeds resource budget")
    @classmethod
    def text(cls,text):return cls(text.encode("utf-8"),"text")
    def tokens(self):return np.frombuffer(self.payload,dtype=np.uint8).copy()
    def project(self):
        # Signed reversal is an orthogonal operator, O(n), with no dense matrix.
        y=self.tokens().astype(np.float64)/255.0
        signs=np.where(np.arange(len(y))%2, -1.0, 1.0)
        return y[::-1]*signs
    @staticmethod
    def recover(z,modality="binary"):
        if len(z)>16*1024*1024:raise ValueError("projection exceeds byte budget")
        z=np.asarray(z,dtype=np.float64)
        if z.ndim!=1 or not np.isfinite(z).all():raise ValueError("invalid projection")
        signs=np.where(np.arange(len(z))%2,-1.0,1.0)
        values=(z*signs)[::-1]*255.0
        rounded=np.rint(values)
        if np.any(rounded<0) or np.any(rounded>255) or not np.allclose(values,rounded,rtol=0,atol=1e-9):
            raise ValueError("projection is not an exact byte encoding")
        return BytePacket(rounded.astype(np.uint8).tobytes(),modality)
    def decode(self):return self.payload.decode("utf-8") if self.modality=="text" else self.payload

class PatternNodeSpace:
    """One normalized anchor per assigned pattern; bounded exact cosine search.

    Pattern assignments come from training labels or an existing learned bank.
    This registry does not discover meaning from byte similarity.
    """
    def __init__(self,dimensions=256,max_nodes=4096,threshold=.8):
        if type(dimensions)!=int or dimensions<2 or type(max_nodes)!=int or max_nodes<1:
            raise ValueError("invalid capacity")
        if not -1<=threshold<=1:raise ValueError("invalid threshold")
        self.dimensions=dimensions;self.max_nodes=max_nodes;self.threshold=threshold
        if dimensions>8192 or max_nodes*dimensions>16_777_216:raise ValueError("configured node storage exceeds resource budget")
        self.ids=[];self.index={};self.anchors=np.zeros((0,dimensions))
        self.counts=np.zeros(0,dtype=np.int64);self._lock=RLock()
    def __getstate__(self):
        with self._lock:
            state=self.__dict__.copy();state.pop("_lock",None);return state
    def __setstate__(self,state):
        self.__dict__.update(state);self._lock=RLock()
    def _reserve(self,needed):
        if needed<=len(self.anchors):return
        capacity=min(self.max_nodes,max(needed,16,len(self.anchors)*2))
        anchors=np.zeros((capacity,self.dimensions));counts=np.zeros(capacity,dtype=np.int64)
        anchors[:len(self.anchors)]=self.anchors;counts[:len(self.counts)]=self.counts
        self.anchors=anchors;self.counts=counts
    @locked
    def save_nodes(self,path):
        metadata={"schema":1,"ids":self.ids,"dimensions":self.dimensions,"max_nodes":self.max_nodes,"threshold":self.threshold}
        raw=json.dumps(metadata).encode('utf-8')
        if len(raw)>1024*1024:raise ValueError('node metadata exceeds budget')
        with open(path,'wb') as stream:
            np.savez_compressed(stream,metadata=np.frombuffer(raw,dtype=np.uint8),anchors=self.anchors[:len(self.ids)],counts=self.counts[:len(self.ids)])
    @classmethod
    def load_nodes(cls,path):
        if Path(path).stat().st_size>128*1024*1024:raise ValueError('node file exceeds budget')
        with zipfile.ZipFile(path) as archive:
            if sum(i.file_size for i in archive.infolist())>160*1024*1024:raise ValueError('expanded node file exceeds budget')
        with np.load(path,allow_pickle=False) as data:
            raw=data['metadata']
            if raw.dtype!=np.uint8 or raw.ndim!=1 or raw.size>1024*1024:raise ValueError('invalid metadata')
            m=json.loads(raw.tobytes());ids=m['ids']
            if m.get('schema')!=1 or not isinstance(ids,list) or any(not isinstance(i,str) or not i or len(i)>256 for i in ids) or len(set(ids))!=len(ids):raise ValueError('invalid node identity')
            space=cls(m['dimensions'],m['max_nodes'],m['threshold'])
            a=data['anchors'];counts=data['counts'];n=len(ids)
            if n>space.max_nodes or a.dtype!=np.float64 or a.shape!=(n,space.dimensions) or not np.isfinite(a).all():raise ValueError('invalid anchors')
            if counts.dtype!=np.int64 or counts.shape!=(n,) or np.any(counts<1):raise ValueError('invalid evidence counts')
            if not np.allclose(np.linalg.norm(a,axis=1),1,rtol=0,atol=1e-8):raise ValueError('anchors must be unit normalized')
            space._reserve(n);space.ids=ids;space.index={v:i for i,v in enumerate(ids)};space.anchors[:n]=a;space.counts[:n]=counts
            return space
    def byte_vector(self,packet):
        # Stable bounded byte + adjacent-pair histogram, O(payload length).
        b=packet.tokens().astype(np.int64)
        vector=np.zeros(self.dimensions)
        if not len(b):vector[0]=1;return vector
        vector+=np.bincount(b%self.dimensions,minlength=self.dimensions)
        if len(b)>1:
            vector+=np.bincount((b[:-1]*257+b[1:]+65537)%self.dimensions,minlength=self.dimensions)
        return unit(vector)
    def _vector(self,vector):
        y=unit(vector)
        if len(y)!=self.dimensions:raise ValueError("dimension mismatch")
        return y
    @locked
    def observe(self,node_id,vector,eta=.01):
        if not isinstance(node_id,str) or not node_id or len(node_id)>256:raise ValueError("node id required")
        if not np.isfinite(eta) or not 0<=eta<=1:raise ValueError("eta must be in [0,1]")
        y=self._vector(vector)
        if node_id not in self.index:
            if len(self.ids)>=self.max_nodes:raise ValueError("node capacity reached")
            i=len(self.ids);self._reserve(i+1);self.index[node_id]=i;self.ids.append(node_id)
            self.anchors[i]=y;drift=0.0
        else:
            i=self.index[node_id];old=self.anchors[i].copy()
            candidate=(1-eta)*old+eta*y
            # Opposite vectors at eta=.5 cancel: keep the prior valid anchor.
            if np.linalg.norm(candidate)>1e-12:self.anchors[i]=unit(candidate)
            drift=float(np.linalg.norm(self.anchors[i]-old))
        self.counts[i]+=1
        return {"node_id":node_id,"v_self_post_norm":float(np.linalg.norm(self.anchors[i])),"state_drift_delta":drift}
    @locked
    def route(self,vector,k=1):
        if type(k)!=int or k<1:raise ValueError("k must be positive")
        y=self._vector(vector);n=len(self.ids)
        if not n:return []
        scores=self.anchors[:n]@y;k=min(k,n)
        selected=np.argpartition(-scores,k-1)[:k]
        selected=sorted(selected,key=lambda i:(-scores[i],i))
        return [{"node_id":self.ids[i],"activation_score":float(scores[i]),
                 "evidence_count":int(self.counts[i]),"status":"ACTIVE" if scores[i]>=self.threshold else "ROUTE_ADJACENT"} for i in selected]
    @classmethod
    def from_assignments(cls,x,labels,**kwargs):
        x=np.asarray(x,dtype=float);labels=np.asarray(labels)
        if x.ndim!=2 or len(x)!=len(labels) or not len(x):raise ValueError("invalid assigned examples")
        space=cls(dimensions=x.shape[1],**kwargs)
        normalized=np.array([unit(row) for row in x])
        for label in np.unique(labels):
            space.observe(str(label),normalized[labels==label].mean(axis=0))
            space.counts[space.index[str(label)]]=int(np.sum(labels==label))
        return space
