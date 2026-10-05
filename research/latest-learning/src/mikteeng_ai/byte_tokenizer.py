"""Shared lossless byte tokenization and bounded observed-pattern hierarchy."""
from dataclasses import dataclass
from hashlib import blake2b
import json
import math
from threading import RLock
from .vector_nodes import locked
import numpy as np
from .vector_nodes import BytePacket, PatternNodeSpace

@dataclass(frozen=True)
class MultimodalPacket:
    payload: bytes
    modality: str
    metadata_json: str
    def __post_init__(self):
        if not isinstance(self.payload,bytes) or len(self.payload)>16*1024*1024:raise ValueError('packet exceeds 16 MiB budget or is not bytes')
        if not isinstance(self.metadata_json,str) or len(self.metadata_json)>4096:raise ValueError('invalid metadata size')
        m=json.loads(self.metadata_json)
        if not isinstance(m,dict):raise ValueError('metadata must be an object')
        if self.modality not in ('text','binary','integer','image','audio','numeric','color'):raise ValueError('unsupported modality')
        if self.modality in ('image','audio','numeric','color'):
            dtype=np.dtype(m.get('dtype'))
            shape=m.get('shape')
            if dtype.kind not in 'biufc' or dtype.hasobject:raise ValueError('unsafe dtype')
            if not isinstance(shape,list) or len(shape)>8 or any(type(v)!=int or v<0 for v in shape):raise ValueError('invalid shape')
            if math.prod(shape)*dtype.itemsize!=len(self.payload):raise ValueError('shape does not match payload')
            if self.modality=='audio' and (type(m.get('sample_rate'))!=int or m['sample_rate']<=0):raise ValueError('invalid sample rate')
    @property
    def metadata(self):return json.loads(self.metadata_json)
    @property
    def tokens(self):return np.frombuffer(self.payload,dtype=np.uint8).copy()
    def project(self):return BytePacket(self.payload,self.modality).project()
    def decode(self):
        m=self.metadata
        if self.modality=='text':return self.payload.decode('utf-8')
        if self.modality=='binary':return self.payload
        if self.modality=='integer':return int.from_bytes(self.payload,'little',signed=False)
        if self.modality in ('image','audio','numeric','color'):
            arr=np.frombuffer(self.payload,dtype=np.dtype(m['dtype'])).reshape(m['shape']).copy()
            return (arr,m['sample_rate']) if self.modality=='audio' else arr
        raise ValueError('unsupported modality')
    @classmethod
    def recover(cls,z,modality,metadata):
        packet=BytePacket.recover(z,modality)
        result=cls(packet.payload,modality,json.dumps(metadata,sort_keys=True))
        result.decode() # Validate that the recovered bytes match the descriptor.
        return result

class MultimodalByteTokenizer:
    vocabulary_size=256
    @staticmethod
    def _packet(payload,modality,metadata):
        return MultimodalPacket(bytes(payload),modality,json.dumps(metadata,sort_keys=True))
    def encode(self,value,modality='text',*,sample_rate=None,color_space='RGB'):
        if modality=='text':
            if not isinstance(value,str):raise TypeError('text requires a string')
            return self._packet(value.encode('utf-8'),'text',{'encoding':'utf-8'})
        if modality=='binary':
            if not isinstance(value,(bytes,bytearray,memoryview)):raise TypeError('binary requires bytes')
            return self._packet(value,'binary',{})
        if modality=='integer':
            if type(value)!=int or value<0:raise ValueError('integer requires a nonnegative int')
            length=max(1,(value.bit_length()+7)//8)
            return self._packet(value.to_bytes(length,'little'),'integer',{'endian':'little','length':length})
        if modality not in ('image','audio','numeric','color'):raise ValueError('unsupported modality')
        arr=np.asarray(value)
        if arr.nbytes>16*1024*1024:raise ValueError('array exceeds 16 MiB budget')
        if arr.dtype.kind not in 'biufc':raise ValueError('numeric arrays only; objects are not serializable')
        metadata={}
        if modality=='image':
            if arr.dtype!=np.uint8 or arr.ndim not in (2,3) or (arr.ndim==3 and arr.shape[2] not in (1,3,4)):
                raise ValueError('image requires uint8 gray/RGB/RGBA pixels')
            if any(d==0 for d in arr.shape):raise ValueError('image must be nonempty')
            metadata['color_space']='GRAY' if arr.ndim==2 or arr.shape[2]==1 else 'RGB' if arr.shape[2]==3 else 'RGBA'
        if modality=='audio':
            if arr.ndim not in (1,2) or not arr.size or arr.dtype.kind not in 'iuf':raise ValueError('audio requires real mono/multichannel samples')
            if type(sample_rate)!=int or sample_rate<=0:raise ValueError('positive integer sample rate required')
            metadata['sample_rate']=sample_rate
        if modality=='color':
            if arr.shape!=(3,) or color_space not in ('RGB','HSV'):raise ValueError('color requires three components and RGB or HSV')
            if color_space=='RGB' and (arr.dtype.kind not in 'iu' or np.any(arr<0) or np.any(arr>255)):
                raise ValueError('RGB requires integers in [0,255]')
            if color_space=='RGB':arr=arr.astype(np.uint8)
            metadata['color_space']=color_space
        # Canonical little-endian numeric arrays; preserve dtype width and shape.
        dtype=arr.dtype.newbyteorder('<')
        arr=np.ascontiguousarray(arr.astype(dtype,copy=False))
        metadata.update(dtype=dtype.str,shape=list(arr.shape),order='C')
        return self._packet(arr.tobytes(),modality,metadata)
    def encode_image_file(self,path):
        # Explicit decoded-pixel path: original file/container bytes require binary.
        from PIL import Image
        with Image.open(path) as im:
            if im.width*im.height*3>16*1024*1024:raise ValueError('decoded image exceeds budget')
            return self.encode(np.asarray(im.convert('RGB')), 'image')
    def encode_audio_file(self,path):
        from scipy.io import wavfile
        from pathlib import Path
        if Path(path).stat().st_size>16*1024*1024:raise ValueError("use iter_wav for large recordings")
        rate,samples=wavfile.read(path)
        return self.encode(samples,'audio',sample_rate=int(rate))

    def iter_wav(self,path,frames_per_chunk=16000):
        """Bounded PCM WAV blocks; 24-bit/float/compressed WAV require a decoder."""
        import wave
        if type(frames_per_chunk)!=int or not 1<=frames_per_chunk<=65536:raise ValueError('chunk size must be 1..65536')
        with wave.open(str(path),'rb') as stream:
            width=stream.getsampwidth();channels=stream.getnchannels()
            if width not in (1,2,4) or not 1<=channels<=8:raise ValueError('unsupported PCM format')
            dtype={1:'u1',2:'<i2',4:'<i4'}[width]
            while True:
                raw=stream.readframes(frames_per_chunk)
                if not raw:break
                samples=np.frombuffer(raw,dtype=dtype)
                if channels>1:samples=samples.reshape(-1,channels)
                yield self.encode(samples,'audio',sample_rate=stream.getframerate())

class BytePatternHierarchy:
    """Learn exact recurring groups bottom-up; vectors support nearby routing.

    Grouping is fixed. Pattern identities and counts come from observations.
    Exact child tuples determine node reuse; cosine similarity does not merge
    different byte sequences. These are structural patterns, not semantic roles.
    """
    def __init__(self,depth=3,group_size=4,dimensions=128,max_nodes_per_level=4096):
        if type(depth)!=int or not 1<=depth<=16:raise ValueError('depth must be 1..16')
        if type(group_size)!=int or not 2<=group_size<=256:raise ValueError('group size must be 2..256')
        self.depth=depth;self.group_size=group_size;self._lock=RLock()
        self.spaces=[PatternNodeSpace(dimensions,max_nodes_per_level) for _ in range(depth)]
        self.patterns=[{} for _ in range(depth)]
    def __getstate__(self):
        with self._lock:
            state=self.__dict__.copy();state.pop("_lock",None);return state
    def __setstate__(self,state):
        self.__dict__.update(state);self._lock=RLock()
    def _vector(self,key,dimensions):
        vector=np.zeros(dimensions)
        # Position-dependent stable hashing of exact child identities.
        for position,child in enumerate(key):
            digest=blake2b(f'{position}:{child}'.encode(),digest_size=8).digest()
            vector[int.from_bytes(digest,'little')%dimensions]+=1
        vector[len(key)%dimensions]+=1
        return vector
    @locked
    def observe(self,packet):
        if not isinstance(packet,MultimodalPacket):raise TypeError('expected MultimodalPacket')
        values=packet.tokens.astype(int).tolist();trace=[]
        # Preflight every capacity before mutating: overflow leaves state unchanged.
        proposed=[]
        for level in range(self.depth):
            known=self.patterns[level];new={};groups=[];next_values=[]
            for start in range(0,len(values),self.group_size):
                key=tuple(values[start:start+self.group_size]);groups.append(key)
                if key in known:identifier=known[key]
                elif key in new:identifier=new[key]
                else:identifier=len(known)+len(new);new[key]=identifier
                next_values.append(identifier)
            if len(known)+len(new)>self.spaces[level].max_nodes:raise ValueError(f'node capacity reached at level {level+1}')
            proposed.append((new,groups,next_values));values=next_values
        for level,(new,groups,values) in enumerate(proposed):
            self.patterns[level].update(new);space=self.spaces[level]
            for key in groups:
                identifier=self.patterns[level][key]
                space.observe(str(identifier),self._vector(key,space.dimensions))
            trace.append({'level':level+1,'node_ids':values,'unique_nodes':len(self.patterns[level]),'new_nodes':len(new)})
        return {'modality':packet.modality,'metadata':packet.metadata,'byte_count':len(packet.payload),'levels':trace}
    @locked
    def statistics(self):
        return [{'level':i+1,'nodes':len(space.ids),'observations':int(space.counts.sum()),
                 'anchor_bytes_allocated':space.anchors.nbytes} for i,space in enumerate(self.spaces)]
