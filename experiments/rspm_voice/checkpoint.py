"""Versioned, atomic JSON checkpoints; no executable pickle deserialization."""
import base64
import json
import os
import tempfile
from pathlib import Path
import numpy as np
try:
 from .engine import BoundedRSPM
except ImportError:
 from engine import BoundedRSPM

FORMAT='mikteeng-rspm-pattern-router-v1'
MAX_BYTES=256*1024*1024

def pack(x):
    if isinstance(x,np.ndarray):
        return {'array':base64.b64encode(x.tobytes()).decode(), 'dtype':x.dtype.str,'shape':list(x.shape)}
    if isinstance(x,dict): return {'dict':[[pack(k),pack(v)] for k,v in x.items()]}
    if isinstance(x,list): return {'list':[pack(v) for v in x]}
    if isinstance(x,np.generic): return x.item()
    return x

def unpack(x):
    if not isinstance(x,dict): return x
    if 'array' in x:
        dtype=np.dtype(x['dtype'])
        if dtype.hasobject: raise ValueError('Object arrays are not supported')
        return np.frombuffer(base64.b64decode(x['array'],validate=True),dtype=dtype).copy().reshape(x['shape'])
    if 'dict' in x: return {unpack(k):unpack(v) for k,v in x['dict']}
    if 'list' in x: return [unpack(v) for v in x['list']]
    raise ValueError('Invalid checkpoint structure')

def save(engine,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload={'format':FORMAT,'state':pack(engine.snapshot())}
    name=None
    try:
        with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False,encoding='utf-8') as f:
            name=f.name; json.dump(payload,f,allow_nan=False); f.flush(); os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if name and os.path.exists(name): os.unlink(name)

def load(path):
    path=Path(path)
    if path.stat().st_size>MAX_BYTES: raise ValueError('Checkpoint exceeds 256 MiB loader limit')
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('format')!=FORMAT: raise ValueError('Not a Mikteeng RSPM checkpoint')
    s=unpack(data['state']); cfg=s['config']; engine=BoundedRSPM(**cfg)
    if s.get('basis_writeable') is not False: raise ValueError('Checkpoint basis must be frozen')
    dim=cfg['dim']
    if s['B'].shape!=(dim,dim) or not np.isfinite(s['B']).all(): raise ValueError('Invalid basis')
    if not np.allclose(s['B']@s['B'].T,np.eye(dim),atol=1e-8,rtol=0): raise ValueError('Non-orthogonal basis')
    engine.restore(s); engine.check_bounds()
    for c in engine.contexts.values():
        if c['R'].shape!=(dim,dim) or not np.isfinite(c['R']).all(): raise ValueError('Invalid local operator')
        engine.vector(c['key'])
        for store in c['stores'].values():
            engine.vector(store['key'])
            for mode in store['modes']+store['candidates']:
                engine.vector(mode['centroid']); engine.vector(mode['sum'])
                if mode['count']<1: raise ValueError('Invalid evidence count')
    return engine
