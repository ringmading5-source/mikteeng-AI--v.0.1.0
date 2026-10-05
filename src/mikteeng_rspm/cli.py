"""Stream vector JSONL datasets without loading the dataset into RAM."""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from .engine import BoundedRSPM
from .checkpoint import load,save

def jsonable(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,list): return [jsonable(v) for v in x]
    if isinstance(x,np.generic): return x.item()
    return x

def train(args):
    if args.checkpoint_every<1 or args.epochs<1: raise ValueError('Positive epochs and checkpoint interval required')
    model=load(args.resume) if args.resume else BoundedRSPM(dim=args.dim,max_contexts=args.max_contexts,max_triggers=args.max_triggers,max_modes=args.max_modes,max_candidates=args.max_candidates,ttl=args.ttl)
    counts=Counter(); observations=0
    try:
        for epoch in range(args.epochs):
            with open(args.dataset,encoding='utf-8') as f:
                for number,line in enumerate(iter(lambda:f.readline(2*1024*1024+1),''),1):
                    if len(line)>2*1024*1024: raise ValueError(f'Line {number}: exceeds 2 MiB')
                    if not line.strip(): continue
                    try:
                        row=json.loads(line)
                        result=model.train(row['history'],row['trigger'],row['target'])
                    except (ValueError,TypeError,KeyError) as exc:
                        raise ValueError(f'Epoch {epoch+1}, line {number}: invalid vector training record') from exc
                    observations+=1; counts[result.get('event',result['status'])]+=1
                    if observations%args.checkpoint_every==0:
                        save(model,args.output)
                        print(json.dumps({'observations':observations,'counts':dict(counts)}),file=sys.stderr,flush=True)
    finally:
        save(model,args.output)
    result={'model':'Mikteeng RSPM','observations':observations,'training_steps_total':model.step,'counts':dict(counts),'contexts':len(model.contexts),'persistent_contexts':sum(c['persistent'] for c in model.contexts.values()),'checkpoint':str(args.output)}
    print(json.dumps(result,indent=2))
    return result

def main():
    p=argparse.ArgumentParser(prog='mikteeng-rspm'); sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train'); t.add_argument('dataset'); t.add_argument('--output',required=True); t.add_argument('--resume'); t.add_argument('--dim',type=int,default=64); t.add_argument('--epochs',type=int,default=1); t.add_argument('--checkpoint-every',type=int,default=1000)
    for flag,default in [('max-contexts',10),('max-triggers',8),('max-modes',4),('max-candidates',8),('ttl',30)]: t.add_argument('--'+flag,type=int,default=default)
    q=sub.add_parser('query'); q.add_argument('--checkpoint',required=True); q.add_argument('record',help='JSON file containing history and trigger')
    s=sub.add_parser('serve'); s.add_argument('--checkpoint',required=True); s.add_argument('--host',default='127.0.0.1'); s.add_argument('--port',type=int,default=int(os.environ.get('PORT','8000')))
    args=p.parse_args()
    if args.command=='train': train(args)
    elif args.command=='query':
        row=json.loads(Path(args.record).read_text()); print(json.dumps(jsonable(load(args.checkpoint).query(row['history'],row['trigger'])),indent=2))
    else:
        from .server import make_server
        server=make_server(load(args.checkpoint),args.host,args.port)
        print(f'Mikteeng RSPM on {args.host}:{server.server_port}',flush=True); server.serve_forever()
