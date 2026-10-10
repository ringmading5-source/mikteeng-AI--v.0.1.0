"""CPU sequence-generation experiment. 256 UTF-8 bytes + end-of-response control.
Separate from TBIN's modality projections; no pretrained weights or external AI.
"""
import json,argparse
import numpy as np
class SequenceLearner:
    def __init__(self,seed=7):
        r=np.random.default_rng(seed);self.seed=seed
        self.p={k:np.asarray(v,dtype=np.float32) for k,v in {
          'E':r.normal(0,.1,(257,24)),'X':r.normal(0,.2,(24,48)),
          'H':r.normal(0,.07,(48,48)),'b':np.zeros(48),
          'O':r.normal(0,.05,(48,257)),'c':np.zeros(257)}.items()}
    def prefix(self,s):return list(('Question: '+s+'\nAnswer: ').encode())
    def loss_grad(self,prompt,response):
        prefix=self.prefix(prompt);target=list(response.encode())+[256];tokens=prefix+target[:-1]
        if len(tokens)>512:raise ValueError('Example exceeds 512 bytes')
        h=np.zeros((len(tokens)+1,48),np.float32)
        for t,x in enumerate(tokens):h[t+1]=np.tanh(self.p['E'][x]@self.p['X']+h[t]@self.p['H']+self.p['b'])
        start=len(prefix);active=h[start:];z=active@self.p['O']+self.p['c'];z-=z.max(1,keepdims=True)
        prob=np.exp(z);prob/=prob.sum(1,keepdims=True)
        loss=float(-np.log(prob[np.arange(len(target)),target]+1e-12).mean())
        g={k:np.zeros_like(v) for k,v in self.p.items()}
        prob[np.arange(len(target)),target]-=1;prob/=len(target)
        g['O']=active.T@prob;g['c']=prob.sum(0);dh=np.zeros_like(h);dh[start:]=prob@self.p['O'].T
        for t in range(len(tokens)-1,-1,-1):
            a=dh[t+1]*(1-h[t+1]**2);g['X']+=np.outer(self.p['E'][tokens[t]],a)
            g['H']+=np.outer(h[t],a);g['b']+=a;g['E'][tokens[t]]+=a@self.p['X'].T;dh[t]+=a@self.p['H'].T
        return loss,g
    def train(self,rows,epochs=120,lr=.003,batch_size=8):
        if not rows or epochs<1 or batch_size<1 or lr<=0:raise ValueError('Invalid training settings')
        for r in rows:
            if len(self.prefix(r['prompt']))+len(r['response'].encode())>512:raise ValueError('Example exceeds 512 bytes')
        rng=np.random.default_rng(self.seed);m={k:np.zeros_like(v) for k,v in self.p.items()};v={k:x.copy() for k,x in m.items()};step=0;history=[]
        for epoch in range(epochs):
            order=rng.permutation(len(rows));total=0
            for offset in range(0,len(order),batch_size):
                ids=order[offset:offset+batch_size];g={k:np.zeros_like(x) for k,x in self.p.items()}
                for i in ids:
                    loss,gi=self.loss_grad(rows[i]['prompt'],rows[i]['response']);total+=loss
                    for k in g:g[k]+=gi[k]/len(ids)
                scale=min(1.,1/max(np.sqrt(sum(float(np.sum(x*x)) for x in g.values())),1e-9));step+=1
                for k in g:
                    g[k]*=scale;m[k]=.9*m[k]+.1*g[k];v[k]=.999*v[k]+.001*g[k]**2
                    self.p[k]-=lr*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
            history.append(total/len(rows))
        return history
    def generate(self,prompt,max_bytes=128):
        if len(self.prefix(prompt))>512 or max_bytes<1:raise ValueError('Invalid length')
        h=np.zeros(48,np.float32)
        for x in self.prefix(prompt):h=np.tanh(self.p['E'][x]@self.p['X']+h@self.p['H']+self.p['b'])
        out=[];ended=False
        for _ in range(max_bytes):
            x=int(np.argmax(h@self.p['O']+self.p['c']))
            if x==256:ended=True;break
            out.append(x);h=np.tanh(self.p['E'][x]@self.p['X']+h@self.p['H']+self.p['b'])
        raw=bytes(out)
        try:text=raw.decode();valid=True
        except UnicodeDecodeError:text=raw.decode(errors='replace');valid=False
        return {'text':text,'ended':ended,'valid_utf8':valid}
    def save(self,path):np.savez_compressed(path,**self.p,seed=self.seed)
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f:
            obj=cls(int(f['seed']))
            for k in obj.p:
                if f[k].shape!=obj.p[k].shape or not np.isfinite(f[k]).all():raise ValueError('Invalid weights')
                obj.p[k]=f[k].astype(np.float32)
        return obj
if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('train');a.add_argument('data');a.add_argument('--model',default='sequence_model.npz');a.add_argument('--epochs',type=int,default=120)
    a=s.add_parser('generate');a.add_argument('model');a.add_argument('prompt');a=p.parse_args()
    if a.cmd=='train':
        with open(a.data) as f:rows=[json.loads(x) for x in f if x.strip()]
        m=SequenceLearner();h=m.train(rows,epochs=a.epochs);m.save(a.model);print(json.dumps({'first_loss':h[0],'last_loss':h[-1]}))
    else:print(json.dumps(SequenceLearner.load(a.model).generate(a.prompt),ensure_ascii=False))
