"""TBIN 1.1 integrated experimental CPU toolkit. Not a general-purpose AI.

Train cross-modal associations from paired image/audio/text observations, retain
text statistics and relational demonstrations, analyze long WAV files, persist state.
"""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
from tbin_v25 import TBIN25
from tbin_v26 import load_audio

class TBIN:
    def __init__(self, seed=7):
        self.core=TBIN25(seed=seed)
        self.pairs_seen=0
        self.seed=seed
    def observe_text(self, text):
        self.core.observe_text(text)
    def observe_image(self, filename):
        with Image.open(filename) as im:
            rgb=np.asarray(im.convert('RGB'))
        return self.core.observe_image(rgb)
    def observe_audio(self, filename):
        return self.core.observe_audio(filename)
    @staticmethod
    def _resolve(value, root):
        p=Path(value)
        return p if p.is_absolute() else root/p
    def iter_rows(self, manifest):
        """Stream JSONL and decode one observation at a time; 2+ common modalities.
        group is optional supervision indicating equivalent observations.
        """
        path=Path(manifest).resolve()
        with path.open(encoding='utf-8') as source:
            for lineno,line in enumerate(source,1):
                if not line.strip():continue
                entry=json.loads(line); row={}
                if 'text' in entry:row['text']=str(entry['text'])
                if 'image' in entry:
                    with Image.open(self._resolve(entry['image'],path.parent)) as im:
                        row['image']=np.asarray(im.convert('RGB').resize((32,32)),dtype=np.uint8)
                if 'audio' in entry:row['audio']=load_audio(self._resolve(entry['audio'],path.parent))
                if len(row)<2:raise ValueError(f'Line {lineno}: provide at least two modalities')
                if 'group' in entry:
                    if not isinstance(entry['group'],str) or not entry['group']:raise ValueError('group must be a nonempty string')
                    row['group']=entry['group']
                yield row
    def load_rows(self, manifest):
        """Legacy convenience method. train() uses iter_rows instead."""
        return list(self.iter_rows(manifest))
    def train(self, manifest, epochs=80, lr=.025, batch_size=64, learn_text=False):
        # Text/graph memory grows with vocabulary and facts, so opt in explicitly.
        history=self.core.train_unlabeled_pairs(lambda:self.iter_rows(manifest),epochs=epochs,
                                               lr=lr,batch_size=batch_size)
        if learn_text:
            for row in self.iter_rows(manifest):
                if 'text' in row:self.observe_text(row['text'])
        count=self.core.training_stats['pairs'];self.pairs_seen+=count
        first=next(self.iter_rows(manifest))
        return {'pairs':count,'modalities':sorted(set(first)&{'text','audio','image'}),
                'history':history,'training':self.core.training_stats,
                'warning':'Paired supervision. Bounded batches; graph/text memory is optional and grows with knowledge.'}
    def retrieve(self, source_modality, source, target_modality, gallery):
        if source_modality==target_modality:raise ValueError('Choose different modalities')
        def read(m, item):
            if m=='text':return item
            if m=='audio':return load_audio(item)
            if m=='image':
                with Image.open(item) as im:return np.asarray(im.convert('RGB').resize((32,32)),dtype=np.uint8)
            raise ValueError('Unknown modality '+m)
        return self.core.cross_retrieve(source_modality,read(source_modality,source),target_modality,[read(target_modality,g) for g in gallery])
    def observe_fact(self, subject, relation, obj):
        self.core.reasoner.observe_fact(subject,relation,obj)
    def teach_relation(self, sentence, question=None, subject=None, answer=None):
        r=self.core.reasoner
        if question is not None and (subject is None or answer is None):raise ValueError('Need subject and answer')
        # Demonstration can supervise boundaries even in a longer sentence.
        if question is not None:
            from tbin_v20 import tok
            words=tok(sentence);start=tok(subject);end=tok(answer)
            matches=[]
            for i in range(len(words)-len(start)+1):
                if words[i:i+len(start)]!=start:continue
                for j in range(i+len(start)+1,len(words)-len(end)+1):
                    if words[j:j+len(end)]==end:matches.append(' '.join(words[i+len(start):j]))
            if len(matches)==1:r.observe_fact(subject,matches[0],answer)
        r.observe(sentence)
        if question is not None:
            if not r.demonstrate(question,subject,answer):return False
            answer=r.entity(answer)
            # Explicit demonstration trains a small ranking head against stored alternatives.
            negatives=[]
            for edges in r.edges.values():
                for _,target in edges:
                    if target!=answer and target not in negatives:negatives.append(target)
                    if len(negatives)>=16:break
                if len(negatives)>=16:break
            if negatives:r.train_neural([(question,answer,1)]+[(question,n,0) for n in negatives],epochs=30,lr=.05)
            return True
    def answer_relation(self, question, subject):
        return self.core.reasoner.answer(question,subject)
    def save(self, path):
        # Local trusted-file format; pickle must never be loaded from untrusted sources.
        with open(path,'wb') as f:pickle.dump({'core':self.core,'pairs_seen':self.pairs_seen,'seed':self.seed},f,protocol=5)
    @classmethod
    def load(cls,path):
        with open(path,'rb') as f:obj=pickle.load(f)
        m=cls(seed=obj['seed']);m.core=obj['core'];m.pairs_seen=obj['pairs_seen'];return m

def main():
    p=argparse.ArgumentParser(description='TBIN 1.1 experimental CPU multimodal toolkit')
    sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('train');a.add_argument('manifest');a.add_argument('--model',default='tbin_model.pkl');a.add_argument('--epochs',type=int,default=80);a.add_argument('--batch-size',type=int,default=64);a.add_argument('--learn-text',action='store_true')
    a=sub.add_parser('image');a.add_argument('file')
    a=sub.add_parser('audio');a.add_argument('file')
    a=sub.add_parser('retrieve');a.add_argument('model');a.add_argument('source_modality',choices=['text','image','audio']);a.add_argument('source');a.add_argument('target_modality',choices=['text','image','audio']);a.add_argument('gallery',nargs='+')
    a=sub.add_parser('relation');a.add_argument('model');a.add_argument('question');a.add_argument('subject')
    args=p.parse_args()
    if args.cmd=='train':
        m=TBIN();result=m.train(args.manifest,epochs=args.epochs,batch_size=args.batch_size,learn_text=args.learn_text);m.save(args.model);result['saved_model']=args.model
    elif args.cmd=='image':result=TBIN().observe_image(args.file)
    elif args.cmd=='audio':result=TBIN().observe_audio(args.file)
    elif args.cmd=='retrieve':result=TBIN.load(args.model).retrieve(args.source_modality,args.source,args.target_modality,args.gallery)
    else:result={'answer':TBIN.load(args.model).answer_relation(args.question,args.subject)}
    print(json.dumps(result,indent=2,default=str))
if __name__=='__main__':main()
