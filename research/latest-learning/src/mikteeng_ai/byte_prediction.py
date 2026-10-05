"""Supervised next-packet prediction from bytes and frozen pattern nodes."""
from collections import Counter
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import RidgeClassifier
from .byte_tokenizer import MultimodalByteTokenizer, BytePatternHierarchy

class ByteHierarchyPredictor:
    def __init__(self,context=3,alpha=1.0,depth=3,max_nodes=4096,include_hierarchy=True):
        if type(context)!=int or context<1:raise ValueError('context must be positive')
        self.context=context;self.alpha=alpha;self.depth=depth;self.max_nodes=max_nodes
        self.include_hierarchy=include_hierarchy
        self.tokenizer=MultimodalByteTokenizer()
    def features(self,context):
        if len(context)<self.context:raise ValueError('insufficient context')
        f={}
        for slot,text in enumerate(context[-self.context:]):
            packet=self.tokenizer.encode(text);values=packet.tokens.astype(int).tolist()
            f[f'{slot}:length']=len(values)/256
            for position,byte in enumerate(values):f[f'{slot}:byte:{position}:{byte}']=1
            for level in range(self.hierarchy.depth if self.include_hierarchy else 0):
                groups=[tuple(values[i:i+self.hierarchy.group_size]) for i in range(0,len(values),self.hierarchy.group_size)]
                known=self.hierarchy.patterns[level]
                ids=[known.get(group,-1) for group in groups]
                for position,node in enumerate(ids):
                    if node>=0:f[f'{slot}:L{level}:position:{position}:node:{node}']=1
                for node,count in Counter(ids).items():
                    if node>=0:f[f'{slot}:L{level}:count:{node}']=count
                values=ids
        return f
    def fit(self,sequences):
        sequences=[list(s) for s in sequences]
        if not sequences or any(len(s)<=self.context for s in sequences):raise ValueError('sequences too short')
        self.hierarchy=BytePatternHierarchy(depth=self.depth,max_nodes_per_level=self.max_nodes)
        for sequence in sequences:
            for text in sequence:self.hierarchy.observe(self.tokenizer.encode(text))
        features=[];targets=[]
        for sequence in sequences:
            for t in range(self.context,len(sequence)):
                features.append(self.features(sequence[t-self.context:t]))
                targets.append(self.tokenizer.encode(sequence[t]).tokens.astype(int).tolist())
        self.width=max(map(len,targets))+1
        self.vectorizer=DictVectorizer(sparse=True)
        x=self.vectorizer.fit_transform(features);self.heads=[]
        # -1 is an output stopping control, outside the 256 input byte vocabulary.
        y=np.full((len(targets),self.width),-1,dtype=int)
        for i,target in enumerate(targets):y[i,:len(target)]=target
        for position in range(self.width):
            labels=y[:,position]
            if len(np.unique(labels))==1:self.heads.append(int(labels[0]))
            else:self.heads.append(RidgeClassifier(alpha=self.alpha,solver='lsqr').fit(x,labels))
        self.training_examples=len(targets);return self
    def predict(self,context,steps=1):
        if type(steps)!=int or not 1<=steps<=64:raise ValueError('steps must be 1..64')
        context=list(context)[-self.context:];result=[]
        for _ in range(steps):
            x=self.vectorizer.transform([self.features(context)]);raw=[]
            for head in self.heads:
                byte=head if isinstance(head,int) else int(head.predict(x)[0])
                if byte==-1:break
                raw.append(byte)
            payload=bytes(raw)
            try:text=payload.decode('utf-8');status='decoded'
            except UnicodeDecodeError:text=None;status='invalid_utf8'
            result.append({'bytes':raw,'text':text,'status':status})
            if text is None:break
            context.append(text);context=context[-self.context:]
        return result
