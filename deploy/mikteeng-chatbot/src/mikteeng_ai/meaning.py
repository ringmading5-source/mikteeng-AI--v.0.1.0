"""Supervised, order-sensitive span prediction and evidence-backed concept links.

Role names and annotations specify the task. Semantic answers are fitted from
examples; no action-to-consequence rules or grammar parser are included.
"""
import re
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np


def words(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('text must be nonempty')
    return re.findall(r'\w+|[^\w\s]', text.casefold())

class MeaningLearner:
    def __init__(self, max_span=4, threshold=.5):
        if not isinstance(max_span, int) or max_span < 1:
            raise ValueError('max_span must be a positive integer')
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must be between zero and one')
        self.max_span = max_span
        self.threshold = threshold
        self.vectorizer = DictVectorizer()
        self.heads = {}
        self.evidence = []
        self.concepts = defaultdict(list)

    def _candidates(self, tokens):
        return [(a,b) for a in range(len(tokens)) for b in range(a+1,min(len(tokens),a+self.max_span)+1)]

    @staticmethod
    def _features(tokens, span):
        a,b=span
        f={'span_length':b-a,'relative_position':a/max(1,len(tokens)-1)}
        for offset in range(-8,9):
            index=a+offset
            if 0<=index<len(tokens):f[f'offset:{offset}:{tokens[index]}']=1
        for offset in range(-3,4):
            index=b+offset
            if 0<=index<len(tokens):f[f'end:{offset}:{tokens[index]}']=1
        # Whole-observation context includes explicit conditions such as negation.
        for i,word in enumerate(tokens):
            f['context:'+word]=1
            if i+1<len(tokens):f['pair:'+word+' '+tokens[i+1]]=1
        return f

    def fit(self, data):
        rows=list(data)
        if not rows:raise ValueError('meaning training data is empty')
        feature_rows=[];annotations=[];evidence=[]
        for row in rows:
            if not isinstance(row,dict) or not {'text','slots'}<=row.keys():
                raise ValueError('each row requires text and slots')
            tokens=words(row['text']);slots=row['slots']
            if not isinstance(slots,dict) or not slots:
                raise ValueError('slots must be a nonempty dictionary')
            normalized={}
            for name,span in slots.items():
                if not isinstance(name,str) or not name:raise ValueError('slot names must be nonempty strings')
                if span is None:normalized[name]=None;continue
                if not isinstance(span,(list,tuple)) or len(span)!=2 or any(type(v)!=int for v in span):
                    raise ValueError('slot spans must be [start,end] or null')
                a,b=span
                if not 0<=a<b<=len(tokens) or b-a>self.max_span:
                    raise ValueError('invalid span or span exceeds max_span')
                normalized[name]=(a,b)
            candidates=self._candidates(tokens)
            feature_rows.extend(self._features(tokens,s) for s in candidates)
            annotations.extend((normalized,s) for s in candidates)
            evidence.append({'text':row['text'],'tokens':tokens,'slots':normalized,
                             'concepts':row.get('concepts',[])})
            if not isinstance(evidence[-1]['concepts'],list) or any(not isinstance(c,str) for c in evidence[-1]['concepts']):
                raise ValueError('concepts must be a list of strings')
        vectorizer=DictVectorizer();X=vectorizer.fit_transform(feature_rows)
        heads={}
        for name in sorted({n for entry in evidence for n in entry['slots']}):
            indices=[i for i,(slots,span) in enumerate(annotations) if name in slots]
            labels=[int(annotations[i][0][name]==annotations[i][1]) for i in indices]
            if len(set(labels))<2:raise ValueError('each slot needs positive and negative candidates: '+name)
            heads[name]=LogisticRegression(C=10,max_iter=1000,class_weight='balanced',random_state=42).fit(X[indices],labels)
        concepts=defaultdict(list)
        for index,entry in enumerate(evidence):
            for concept in entry['concepts']:concepts[concept.casefold()].append(index)
        self.vectorizer=vectorizer;self.heads=heads;self.evidence=evidence;self.concepts=concepts
        return self

    def predict(self,text):
        if not self.heads:raise RuntimeError('train meaning first')
        tokens=words(text);candidates=self._candidates(tokens)
        X=self.vectorizer.transform([self._features(tokens,s) for s in candidates])
        slots={}
        for name,head in self.heads.items():
            scores=head.predict_proba(X)[:,1];index=int(scores.argmax());a,b=candidates[index]
            score=float(scores[index])
            slots[name]={'value':' '.join(tokens[a:b]) if score>=self.threshold else None,
                         'span':[a,b] if score>=self.threshold else None,'score':score}
        return {'text':text,'tokens':tokens,'slots':slots}

    def describe_concept(self,concept):
        """Return supplied evidence; this is retrieval, not new inference."""
        words(concept)
        return [self.evidence[i] for i in self.concepts.get(concept.casefold(),[])]
