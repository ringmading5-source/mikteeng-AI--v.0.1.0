"""Supervised discourse relations and learned joining templates.

The model learns templates from labels; it does not infer universal causality.
Scores are uncalibrated. Text slots copy the supplied clauses verbatim.
"""
import json
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class JoiningLearner:
    @staticmethod
    def _input(left,right,context):
        for value in (left,right):
            if not isinstance(value,str) or not value.strip():raise ValueError('clauses must be nonempty strings')
        if not isinstance(context,str):raise ValueError('context must be a string')
        return 'LEFT '+left+' RIGHT '+right+' CONTEXT '+context
    def fit(self,data):
        rows=list(data)
        if not rows:raise ValueError('joining data is empty')
        inputs=[];relations=[];templates=[]
        for row in rows:
            inputs.append(self._input(row['left'],row['right'],row.get('context','')))
            if not isinstance(row['relation'],str) or not row['relation']:raise ValueError('relation must be nonempty')
            template=row['template']
            if template is not None:
                if not isinstance(template,list) or not template:raise ValueError('template must be a nonempty list or null')
                if any(not isinstance(p,dict) or len(p)!=1 or (set(p)!={'slot'} and set(p)!={'literal'}) for p in template):
                    raise ValueError('template pieces require slot or literal')
                if any('slot' in p and p['slot'] not in ('left','right') for p in template):raise ValueError('invalid text slot')
                if any('literal' in p and not isinstance(p['literal'],str) for p in template):raise ValueError('literal must be text')
                if sorted(p['slot'] for p in template if 'slot' in p)!=['left','right']:raise ValueError('template must use each clause once')
            relations.append(row['relation']);templates.append(json.dumps(template,sort_keys=True))
        if len(set(relations))<2:raise ValueError('at least two relationship classes required')
        vectorizer=FeatureUnion([('words',TfidfVectorizer(ngram_range=(1,2))),
                                 ('characters',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5)))])
        X=vectorizer.fit_transform(inputs)
        relation_head=LogisticRegression(C=20,max_iter=1000,random_state=42).fit(X,relations)
        # A second learned head predicts how the selected relation is expressed.
        realization_vectorizer=TfidfVectorizer(ngram_range=(1,2))
        R=realization_vectorizer.fit_transform(relations)
        realization_head=LogisticRegression(C=100,max_iter=1000,random_state=42).fit(R,templates)
        self.vectorizer=vectorizer;self.relation_head=relation_head
        self.realization_vectorizer=realization_vectorizer;self.realization_head=realization_head
        self.training_count=len(rows)
        return self
    def join(self,left,right,*,context='',min_score=.7):
        if not 0<=min_score<=1:raise ValueError('min_score must be between zero and one')
        if not hasattr(self,'relation_head'):raise RuntimeError('train joining first')
        X=self.vectorizer.transform([self._input(left,right,context)])
        p=self.relation_head.predict_proba(X)[0];index=int(p.argmax());relation=str(self.relation_head.classes_[index])
        rp=self.realization_head.predict_proba(self.realization_vectorizer.transform([relation]))[0]
        ri=int(rp.argmax());template=json.loads(self.realization_head.classes_[ri])
        scores={'relation_score':float(p[index]),'realization_score':float(rp[ri]),'calibrated':False}
        if template is None or min(scores['relation_score'],scores['realization_score'])<min_score:
            return {'text':'','status':'insufficient_evidence','relation':relation,'scores':scores}
        clauses={'left':left.strip().rstrip('.!?'),'right':right.strip().rstrip('.!?')}
        text=''.join(clauses[p['slot']] if 'slot' in p else p['literal'] for p in template)
        return {'text':text,'status':'joined','relation':relation,'scores':scores,'template':template}
