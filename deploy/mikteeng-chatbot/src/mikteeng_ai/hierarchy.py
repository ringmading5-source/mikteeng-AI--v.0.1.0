"""Learned subject -> relationship -> word-pointer explanation pipeline.

Supervised ordering, span extraction and word copying. Not a general causal
reasoner. No verb-specific transition rules or pretrained models are used.
"""
import re
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from .meaning import MeaningLearner, words

class BinaryHead:
    def fit(self, features, labels):
        if len(set(labels)) < 2:raise ValueError('head requires positive and negative examples')
        self.vectorizer=DictVectorizer()
        self.model=LogisticRegression(C=10,max_iter=1000,class_weight='balanced',random_state=42)
        self.model.fit(self.vectorizer.fit_transform(features),labels)
        return self
    def scores(self,features):
        return self.model.predict_proba(self.vectorizer.transform(features))[:,1]

class HierarchicalLearner:
    def __init__(self):
        self.meaning=None
        self.subject_head=None
        self.relationship_head=None
        self.stop_head=None
        self.word_head=None

    @staticmethod
    def _frame(fact):
        ts=words(fact['text'])
        values={}
        for key in ('subject','relationship','object'):
            span=fact['slots'][key]
            if span is None:raise ValueError('hierarchy facts need all three role spans')
            a,b=span;values[key]=' '.join(ts[a:b])
        return {'text':fact['text'],**values}

    @staticmethod
    def _plan_features(frame,question,history,count):
        q=set(words(question))
        subject=set(words(frame['subject']));obj=set(words(frame['object']))
        f={'step':len(history),'fraction':len(history)/max(1,count),
           'subject_query_overlap':len(q & subject), 'object_query_overlap':len(q & obj)}
        for w in words(frame['relationship']):f['relation:'+w]=1
        for w in q:f['question:'+w]=1
        if history:
            last=history[-1]
            for a in ('subject','relationship','object'):
                for b in ('subject','relationship','object'):
                    f['last_equal:'+a+':'+b]=float(last[a]==frame[b])
            for w in words(last['relationship']):f['previous_relation:'+w]=1
            for previous in history:
                for a in ('subject','object'):
                    for b in ('subject','object'):
                        f['history_equal:'+a+':'+b]=f.get('history_equal:'+a+':'+b,0)+float(previous[a]==frame[b])
        return f

    @staticmethod
    def _stop_features(history,remaining,total):
        f={'steps':len(history),'remaining':len(remaining),'fraction':len(history)/max(1,total)}
        if history:
            for word in words(history[-1]['relationship']):f['previous_relation:'+word]=1
            f['links_to_last_object']=sum(float(r['subject']==history[-1]['object']) for r in remaining)
        return f

    @staticmethod
    def _word_candidates(frame):
        result=[]
        for role in ('subject','relationship','object'):
            for offset,word in enumerate(words(frame[role])):
                result.append({'word':word,'role':role,'offset':offset})
        result.extend([{'word':'.','role':'punctuation','offset':0},
                       {'word':'<end>','role':'end','offset':0}])
        return result

    @staticmethod
    def _word_features(candidate,index,history,used):
        f={'role:'+candidate['role']:1,'offset':candidate['offset'],
           'emitted':len(history),'used':float(index in used)}
        for back,previous in enumerate(history[-2:][::-1]):
            f[f'previous:{back}:role:'+previous['role']]=1
            f[f'previous:{back}:offset']=previous['offset']
            f[f'previous:{back}:same_role']=float(previous['role']==candidate['role'])
            f[f'previous:{back}:offset_difference']=candidate['offset']-previous['offset']
        return f

    def fit(self,data):
        rows=list(data)
        if not rows:raise ValueError('hierarchical training data is empty')
        meaning_rows=[];subject_f=[];subject_y=[];relation_f=[];relation_y=[]
        stop_f=[];stop_y=[];word_f=[];word_y=[]
        for row in rows:
            words(row['question'])
            facts=row['facts'];order=row['explanation_order']
            if not facts or not order or len(set(order))!=len(order) or any(type(i)!=int or not 0<=i<len(facts) for i in order):
                raise ValueError('explanation_order must contain unique valid fact indices')
            frames=[self._frame(f) for f in facts]
            meaning_rows.extend(facts)
            history=[];used=set()
            for selected in order:
                remaining=[f for i,f in enumerate(frames) if i not in used]
                stop_f.append(self._stop_features(history,remaining,len(frames)));stop_y.append(0)
                target=frames[selected]
                for i,frame in enumerate(frames):
                    if i in used:continue
                    features=self._plan_features(frame,row['question'],history,len(frames))
                    subject_f.append(features);subject_y.append(int(frame['subject']==target['subject']))
                    # Train relationship choice across all candidates, then constrain
                    # its prediction to the independently predicted subject.
                    relation_f.append(features);relation_y.append(int(i==selected))
                history.append(target);used.add(selected)
                candidates=self._word_candidates(target)
                # Default supervised word order follows the annotated fact's text.
                # Optional explicit permutation can teach a different rendering.
                emission=facts[selected].get('word_order',list(range(len(candidates))))
                if sorted(emission)!=list(range(len(candidates))):raise ValueError('word_order must permute every word candidate')
                emitted=[];word_used=set()
                for correct in emission:
                    for i,candidate in enumerate(candidates):
                        word_f.append(self._word_features(candidate,i,emitted,word_used));word_y.append(int(i==correct))
                    emitted.append(candidates[correct]);word_used.add(correct)
            remaining=[f for i,f in enumerate(frames) if i not in used]
            stop_f.append(self._stop_features(history,remaining,len(frames)));stop_y.append(1)
        candidate=HierarchicalLearner()
        candidate.meaning=MeaningLearner(max_span=4,threshold=.5).fit(meaning_rows)
        candidate.subject_head=BinaryHead().fit(subject_f,subject_y)
        candidate.relationship_head=BinaryHead().fit(relation_f,relation_y)
        candidate.stop_head=BinaryHead().fit(stop_f,stop_y)
        candidate.word_head=BinaryHead().fit(word_f,word_y)
        self.__dict__.update(candidate.__dict__)
        return self

    def _render(self,frame):
        candidates=self._word_candidates(frame);history=[];used=set();out=[];trace=[]
        for _ in range(len(candidates)):
            available=[i for i in range(len(candidates)) if i not in used]
            scores=self.word_head.scores([self._word_features(candidates[i],i,history,used) for i in available])
            best=int(np.argmax(scores));index=available[best];candidate=candidates[index]
            trace.append({'word':candidate['word'],'role':candidate['role'],'score':float(scores[best])})
            if candidate['role']=='end':break
            out.append(candidate['word']);history.append(candidate);used.add(index)
        return re.sub(r'\s+([.,!?])',r'\1',' '.join(out)),trace

    def represent(self,passage):
        words(passage)
        if self.meaning is None:raise RuntimeError('train hierarchy first')
        sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+',passage.strip()) if s.strip()]
        frames=[];extractions=[]
        for sentence in sentences:
            result=self.meaning.predict(sentence);extractions.append(result)
            if any(result['slots'][role]['value'] is None for role in ('subject','relationship','object')):continue
            frames.append({'text':sentence,**{role:result['slots'][role]['value'] for role in ('subject','relationship','object')}})
        return {"frames":frames,"extractions":extractions}

    def explain(self,passage,question,*,max_sentences=12):
        return self.explain_representation(self.represent(passage),question,max_sentences=max_sentences)

    def explain_representation(self,representation,question,*,max_sentences=12):
        words(question)
        if type(max_sentences)!=int or max_sentences<1:raise ValueError('max_sentences must be positive')
        frames=representation['frames'];extractions=representation['extractions']
        history=[];used=set();outputs=[];trace=[];status='no_frames'
        for step in range(min(max_sentences,len(frames))):
            remaining=[i for i in range(len(frames)) if i not in used]
            if not remaining:status='exhausted';break
            stop=float(self.stop_head.scores([self._stop_features(history,[frames[i] for i in remaining],len(frames))])[0])
            if stop>=.5:status='learned_stop';break
            features=[self._plan_features(frames[i],question,history,len(frames)) for i in remaining]
            subjects=self.subject_head.scores(features)
            subject_index=int(np.argmax(subjects));subject=frames[remaining[subject_index]]['subject']
            eligible=[i for i in remaining if frames[i]['subject']==subject]
            relations=self.relationship_head.scores([self._plan_features(frames[i],question,history,len(frames)) for i in eligible])
            best=int(np.argmax(relations));index=eligible[best];frame=frames[index]
            sentence,word_trace=self._render(frame)
            trace.append({'step':step,'subject':subject,'relationship':frame['relationship'],
                          'object':frame['object'],'evidence_index':index,'subject_score':float(subjects[subject_index]),
                          'relationship_score':float(relations[best]),'stop_score':stop,'word_trace':word_trace})
            outputs.append(sentence);history.append(frame);used.add(index);status='sentence_limit' if step+1==max_sentences else 'exhausted'
        return {'text':' '.join(outputs),'status':status,'sentence_trace':trace,'extractions':extractions}
