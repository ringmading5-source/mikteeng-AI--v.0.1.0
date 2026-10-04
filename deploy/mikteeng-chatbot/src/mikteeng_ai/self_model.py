"""Question-independent predicted representation plus learned unknown selection."""
from dataclasses import dataclass
from copy import deepcopy
import uuid,json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from .hierarchy import BinaryHead
from .meaning import words

@dataclass(frozen=True)
class PredictedSelf:
    identifier: str
    representation_version: str
    transition_version: str | None
    observations: tuple
    representation: dict
    state: dict | None
    actions: tuple
    goal: dict | None
    predicted_outcome: dict | None
    provenance: str = 'learned_predictions_from_observations'

class GapPredictor:
    @staticmethod
    def features(frame,question,slot):
        q=set(words(question));f={'slot:'+slot:1}
        for role in ('subject','relationship','object'):
            f['overlap:'+role]=len(set(words(frame[role])) & q)
            f[slot+':overlap:'+role]=f['overlap:'+role]
            for word in words(frame[role]):
                if word in q:f['matched:'+role+':'+word]=1
        for word in q:f['question:'+word]=1
        for word in words(frame['relationship']):f['relationship:'+word]=1
        return f
    def fit(self,data):
        rows=list(data)
        if not rows:raise ValueError('gap training data is empty')
        labels=[];questions=[];features=[];targets=[]
        for row in rows:
            words(row['question']);kind=row['kind']
            if kind not in ('fact','explanation','state','outcome','plan'):raise ValueError('invalid gap kind')
            slot=row.get('slot') if kind=='fact' else row.get('field') if kind in ('state','outcome') else None
            if kind=='fact':
                if slot not in ('subject','relationship','object'):raise ValueError('fact slot is invalid')
                frames=row['frames'];target=row['frame_index']
                if target is not None and (type(target)!=int or not 0<=target<len(frames)):raise ValueError('frame_index is invalid')
                for i,frame in enumerate(frames):
                    features.append(self.features(frame,row['question'],slot));targets.append(int(i==target))
            if kind in ('state','outcome') and (not isinstance(slot,str) or not slot):raise ValueError('state questions need a field')
            labels.append(json.dumps([kind,slot]));questions.append(row['question'])
        vectorizer=TfidfVectorizer(ngram_range=(1,2),analyzer='word')
        X=vectorizer.fit_transform(questions)
        head=LogisticRegression(C=30,max_iter=1000,random_state=42).fit(X,labels)
        frame_head=BinaryHead().fit(features,targets)
        self.vectorizer=vectorizer;self.head=head;self.frame_head=frame_head
        return self
    def predict(self,question):
        words(question);X=self.vectorizer.transform([question]);p=self.head.predict_proba(X)[0];i=int(p.argmax())
        kind,slot=json.loads(self.head.classes_[i])
        return {'kind':kind,'slot':slot,'score':float(p[i])}


def build_self(ai,observations,*,state=None,actions=None,goal=None):
    model=getattr(ai,'hierarchical_model',None)
    if model is None:raise RuntimeError('train hierarchy first')
    observations=(observations,) if isinstance(observations,str) else tuple(observations)
    if not observations:raise ValueError('observations are empty')
    for observation in observations:words(observation)
    if not hasattr(ai,'representation_version'):ai.representation_version=uuid.uuid4().hex
    representation=model.represent(' '.join(observations))
    actions=tuple(deepcopy(list(actions or [])))
    outcome=ai.predict_outcome(state,actions) if state is not None and actions else None
    return PredictedSelf(uuid.uuid4().hex,ai.representation_version,getattr(ai,'transition_version',None),
                         observations,deepcopy(representation),deepcopy(state),actions,deepcopy(goal),deepcopy(outcome))

def validate_self(ai,predicted_self):
    if not isinstance(predicted_self,PredictedSelf):raise ValueError('expected a PredictedSelf representation')
    if predicted_self.representation_version!=getattr(ai,'representation_version',None):
        raise ValueError('representation is stale or belongs to a different model; rebuild self')


def respond_self(ai,predicted_self,question,*,min_score=.65,threshold=.9,joining_contexts=None):
    if not 0<=min_score<=1 or not 0<=threshold<=1:raise ValueError('scores must be between zero and one')
    validate_self(ai,predicted_self)
    gap=getattr(ai,'gap_model',None)
    if gap is None:raise RuntimeError('train gap prediction first')
    unknown=gap.predict(question)
    response={'self_id':predicted_self.identifier,'unknown':unknown,'text':'','status':'unsupported_gap',
              'calibrated':False}
    if unknown['score']<min_score:response['status']='uncertain_question';return response
    kind=unknown['kind'];slot=unknown['slot'];frames=predicted_self.representation['frames']
    if kind=='fact':
        if not frames:return response
        scores=gap.frame_head.scores([gap.features(f,question,slot) for f in frames]);index=int(np.argmax(scores))
        response['support_score']=float(scores[index])
        if scores[index]<min_score:response['status']='insufficient_support';return response
        response.update(text=frames[index][slot],status='answered',support=deepcopy(frames[index]),frame_index=index)
    elif kind=='explanation':
        result=ai.hierarchical_model.explain_representation(deepcopy(predicted_self.representation),question)
        result=ai._assess_prediction(result,threshold=threshold)
        response.update(text=result['text'],status=result['status'],explanation=result)
        if joining_contexts is not None:
            joined=ai._compose_prediction(result,contexts=joining_contexts)
            response.update(text=joined['text'],status=joined['status'],joined_explanation=joined)
    else:
        if predicted_self.transition_version!=getattr(ai,'transition_version',None):
            response['status']='stale_state_prediction';return response
        state=predicted_self.state
        if kind=='outcome':
            outcome=predicted_self.predicted_outcome
            if outcome is None or outcome['status']!='predicted':return response
            state=outcome['final_state'];response['outcome']=deepcopy(outcome)
        if kind in ('state','outcome'):
            if state is None or slot not in state:return response
            response.update(text=str(state[slot]),status='answered',value=deepcopy(state[slot]))
        elif kind=='plan':
            if state is None or predicted_self.goal is None:return response
            plan=ai.plan(state,predicted_self.goal,predicted_self.actions)
            response.update(text=json.dumps(plan['actions']),status=plan['status'],plan=plan)
    return response
