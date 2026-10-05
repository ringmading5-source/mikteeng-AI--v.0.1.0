"""Supervised discrete state changes, bounded planning, and feedback learning.

Actions are structured inputs. No real-world tools are executed. Predicted
confidence is an uncalibrated tree vote, not a correctness guarantee.
"""
import json
from collections import deque
from copy import deepcopy
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import ExtraTreesClassifier


def _code(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False)

class StateTransitionLearner:
    def __init__(self):
        self.rows=[];self.heads={};self.keys=()

    @staticmethod
    def _check(value,name):
        if not isinstance(value,dict) or not value:raise ValueError(name+' must be a nonempty dictionary')
        if any(not isinstance(k,str) or not isinstance(v,(str,int,float,bool,type(None))) for k,v in value.items()):
            raise ValueError(name+' must have string keys and scalar JSON values')
        json.dumps(value,allow_nan=False)

    @staticmethod
    def _features(state,action):
        f={}
        for k,v in state.items():f['state:'+k+':'+_code(v)]=1
        for k,v in action.items():f['action:'+k+':'+_code(v)]=1
        for sk,sv in state.items():
            for ak,av in action.items():f['equal:'+sk+':'+ak]=float(_code(sv)==_code(av))
        return f

    @staticmethod
    def _target(key,value,state,action):
        # Generic update representations, not action-specific rules.
        if _code(value)==_code(state[key]):return _code(['keep',key])
        for ak,av in action.items():
            if _code(value)==_code(av):return _code(['action',ak])
        for sk,sv in state.items():
            if _code(value)==_code(sv):return _code(['state',sk])
        return _code(['literal',value])

    def fit(self,data):
        rows=deepcopy(list(data))
        if not rows:raise ValueError('transition data is empty')
        for row in rows:
            for name in ('state','action','next_state'):self._check(row[name],name)
            if set(row['state'])!=set(row['next_state']):raise ValueError('state schema must be stable')
        keys=tuple(sorted(rows[0]['state']))
        if any(tuple(sorted(r['state']))!=keys for r in rows):raise ValueError('all rows must share state keys')
        vectorizer=DictVectorizer();X=vectorizer.fit_transform([self._features(r['state'],r['action']) for r in rows])
        heads={}
        for key in keys:
            labels=[self._target(key,r['next_state'][key],r['state'],r['action']) for r in rows]
            heads[key]=ExtraTreesClassifier(n_estimators=64,min_samples_leaf=1,random_state=42).fit(X,labels)
        self.rows=rows;self.keys=keys;self.vectorizer=vectorizer;self.heads=heads
        return self

    def predict(self,state,action):
        if not self.heads:raise RuntimeError('train transitions first')
        self._check(state,'state');self._check(action,'action')
        if tuple(sorted(state))!=self.keys:raise ValueError('state keys differ from training')
        X=self.vectorizer.transform([self._features(state,action)]);result={};trace={}
        for key,head in self.heads.items():
            probabilities=head.predict_proba(X)[0];index=int(probabilities.argmax())
            operation,argument=json.loads(head.classes_[index])
            source=state if operation in ('keep','state') else action
            if operation!='literal' and argument not in source:
                return {'state':deepcopy(state),'status':'missing_action_parameter','confidence':0.,'trace':trace}
            result[key]=argument if operation=='literal' else source[argument]
            trace[key]={'operation':operation,'argument':argument,'score':float(probabilities[index])}
        return {'state':result,'status':'predicted','confidence':min(t['score'] for t in trace.values()),'trace':trace}

    def rollout(self,state,actions,*,min_score=0.):
        if not 0<=min_score<=1:raise ValueError('min_score must be between zero and one')
        current=deepcopy(state);steps=[]
        for action in list(actions):
            prediction=self.predict(current,action)
            if prediction['status']!='predicted' or prediction['confidence']<min_score:
                return {'initial_state':state,'final_state':current,'steps':steps,'status':'uncertain_transition'}
            steps.append({'before':current,'action':deepcopy(action),**prediction});current=prediction['state']
        return {'initial_state':deepcopy(state),'final_state':current,'steps':steps,'status':'predicted'}

    def plan(self,state,goal,actions,*,max_depth=6,max_nodes=1000,min_score=.9):
        self._check(state,'state');self._check(goal,'goal')
        if not set(goal)<=set(state):raise ValueError('goal keys must be state keys')
        if type(max_depth)!=int or max_depth<0 or type(max_nodes)!=int or max_nodes<1:raise ValueError('invalid planning bounds')
        if not 0<=min_score<=1:raise ValueError('min_score must be between zero and one')
        actions=deepcopy(list(actions))
        for action in actions:self._check(action,'action')
        def reached(s):return all(_code(s[k])==_code(v) for k,v in goal.items())
        queue=deque([(deepcopy(state),[],[])]);visited={_code(state)};expanded=0
        while queue and expanded<max_nodes:
            current,path,trace=queue.popleft()
            if reached(current):return {'status':'plan_found','actions':path,'final_state':current,'trace':trace,'expanded':expanded}
            expanded+=1
            if len(path)>=max_depth:continue
            for action in actions:
                prediction=self.predict(current,action)
                if prediction['status']!='predicted' or prediction['confidence']<min_score:continue
                key=_code(prediction['state'])
                if key in visited:continue
                if len(visited)>=max_nodes:
                    return {'status':'node_limit','actions':[],'expanded':expanded}
                visited.add(key);queue.append((prediction['state'],path+[action],trace+[prediction]))
        return {'status':'no_plan_within_bounds' if not queue else 'node_limit','actions':[],'expanded':expanded}

    def observe(self,state,action,observed_state,*,learn=True):
        prediction=self.predict(state,action);self._check(observed_state,'observed_state')
        if tuple(sorted(observed_state))!=self.keys:raise ValueError('observed state schema differs')
        differences={k:{'predicted':prediction['state'][k],'observed':observed_state[k]}
                     for k in self.keys if _code(prediction['state'][k])!=_code(observed_state[k])}
        if learn:self.fit(self.rows+[{'state':state,'action':action,'next_state':observed_state}])
        return {'prediction':prediction,'matched':not differences,'differences':differences,
                'learned':learn,'training_examples':len(self.rows)}

class FrameActionLearner:
    """Learn the bridge from explanation frames to structured actions."""
    def fit(self,data):
        rows=deepcopy(list(data))
        if not rows:raise ValueError('frame/action training data is empty')
        keys=tuple(sorted(rows[0]['action']))
        features=[];labels={key:[] for key in keys}
        for row in rows:
            frame=row['frame'];action=row['action']
            StateTransitionLearner._check(frame,'frame');StateTransitionLearner._check(action,'action')
            if set(frame)!={'subject','relationship','object'}:raise ValueError('frame requires subject, relationship and object')
            if tuple(sorted(action))!=keys:raise ValueError('action schema must be stable')
            features.append(StateTransitionLearner._features(frame,{}))
            for key,value in action.items():
                source=next((k for k,v in frame.items() if _code(v)==_code(value)),None)
                labels[key].append(_code(['frame',source]) if source else _code(['literal',value]))
        vectorizer=DictVectorizer();X=vectorizer.fit_transform(features)
        heads={key:ExtraTreesClassifier(n_estimators=64,random_state=42).fit(X,values) for key,values in labels.items()}
        self.vectorizer=vectorizer;self.heads=heads
        return self
    def predict(self,frame):
        StateTransitionLearner._check(frame,'frame')
        if set(frame)!={'subject','relationship','object'}:raise ValueError('frame requires subject, relationship and object')
        X=self.vectorizer.transform([StateTransitionLearner._features(frame,{})]);action={};scores=[]
        for key,head in self.heads.items():
            p=head.predict_proba(X)[0];index=int(p.argmax());operation,arg=json.loads(head.classes_[index])
            action[key]=frame[arg] if operation=='frame' else arg;scores.append(float(p[index]))
        return {'action':action,'confidence':min(scores)}
