"""Bounded internal rollouts of Mikteeng's learned transition mappings.
Generic goal/constraint evaluation; no stored complete answer or domain rules.
"""
from dataclasses import dataclass
from copy import deepcopy
from hashlib import blake2b
import json
import numpy as np
from .transitions import StateTransitionLearner,_code

@dataclass(frozen=True)
class DeliberationSelf:
    version: str
    observation_json: str
    goal_json: str
    actions_json: str
    constraints_json: str
    v_self: tuple

def state_vector(state,dimensions=128):
    vector=np.zeros(dimensions)
    for key,value in sorted(state.items()):
        digest=blake2b(_code([key,value]).encode(),digest_size=8).digest()
        vector[int.from_bytes(digest,'little')%dimensions]+=1
    return vector/np.linalg.norm(vector)

def build_self(ai,state,goal,actions,constraints=()):
    model=ai._transitions();model._check(state,'state');model._check(goal,'goal')
    if tuple(sorted(state))!=model.keys or not set(goal)<=set(state):raise ValueError('invalid state/goal schema')
    actions=list(actions);constraints=list(constraints)
    if not 1<=len(actions)<=64 or len(constraints)>16:raise ValueError('action/constraint count exceeds bounds')
    for action in actions:model._check(action,'action')
    for forbidden in constraints:
        model._check(forbidden,'constraint')
        if not set(forbidden)<=set(state):raise ValueError('constraint keys must be state keys')
    serialized=[_code(v) for v in [state,goal,actions,constraints]]
    if sum(map(len,serialized))>65536:raise ValueError('Self description exceeds budget')
    return DeliberationSelf(ai.transition_version,*serialized,tuple(state_vector(state)))

def respond(ai,self_state,*,max_depth=6,max_predictions=256,beam_width=16,min_score=.8):
    if not isinstance(self_state,DeliberationSelf) or self_state.version!=ai.transition_version:raise ValueError('stale or invalid Self')
    if type(max_depth)!=int or not 0<=max_depth<=16 or type(max_predictions)!=int or not 1<=max_predictions<=4096 or type(beam_width)!=int or not 1<=beam_width<=64:raise ValueError('invalid search bounds')
    if not 0<=min_score<=1:raise ValueError('invalid score threshold')
    model=ai._transitions();initial=json.loads(self_state.observation_json);goal=json.loads(self_state.goal_json)
    actions=json.loads(self_state.actions_json);forbidden=json.loads(self_state.constraints_json)
    def matches(state,query):return all(_code(state[k])==_code(value) for k,value in query.items())
    def allowed(state):return not any(matches(state,f) for f in forbidden)
    if not allowed(initial):return {'status':'invalid_initial_state','response':[],'predictions_evaluated':0}
    frontier=[(initial,[],[],1.)];visited={_code(initial)};evaluated=0;rejected=0;internal=[]
    for depth in range(max_depth+1):
        candidates=[]
        for state,path,trace,confidence in frontier:
            if matches(state,goal):
                return {'status':'generated_path','response':path,'final_state':state,'trace':trace,
                        'internal_predictions':internal,'predictions_evaluated':evaluated,'rejected_predictions':rejected,
                        'mechanism':'composition of learned one-step predictions','complete_answer_lookup':False,'scores_calibrated':False}
            if depth==max_depth:continue
            for action in actions:
                if evaluated>=max_predictions:return {'status':'prediction_budget','response':[],'predictions_evaluated':evaluated,'internal_predictions':internal}
                prediction=model.predict(state,action);evaluated+=1
                candidate=prediction['state']
                if prediction['status']!='predicted' or prediction['confidence']<min_score or not allowed(candidate):rejected+=1;continue
                identity=_code(candidate)
                if identity in visited:continue
                visited.add(identity)
                match_fraction=sum(_code(candidate[k])==_code(v) for k,v in goal.items())/len(goal)
                score=match_fraction-.01*(depth+1)+.001*min(confidence,prediction['confidence'])
                entry={'before':deepcopy(state),'action':deepcopy(action),'predicted_state':deepcopy(candidate),
                       'goal_match_fraction':match_fraction,'score':score,'v_self':state_vector(candidate).tolist(),
                       'transition_score':prediction['confidence']}
                internal.append(entry)
                candidates.append((score,candidate,path+[deepcopy(action)],trace+[entry],min(confidence,prediction['confidence'])))
        candidates.sort(key=lambda v:-v[0])
        frontier=[(state,path,trace,confidence) for _,state,path,trace,confidence in candidates[:beam_width]]
        if not frontier:break
    return {'status':'no_solution_within_bounds','response':[],'predictions_evaluated':evaluated,'rejected_predictions':rejected,'internal_predictions':internal}
