"""Public API. Serialized models must come from a trusted source."""
from pathlib import Path
import json,pickle,zipfile,random
from ._models.generator import Generator,tokens
from ._models.sentence_generator import SentenceGenerator
from ._models.subject_generator import SubjectGenerator
from ._models.revisable_roles import RevisableRoles
from ._models.predictor import Predictor
from ._models.learner import PassageLearner
from ._models.optimized import CharacterStream

class RoleSession:
    """Accumulate words and revise roles within the current sentence."""
    def __init__(self,model): self.model=model;self.words=[]
    def push(self,word):
        if not isinstance(word,str) or not word.strip():raise ValueError('word must be a nonempty string')
        if self.words and self.words[-1] in '.!?':self.words=[]
        self.words.extend(tokens(word))
        return self.model.roles.predictions(self.words)
    def reset(self):self.words=[]

class MikteengAI:
    """One interface to separately trained generation, role and passage heads."""
    def __init__(self):
        self.generator=None;self.roles=None;self.sentence_model=None;self.passage_model=None
    @staticmethod
    def _text(text):
        if not isinstance(text,str) or not text.strip():raise ValueError('text must be a nonempty string')
    @staticmethod
    def _rows(data,keys):
        rows=list(data)
        if not rows:raise ValueError('training data is empty')
        for row in rows:
            if not isinstance(row,dict) or any(k not in row for k in keys):raise ValueError('each row requires: '+', '.join(keys))
        return rows
    def train(self,data,*,task='generation',**kwargs):
        methods={'generation':self.train_generation,'roles':self.train_roles,'sentences':self.train_sentences,'passages':self.train_passages}
        if task not in methods:raise ValueError('task must be generation, roles, sentences or passages')
        return methods[task](data,**kwargs)
    def train_code_predictions(self, examples):
        from .code_prediction import CodePredictionModel
        self.code_prediction_model=CodePredictionModel().fit(examples)
        return self
    def build_code(self, request):
        model=getattr(self,'code_prediction_model',None)
        if model is None:raise RuntimeError('train code prediction first')
        return model.generate(request)
    def train_response_control(self, checked_examples):
        from .response_control import LearnedResponseControl
        self.response_control=LearnedResponseControl().fit(checked_examples)
        return self
    def ask_checked(self, question, *, predicted_self, max_words=85):
        control=getattr(self,'response_control',None)
        if control is None:raise RuntimeError('train checked response control first')
        proposal=self._ask_raw(question,predicted_self=predicted_self,max_words=max_words)
        model=self.sentence_pattern_model
        observations=[model.decode(v) for v in predicted_self.observations]
        decision=control.decide(observations,question,proposal['text'])
        return {'text':proposal['text'] if decision=='answer' else ('insufficient evidence.' if decision=='withhold' else ''),
                'decision':decision,'calibrated':False}
    def train_character_conditioned_questions(self, examples):
        import copy
        from .character_prediction import CharacterPredictionModel
        from .question_prediction import QuestionConditionedPredictor
        rows=list(examples)
        model=getattr(self,'sentence_pattern_model',None)
        if model is None:raise RuntimeError('train sentence representation first')
        new_model=copy.deepcopy(model)
        texts=[text for row in rows for text in row['observations']]+[row['question'] for row in rows]+[row['answer'] for row in rows]
        new_model.character_model=CharacterPredictionModel().fit(texts)
        predictor=QuestionConditionedPredictor(new_model).fit(rows)
        self.sentence_pattern_model=new_model
        self.character_conditioned_predictor=predictor
        return self
    def predict_next_character(self, text):
        model=getattr(getattr(self,'sentence_pattern_model',None),'character_model',None)
        if model is None:raise RuntimeError('train character prediction first')
        return model.predict_next(text)
    def train_relational_predictions(self, examples):
        from .relational_prediction import RelationalPredictor
        model=getattr(self,'sentence_pattern_model',None)
        if model is None:raise RuntimeError('train sentence representation first')
        self.relational_predictor=RelationalPredictor(model).fit(examples)
        return self
    def train_subject_conditioned_questions(self, data, *, role_model=None):
        import copy
        from .question_prediction import QuestionConditionedPredictor
        model=getattr(self,'sentence_pattern_model',None)
        roles=role_model if role_model is not None else self.roles
        if model is None:raise RuntimeError('train sentence patterns first')
        if roles is None:raise RuntimeError('train or supply the existing role model first')
        # Train a replacement before attaching; preserve existing role/sequence weights.
        predictor=QuestionConditionedPredictor(model,self_source='hybrid',role_model=copy.deepcopy(roles)).fit(data)
        self.subject_conditioned_predictor=predictor
        return self
    def train_question_conditioned_predictions(self, data):
        from .question_prediction import QuestionConditionedPredictor
        model=getattr(self,'sentence_pattern_model',None)
        if model is None:raise RuntimeError('train sentence patterns first')
        self.question_conditioned_predictor=QuestionConditionedPredictor(model).fit(data)
        return self
    def predict_question_answer(self, question, *, predicted_self):
        model=getattr(self,'question_conditioned_predictor',None)
        if model is None:raise RuntimeError('train question-conditioned predictions first')
        return model.respond(predicted_self,question)
    def train_recursive_level_selection(self, observed_passages, *, seed=42):
        from .level_selection import LearnedLevelSelector
        model=getattr(self,'sentence_pattern_model',None)
        if model is None or not hasattr(model.learner.higher,'levels'):raise RuntimeError('load a recursive sentence model first')
        system=model.learner.higher
        if hasattr(system,'system'):system=system.system
        x=[];y=[];minimum=model.learner.observation_window+model.learner.prediction_window-1
        for passage in observed_passages:
            passage=list(passage)
            for t in range(minimum,len(passage)):
                snapshot=model.build_self(passage[:t]);x.append(snapshot.pattern);y.append(model.encode([passage[t]])[0])
        if not x:raise ValueError('observed feedback passages are too short or empty')
        selector=LearnedLevelSelector(system,seed=seed).fit(x,y)
        model.learner.higher=selector
        return self
    def learn_response_patterns(self, feedback):
        from .pattern_questions import PatternQuestionReader
        import copy
        reader=getattr(self,'pattern_question_reader',None)
        if reader is None:raise RuntimeError('train pattern questions first')
        if not hasattr(reader,'training_rows'):raise RuntimeError('legacy reader: refit with original question examples before feedback')
        rows=list(feedback)
        if not rows:raise ValueError('feedback is empty')
        checked=[];events=[]
        for row in rows:
            if row.get('verified') is not True:raise ValueError('feedback needs explicit verified=True and a checked answer')
            if 'answer_sentence' not in row:raise ValueError('checked answer_sentence is required; None means absent evidence')
            observations=list(row['observations']);question=row['question']
            snapshot=reader.sentence_model.build_self(observations)
            before=reader.respond(snapshot,question)
            checked.append({'observations':observations,'question':question,'answer_sentence':row['answer_sentence']})
            events.append({'observations':observations,'question':question,'previous_response':before,
                           'checked_answer_sentence':row['answer_sentence'],'verified':True})
        # Fit a replacement before committing. Invalid feedback cannot partially update weights.
        new_reader=PatternQuestionReader(reader.sentence_model,reader.use_prediction_history)
        new_reader.fit(copy.deepcopy(reader.training_rows)+checked)
        new_reader.allow_abstention=True
        self.pattern_question_reader=new_reader
        self.checked_response_events=getattr(self,'checked_response_events',[])+events
        return self
    def train_pattern_questions(self, data):
        from .pattern_questions import PatternQuestionReader
        model=getattr(self,'sentence_pattern_model',None)
        if model is None:raise RuntimeError('train sentence patterns first')
        self.pattern_question_reader=PatternQuestionReader(model).fit(data)
        return self
    def ask_pattern_self(self, question, *, predicted_self):
        reader=getattr(self,'pattern_question_reader',None)
        if reader is None:raise RuntimeError('train pattern questions first')
        return reader.respond(predicted_self,question)
    def train_adaptive_patterns(self, observations, *, self_observations, **kwargs):
        from .pattern_bank import AdaptivePredictionPatternLearner
        self.prediction_pattern_model=AdaptivePredictionPatternLearner(**kwargs).fit(observations,self_observations=self_observations)
        return self
    def train_sentence_patterns(self, observations, *, self_observations, **kwargs):
        from .sentence_patterns import SentencePatternLearner
        self.sentence_pattern_model=SentencePatternLearner(**kwargs).fit(observations,self_observations=self_observations)
        return self
    def build_sentence_prediction_self(self, observations):
        model=getattr(self,'sentence_pattern_model',None)
        if model is None:raise RuntimeError('train sentence patterns first')
        return model.build_self(observations)
    def continue_sentence_patterns(self, learned_self, *, steps=1):
        model=getattr(self,'sentence_pattern_model',None)
        if model is None:raise RuntimeError('train sentence patterns first')
        return model.respond(learned_self,steps=steps)
    def train_prediction_patterns(self, observations, *, self_observations, **kwargs):
        from .prediction_patterns import PredictionPatternLearner
        self.prediction_pattern_model=PredictionPatternLearner(**kwargs).fit(
            observations,self_observations=self_observations)
        return self
    def build_prediction_self(self, observations):
        model=getattr(self,'prediction_pattern_model',None)
        if model is None:raise RuntimeError('train prediction patterns first')
        return model.build_self(observations)
    def respond_prediction_self(self, learned_self, *, steps=1):
        model=getattr(self,'prediction_pattern_model',None)
        if model is None:raise RuntimeError('train prediction patterns first')
        return model.respond(learned_self,steps=steps)
    def train_adaptive_representation(self, observations, **kwargs):
        from .adaptive import AdaptiveRepresentation
        options={key:kwargs.pop(key) for key in ('dimensions','max_features','seed') if key in kwargs}
        self.adaptive_model=AdaptiveRepresentation(**options).fit(observations,**kwargs)
        return self
    def train_adaptive_readout(self, data):
        model=getattr(self,'adaptive_model',None)
        if model is None:raise RuntimeError('train adaptive representation first')
        model.fit_reader(data)
        return self
    def build_adaptive_self(self, observations):
        model=getattr(self,'adaptive_model',None)
        if model is None:raise RuntimeError('train adaptive representation first')
        return model.build(observations)
    def respond_adaptive(self, predicted_self, question, *, min_score=.65):
        model=getattr(self,'adaptive_model',None)
        if model is None:raise RuntimeError('train adaptive representation first')
        return model.respond(predicted_self,question,min_score=min_score)
    def train_gap_prediction(self, data):
        from .self_model import GapPredictor
        self.gap_model=GapPredictor().fit(data)
        return self
    def build_self(self, observations, *, state=None, actions=None, goal=None):
        from .self_model import build_self
        return build_self(self,observations,state=state,actions=actions,goal=goal)
    def respond_self(self, predicted_self, question, *, min_score=.65, threshold=.9, joining_contexts=None):
        from .prediction_patterns import PredictionPatternSelf
        if isinstance(predicted_self,PredictionPatternSelf):
            if getattr(self,'subject_conditioned_predictor',None) is not None:
                return self.subject_conditioned_predictor.respond(predicted_self,question)
            if getattr(self,'character_conditioned_predictor',None) is not None:
                return self.character_conditioned_predictor.respond(predicted_self,question)
            if getattr(self,'relational_predictor',None) is not None:
                return self.relational_predictor.respond(predicted_self,question)
            if getattr(self,'question_conditioned_predictor',None) is not None:
                return self.predict_question_answer(question,predicted_self=predicted_self)
            return self.ask_pattern_self(question,predicted_self=predicted_self)
        from .adaptive import AdaptiveSelf
        if isinstance(predicted_self,AdaptiveSelf):
            if joining_contexts is not None:raise ValueError('adaptive sentence readout does not yet support joining')
            return self.respond_adaptive(predicted_self,question,min_score=min_score)
        from .self_model import respond_self
        return respond_self(self,predicted_self,question,min_score=min_score,threshold=threshold,joining_contexts=joining_contexts)
    def revise_self(self, predicted_self, new_observations):
        from .self_model import validate_self
        validate_self(self,predicted_self)
        extra=(new_observations,) if isinstance(new_observations,str) else tuple(new_observations)
        return self.build_self(predicted_self.observations+extra,state=predicted_self.state,
                               actions=predicted_self.actions,goal=predicted_self.goal)
    def train_joining(self, data):
        from .joining import JoiningLearner
        self.joining_model=JoiningLearner().fit(data)
        return self
    def join_statements(self, left, right, *, context='', min_score=.7):
        model=getattr(self,'joining_model',None)
        if model is None:raise RuntimeError('train joining first')
        return model.join(left,right,context=context,min_score=min_score)
    def compose_explanation(self, passage, question, *, contexts=None, min_score=.7):
        return self._compose_prediction(self.assess(passage,question),contexts=contexts,min_score=min_score)
    def _compose_prediction(self, base, *, contexts=None, min_score=.7):
        from ._models.generator import render
        if base['status']=='withheld_low_reliability':return {'text':'','status':'withheld_explanation','base':base,'joins':[]}
        # Recover the words actually predicted for each selected sentence.
        sentences=[render([w['word'] for w in step['word_trace'] if w['word']!='<end>']) for step in base['sentence_trace']]
        contexts=['']*max(0,len(sentences)-1) if contexts is None else list(contexts)
        if len(contexts)!=max(0,len(sentences)-1):raise ValueError('one context is needed per sentence boundary')
        chunks=[];joins=[];i=0
        while i<len(sentences):
            if i+1<len(sentences):
                joined=self.join_statements(sentences[i],sentences[i+1],context=contexts[i],min_score=min_score)
                joins.append(joined)
                if joined['status']=='joined':chunks.append(joined['text']);i+=2;continue
            chunks.append(sentences[i]);i+=1
        return {'text':' '.join(chunks),'status':'composed','joins':joins,'base':base,
                'reliability':{'estimated_correctness':None,'decision':'uncalibrated_composition'}}
    def train_frame_actions(self, data):
        from .transitions import FrameActionLearner
        self.frame_action_model=FrameActionLearner().fit(data)
        return self
    def predict_frame_outcome(self, state, frames, *, min_score=.9):
        mapper=getattr(self,'frame_action_model',None)
        if mapper is None:raise RuntimeError('train frame/action mapping first')
        if not 0<=min_score<=1:raise ValueError('min_score must be between zero and one')
        mapped=[mapper.predict(frame) for frame in frames]
        if any(item['confidence']<min_score for item in mapped):
            return {'status':'uncertain_frame_mapping','final_state':state,'mappings':mapped}
        result=self.predict_outcome(state,[item['action'] for item in mapped],min_score=min_score)
        result['mappings']=mapped
        return result
    def predict_explanation_outcome(self, passage, question, state, *, min_score=.9):
        explanation=self.assess(passage,question)
        if explanation['status']=='withheld_low_reliability':
            return {'status':'withheld_explanation','final_state':state,'explanation':explanation}
        frames=[{key:step[key] for key in ('subject','relationship','object')} for step in explanation['sentence_trace']]
        result=self.predict_frame_outcome(state,frames,min_score=min_score)
        result['explanation']=explanation
        return result
    def train_transitions(self, data):
        from .transitions import StateTransitionLearner
        self.transition_model = StateTransitionLearner().fit(data)
        import uuid
        self.transition_version=uuid.uuid4().hex
        return self
    def _transitions(self):
        model=getattr(self,'transition_model',None)
        if model is None:raise RuntimeError('train transitions first')
        return model
    def predict_state(self, state, action):
        return self._transitions().predict(state, action)
    def predict_outcome(self, state, actions, **kwargs):
        return self._transitions().rollout(state, actions, **kwargs)
    def plan(self, state, goal, actions, **kwargs):
        return self._transitions().plan(state, goal, actions, **kwargs)
    def learn_feedback(self, state, action, observed_state, *, learn=True):
        result=self._transitions().observe(state, action, observed_state, learn=learn)
        if learn:
            import uuid
            self.transition_version=uuid.uuid4().hex
        return result
    def train_reliability(self, fit_rows, calibration_rows):
        from .reliability import ReliabilityModel, diagnostics
        model = getattr(self, 'hierarchical_model', None)
        if model is None:raise RuntimeError('train hierarchy first')
        seen=set()
        def prepare(rows):
            out=[]
            for row in rows:
                key=(row['passage'].casefold(),row['question'].casefold())
                if key in seen:raise ValueError('fit and calibration examples must be distinct')
                seen.add(key)
                result=model.explain(row['passage'],row['question'])
                correct=(result['text'].strip().casefold()==row['expected'].strip().casefold())
                out.append((diagnostics(model,result),correct))
            return out
        fitted=prepare(list(fit_rows));calibrated=prepare(list(calibration_rows))
        self.reliability_model=ReliabilityModel().fit(fitted,calibrated)
        return self
    def assess(self, passage, question, *, threshold=.9, max_sentences=12):
        from .reliability import diagnostics
        if not 0<=threshold<=1:raise ValueError('threshold must be between 0 and 1')
        result=self.explain(passage,question,max_sentences=max_sentences)
        return self._assess_prediction(result,threshold=threshold)
    def _assess_prediction(self,result,*,threshold=.9):
        from .reliability import diagnostics
        features=diagnostics(self.hierarchical_model,result)
        reliability=getattr(self,'reliability_model',None)
        if reliability is None:
            result['reliability']={'features':features,'estimated_correctness':None,'decision':'uncalibrated'}
            return result
        scores=reliability.predict(features)
        accepted=bool(result['text']) and scores['estimated_correctness']>=threshold
        result['reliability']={**scores,'features':features,'threshold':threshold,
                               'decision':'answer' if accepted else 'request_more_information'}
        if not accepted:
            result['text']='';result['status']='withheld_low_reliability'
        return result
    def train_hierarchy(self, data):
        from .hierarchy import HierarchicalLearner
        self.hierarchical_model = HierarchicalLearner().fit(data)
        self.reliability_model = None  # Old calibration is invalid after retraining.
        import uuid
        self.representation_version=uuid.uuid4().hex
        return self
    def explain(self, passage, question, *, max_sentences=12):
        model = getattr(self, 'hierarchical_model', None)
        if model is None:raise RuntimeError('train hierarchy first')
        return model.explain(passage, question, max_sentences=max_sentences)
    def train_meaning(self, data, *, max_span=4, threshold=.5):
        from .meaning import MeaningLearner
        self.meaning_model = MeaningLearner(max_span, threshold).fit(data)
        return self
    def understand(self, text):
        model = getattr(self, 'meaning_model', None)
        if model is None:raise RuntimeError('train meaning first')
        return model.predict(text)
    def describe_concept(self, concept):
        model = getattr(self, 'meaning_model', None)
        if model is None:raise RuntimeError('train meaning first')
        return model.describe_concept(concept)
    def relate_modality(self, modality, value):
        prediction = self.predict_modality(modality, value)
        return {'prediction': prediction, 'concept_evidence': self.describe_concept(prediction['label'])}
    def train_modalities(self, data, *, codebook_size=32, seed=42):
        from .modalities import MultimodalLearner
        self.multimodal_model = MultimodalLearner(codebook_size, seed).fit(data)
        return self
    def encode_modality(self, modality, value):
        model = getattr(self, 'multimodal_model', None)
        if model is None:raise RuntimeError('train modalities first')
        return model.encoder.encode(modality, value)
    def predict_modality(self, modality, value):
        model = getattr(self, 'multimodal_model', None)
        if model is None:raise RuntimeError('train modalities first')
        return model.predict(modality, value)
    def train_generation(self,data,*,sentence_state=True,use_subjects=False):
        rows=self._rows(data,['input','answer'])
        for row in rows:self._text(row['input']);self._text(row['answer'])
        if use_subjects:
            if self.roles is None:raise RuntimeError('train roles before enabling use_subjects')
            model=SubjectGenerator(self.roles)
        else:model=SentenceGenerator() if sentence_state else Generator()
        self.generator=model.fit(rows)
        return self
    def train_roles(self,data):
        keys=['words','subject','actor','receiver','relation','subject_BIO'];rows=self._rows(data,keys)
        for row in rows:
            if not row['words'] or any(len(row[k])!=len(row['words']) for k in keys):raise ValueError('role labels must align with words')
            if any(x not in ['B','I','O'] for x in row['subject_BIO']):raise ValueError('subject_BIO labels must be B, I or O')
            if any(x not in [-1,0,1] for key in keys[1:-1] for x in row[key]):raise ValueError('role labels must be -1, 0 or 1')
        self.roles=RevisableRoles().fit(rows)
        # Existing generation weights were fitted to the old role head.
        # Keep that head stable until train_generation is called again.
        return self
    def train_sentences(self,data):
        self.sentence_model=Predictor().fit(self._rows(data,['sentence','actor','receiver']));return self
    def train_passages(self,data,*,seed=811):
        self.passage_model=PassageLearner().fit(self._rows(data,['passage','queries']),random.Random(seed));return self
    def generate(self,prompt,*,max_words=85):
        self._text(prompt)
        if not isinstance(max_words,int) or isinstance(max_words,bool) or max_words<1:raise ValueError('max_words must be a positive integer')
        if self.generator is None:raise RuntimeError('train or load a generation model first')
        return self.generator.generate(prompt,max_words)
    def ask(self,question,*,max_words=85,passage=None,predicted_self=None):
        if getattr(self,'response_control',None) is not None and predicted_self is not None:
            if passage is not None:raise ValueError('provide passage or predicted_self, not both')
            return self.ask_checked(question,predicted_self=predicted_self,max_words=max_words)
        return self._ask_raw(question,max_words=max_words,passage=passage,predicted_self=predicted_self)
    def _ask_raw(self,question,*,max_words=85,passage=None,predicted_self=None):
        if predicted_self is not None:
            if passage is not None:raise ValueError('provide passage or predicted_self, not both')
            if type(max_words)!=int or max_words<1:raise ValueError('max_words must be a positive integer')
            result=self.respond_self(predicted_self,question)
            output=tokens(result['text'])
            if len(output)>max_words:
                from ._models.generator import render
                result['text']=render(output[:max_words]);result['status']='word_limit';result['calibrated']=False
            return result
        if passage is None:return self.generate(question,max_words=max_words)
        if type(max_words)!=int or max_words<1:raise ValueError('max_words must be a positive integer')
        result=self.assess(passage,question)
        output=tokens(result['text'])
        if len(output)>max_words:
            from ._models.generator import render
            result['text']=render(output[:max_words]);result['status']='word_limit'
            result['reliability']={'estimated_correctness':None,'decision':'uncalibrated_truncation'}
        return result
    def train_continuation(self,data):
        from ._models.continuation import ContinuationGenerator
        rows=self._rows(data,['passage','question','answer'])
        for row in rows:
            for key in ['passage','question','answer']:self._text(row[key])
        self.continuation_model=ContinuationGenerator().fit(rows)
        return self
    def continue_passage(self,passage,question='What happens next?',*,max_words=55):
        self._text(passage);self._text(question)
        if not isinstance(max_words,int) or isinstance(max_words,bool) or max_words<1:raise ValueError('max_words must be a positive integer')
        model=getattr(self,'continuation_model',None)
        if model is None:raise RuntimeError('train or load a continuation model first')
        return model.continue_passage(passage,question,max_words)
    def predict_roles(self,text):
        self._text(text)
        if self.roles is None:raise RuntimeError('train or load a role model first')
        result=[];sentence=[]
        for word in tokens(text):
            sentence.append(word)
            if word in '.!?':result.extend(self.roles.predictions(sentence));sentence=[]
        if sentence:result.extend(self.roles.predictions(sentence))
        return result
    def role_session(self):
        if self.roles is None:raise RuntimeError('train or load a role model first')
        return RoleSession(self)
    def predict(self,sentence):
        self._text(sentence)
        if self.sentence_model is None:raise RuntimeError('train or load a sentence model first')
        return self.sentence_model.predict(sentence)
    def answer_passage(self,passage,action):
        self._text(passage);self._text(action)
        if self.passage_model is None:raise RuntimeError('train or load a passage model first')
        return self.passage_model.predict(passage,action.lower())
    def new_character_stream(self):
        model=getattr(self.sentence_model,'character',None)
        parameters=getattr(model,'character',None)
        if parameters is None:raise RuntimeError('no trained character encoder is loaded')
        return CharacterStream(parameters)
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        metadata={'format':'mikteeng-ai','schema_version':1,'package_version':'0.1.0'}
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('metadata.json',json.dumps(metadata))
            archive.writestr('state.pkl',pickle.dumps(self))
        return path
    @classmethod
    def load(cls,path):
        """Load trusted files only: internal pickle can execute Python code."""
        with zipfile.ZipFile(path) as archive:
            metadata=json.loads(archive.read('metadata.json'))
            if metadata.get('format')!='mikteeng-ai' or metadata.get('schema_version')!=1:raise ValueError('unsupported model format/version')
            model=pickle.loads(archive.read('state.pkl'))
        if not isinstance(model,cls):raise ValueError('checkpoint does not contain MikteengAI')
        return model
