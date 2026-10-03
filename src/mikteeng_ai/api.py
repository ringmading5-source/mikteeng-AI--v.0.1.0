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
    def ask(self,question,*,max_words=85):return self.generate(question,max_words=max_words)
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
