"""Explicit interface integration; no claim of joint cross-modality learning."""
import re,pickle
from pathlib import Path
from pattern_self import AdaptivePredictionSelf,PatternSelf,RecursiveNumericSelf

class Tokenizer:
    """Lossless text/code tokenizer; token strings are not semantic labels."""
    def __init__(self,level='word'):
        if level not in ('character','word'):raise ValueError('Unsupported token level')
        self.level=level
    def encode(self,text):
        if not isinstance(text,str):raise ValueError('Expected text')
        return list(text) if self.level=='character' else re.findall(r'\s+|\w+|[^\w\s]',text)
    def decode(self,tokens):return ''.join(tokens)

class UnifiedSelf:
    # Tuple markers cannot collide with tokenizer string tokens.
    QUESTION=('question',);ANSWER=('answer',);END=('end',)
    def __init__(self,max_context=16):
        self.tokenizer=Tokenizer()
        self.streams={name:AdaptivePredictionSelf(PatternSelf(max_context)) for name in ('text','code')}
        self.numeric=RecursiveNumericSelf()
        self.response=AdaptivePredictionSelf(PatternSelf(max_context))
        self.interactions=[]

    def observe(self,observation,modality='text'):
        if modality=='numeric':
            # Numeric observations are analyzed locally by respond, not silently
            # converted into checked prediction labels.
            values=list(observation);self.numeric.candidates(values)
            self.interactions.append({'kind':'observation','modality':modality,'values':values})
        elif modality in self.streams:
            self.streams[modality].observe(self.tokenizer.encode(observation))
        else:raise ValueError('Supported modalities: text, code, numeric')
        return self

    def _prefix(self,question,observations=''):
        if not isinstance(question,str) or not question.strip():raise ValueError('Question required')
        return self.tokenizer.encode(observations)+[self.QUESTION]+self.tokenizer.encode(question)+[self.ANSWER]

    def train_responses(self,rows):
        rows=list(rows)
        if not rows:raise ValueError('Empty training data')
        for row in rows:
            answer=row['answer']
            if not isinstance(answer,str) or not answer:raise ValueError('Answer required')
            sequence=self._prefix(row['question'],row.get('observations',''))+self.tokenizer.encode(answer)+[self.END]
            self.response.observe(sequence)
        return self

    def respond(self,question=None,observations='',task='answer',modality='text',max_tokens=64):
        if not isinstance(max_tokens,int) or isinstance(max_tokens,bool) or max_tokens<1:raise ValueError('Positive token limit required')
        if task=='numeric_next':
            if modality!='numeric':raise ValueError('numeric_next requires numeric modality')
            result=self.numeric.predict(observations)
            return {'status':'untrained' if result['prediction'] is None else 'experimental',
                    'prediction':result['prediction'],'trace':result}
        if modality not in self.streams:raise ValueError('Text/code input required')
        if task=='continue':
            history=self.tokenizer.encode(observations);learner=self.streams[modality]
        elif task=='answer':history=self._prefix(question,observations);learner=self.response
        else:raise ValueError('Tasks: answer, continue, numeric_next')
        output=[];trace=[];status='limit_reached'
        for _ in range(max_tokens):
            result=learner.predict(history);token=result['prediction'];trace.append(result)
            if token is None:status='untrained';break
            if token==self.END:status='completed';break
            if not isinstance(token,str):status='boundary_reached';break
            output.append(token);history.append(token)
        return {'status':status,'text':self.tokenizer.decode(output),'trace':trace}

    def feedback(self,question=None,observations='',actual=None,task='answer',modality='text'):
        if task=='numeric_next':
            if modality!='numeric':raise ValueError('numeric_next requires numeric modality')
            self.numeric.feedback(observations,actual);return self
        if modality not in self.streams:raise ValueError('Text/code input required')
        if task=='answer':prefix=self._prefix(question,observations);learner=self.response;end=[self.END]
        elif task=='continue':prefix=self.tokenizer.encode(observations);learner=self.streams[modality];end=[]
        else:raise ValueError('Unknown task')
        if not isinstance(actual,str) or not actual:raise ValueError('Checked text required')
        for token in self.tokenizer.encode(actual)+end:
            learner.feedback(prefix,token);prefix.append(token)
        return self

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(pickle.dumps(self));return path
    @classmethod
    def load(cls,path):
        """Trusted local checkpoints ONLY: pickle can execute code."""
        model=pickle.loads(Path(path).read_bytes())
        if not isinstance(model,cls):raise ValueError('Invalid checkpoint')
        return model
