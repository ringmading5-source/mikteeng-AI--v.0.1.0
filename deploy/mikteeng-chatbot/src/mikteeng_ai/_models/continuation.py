"""Learn continuation tokens from a question and preceding sentence states."""
from scipy.sparse import hstack
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from .sentence_generator import SentenceGenerator
from .generator import tokens,render

class ContinuationGenerator(SentenceGenerator):
 def fit(self,rows):
  self.query=FeatureUnion([('words',TfidfVectorizer(ngram_range=(1,2))),('characters',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5)))])
  Q=self.query.fit_transform([r['question'] for r in rows]);features=[];labels=[];indices=[]
  for i,row in enumerate(rows):
   history=tokens(row['passage'])
   for word in tokens(row['answer'])+['<end>']:
    features.append(self.state(history));labels.append(word);indices.append(i);history.append(word)
  self.vectorizer=DictVectorizer();H=self.vectorizer.fit_transform(features)
  self.model=LogisticRegression(C=30,max_iter=600).fit(hstack([Q[indices]*3,H],format='csr'),labels)
  self.training_steps=len(labels);return self
 def continue_passage(self,passage,question='What happens next?',max_words=55):
  q=self.query.transform([question]);history=tokens(passage);output=[];trace=[]
  for step in range(max_words):
   p=self.distribution(q,history);index=int(p.argmax());word=str(self.model.classes_[index])
   trace.append(dict(step=step,word=word,probability=float(p[index])))
   if word=='<end>':return dict(text=render(output),status='completed',trace=trace)
   output.append(word);history.append(word)
  return dict(text=render(output),status='word_limit',trace=trace)
