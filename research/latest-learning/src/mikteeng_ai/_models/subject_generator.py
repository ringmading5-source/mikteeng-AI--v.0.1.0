from .sentence_generator import SentenceGenerator,sentence_state
class SubjectGenerator(SentenceGenerator):
 def __init__(self,roles):self.roles=roles
 def state(self,words):
  f,completed,current=sentence_state(words)
  for back,sentence in enumerate(completed[-2:][::-1]):
   for name,value in self.roles.summary(sentence).items():f[f'role_previous:{back}:{name}']=value
  for name,value in self.roles.summary(current).items():f['role_current:'+name]=value
  return f
