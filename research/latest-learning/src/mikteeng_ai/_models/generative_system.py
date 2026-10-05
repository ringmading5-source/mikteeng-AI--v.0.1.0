class GenerativeSystem:
 def __init__(self,previous,generator):self.previous=previous;self.generator=generator
 def ask(self,question,max_words=70):return self.generator.generate(question,max_words)
 def predict(self,sentence):return self.previous.predict(sentence)
 def answer(self,passage,action):return self.previous.answer(passage,action)
