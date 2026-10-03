class ExtendedPassageSystem:
 def __init__(self,previous,passage):self.previous=previous;self.passage=passage
 def predict(self,sentence):return self.previous.predict(sentence)
 def answer(self,passage,action):return self.passage.predict(passage,action)
