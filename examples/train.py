from mikteeng_ai import MikteengAI
# A deliberately tiny API example, not a useful trained knowledge model.
data=[{'input':'What is speed?','answer':'Speed is distance divided by time.'},
      {'input':'Explain speed.','answer':'Speed is distance divided by time.'},
      {'input':'What is a cell?','answer':'A cell is a basic unit of life.'},
      {'input':'Explain a cell.','answer':'A cell is a basic unit of life.'}]
model=MikteengAI().train(data,task='generation')
model.save('models/my_model.mkteeng')
print(model.generate('Explain speed.')['text'])
