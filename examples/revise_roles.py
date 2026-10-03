from mikteeng_ai import MikteengAI
model=MikteengAI.load('models/biology_physics.mkteeng')
session=model.role_session()
for word in ['the','cat','was','chased','by','the','dog','.']:
    predictions=session.push(word)
    print(word, predictions)
