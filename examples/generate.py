import sys
from mikteeng_ai import MikteengAI
model=MikteengAI.load(sys.argv[1] if len(sys.argv)>1 else 'models/biology_physics.mkteeng')
print(model.generate('Explain cells and velocity.')['text'])
