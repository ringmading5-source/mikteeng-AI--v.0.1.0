import json
from pathlib import Path
from mikteeng_ai import MikteengAI

def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

ai=MikteengAI()
ai.train_roles(read('experiments/annotated_role_training.jsonl'))
ai.train_generation(read('experiments/training.jsonl'),use_subjects=True)
ai.save('models/retrained.mkteeng')
print(ai.generate('Explain cells and velocity.')['text'])
