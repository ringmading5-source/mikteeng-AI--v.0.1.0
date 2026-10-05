"""Create a tiny reproducible vector dataset, then train the same RSPM learner."""
import json
from mikteeng_rspm import MikteengRSPM
engine=MikteengRSPM(dim=32)
with open('training.jsonl','w') as f:
    for _ in range(30):
        f.write(json.dumps({'history':engine.B[0].tolist(),'trigger':engine.B[1].tolist(),'target':engine.B[2].tolist()})+'\n')
print('python -m mikteeng_rspm train training.jsonl --dim 32 --output trained.json')
