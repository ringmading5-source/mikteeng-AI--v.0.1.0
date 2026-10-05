from pathlib import Path
import json
from threadpoolctl import threadpool_limits
from mikteeng_ai import MikteengAI
from mikteeng_ai.character_prediction import CharacterPredictionModel
root=Path(__file__).resolve().parent
def read(name):
    return [json.loads(line) for line in (root/'data'/name).read_text().splitlines() if line.strip()]
with threadpool_limits(limits=1):
    ai=MikteengAI.load_trusted(root/'data/composition_seed.mkteeng')
    ai.train_generation(read('fundamentals_train.jsonl'))
    ai.train_continuation(read('continuation_train.jsonl'))
    acoustic=MikteengAI.load_trusted(root/'data/acoustic_seed.mkteeng')
    ai.speech_prediction_model=acoustic.speech_prediction_model
    words=json.loads((root/'data/word_formation_train.json').read_text())
    ai.word_formation_model=CharacterPredictionModel(context=3).fit(['^'+word+'$' for word in words])
    ai.save(root/'models/sentence_chatbot.mkteeng')
print('Chatbot checkpoint ready',flush=True)
