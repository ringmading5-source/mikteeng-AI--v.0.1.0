from pathlib import Path
import json
from threadpoolctl import threadpool_limits
from mikteeng_ai import MikteengAI
root=Path(__file__).resolve().parent
def read(name):
    return [json.loads(line) for line in (root/'data'/name).read_text().splitlines() if line.strip()]
with threadpool_limits(limits=1):
    ai=MikteengAI()
    ai.train_generation(read('fundamentals_train.jsonl'))
    ai.train_continuation(read('continuation_train.jsonl'))
    ai.save(root/'models/sentence_chatbot.mkteeng')
print('Chatbot checkpoint ready',flush=True)
