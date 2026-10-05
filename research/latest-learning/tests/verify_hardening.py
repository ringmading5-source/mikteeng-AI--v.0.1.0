import json,tempfile,wave
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from mikteeng_ai import MikteengAI,PatternNodeSpace,MultimodalByteTokenizer,MultimodalPacket
root=Path(__file__).resolve().parents[1]
try:MikteengAI.load(root/'models/passage_patterns_16.mkteeng')
except ValueError:pass
else:raise AssertionError('untrusted load allowed')
ai=MikteengAI.load_trusted(root/'models/passage_patterns_16.mkteeng')
tests=json.loads((root/'data/passage_patterns_test.json').read_text());correct=0
for p in tests:
 for length in [3,4]:
  actual=ai.continue_sentence_patterns(ai.build_sentence_prediction_self(p[:length]),steps=3)['sentences'];correct+=sum(a==b for a,b in zip(actual,p[length:length+3]))
assert correct==156
s=ai.build_sentence_prediction_self(tests[0][:3])
for steps in [True,0,65]:
 try:ai.continue_sentence_patterns(s,steps=steps)
 except ValueError:pass
 else:raise AssertionError('step cap failed')
space=PatternNodeSpace(128,4096);assert space.anchors.nbytes==0
with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lambda _:space.observe('same',np.ones(128)),range(100)))
assert len(space.ids)==1 and space.counts[0]==100
assert space.anchors.nbytes==16*128*8
for metadata in [{'dtype':'O','shape':[1]},{'dtype':'u1','shape':[999]}]:
 try:MultimodalPacket(b'0','numeric',json.dumps(metadata))
 except ValueError:pass
 else:raise AssertionError('metadata accepted')
t=MultimodalByteTokenizer()
with tempfile.TemporaryDirectory() as d:
 path=Path(d)/'sample.wav';samples=np.arange(100,dtype=np.int16)
 with wave.open(str(path),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(samples.tobytes())
 chunks=list(t.iter_wav(path,frames_per_chunk=13))
 restored=np.concatenate([p.decode()[0] for p in chunks]);np.testing.assert_array_equal(restored,samples)
 ai.save(Path(d)/'safe-test.mkteeng');loaded=MikteengAI.load_trusted(Path(d)/'safe-test.mkteeng')
 assert loaded.continue_sentence_patterns(loaded.build_sentence_prediction_self(tests[0][:3]),steps=3)==ai.continue_sentence_patterns(s,steps=3)
print('PASS: original 156/192 retained; explicit checkpoint trust; step caps; concurrent node reuse; lazy memory; metadata rejection; PCM streaming; save/reload')
