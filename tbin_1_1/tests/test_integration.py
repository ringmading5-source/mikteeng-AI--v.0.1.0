import tempfile, unittest, wave
from pathlib import Path
import json
import numpy as np
from PIL import Image
from tbin import TBIN

class Integration(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);rng=np.random.default_rng(1);manifest=[]
            for i in range(4):
                im=np.full((24,24,3),[20+i*45,40+i*20,100+i*15],dtype=np.uint8)
                im=np.clip(im.astype('int16')+rng.integers(-2,3,im.shape),0,255).astype('uint8')
                Image.fromarray(im).save(root/f'{i}.png')
                sr=8000;t=np.arange(4096)/sr
                with wave.open(str(root/f'{i}.wav'),'wb') as w:
                    w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr)
                    w.writeframes((np.sin(2*np.pi*(220+i*100)*t)*16000).astype('<i2').tobytes())
                manifest.append({'image':f'{i}.png','audio':f'{i}.wav','text':f'item {i}'})
            path=root/'pairs.jsonl';path.write_text('\n'.join(json.dumps(r) for r in manifest))
            m=TBIN();result=m.train(path,epochs=4)
            self.assertEqual(result['pairs'],4)
            r=m.retrieve('image',root/'0.png','audio',[root/f'{i}.wav' for i in range(4)])
            self.assertEqual(len(r['scores']),4)
            m.teach_relation('cow likes grass','what does cow like','cow','grass')
            self.assertEqual(m.answer_relation('what does cow like','cow'),'grass')
            ck=root/'model.pkl';m.save(ck);n=TBIN.load(ck)
            self.assertEqual(n.answer_relation('what does cow like','cow'),'grass')
            self.assertEqual(n.retrieve('image',root/'0.png','audio',[root/f'{i}.wav' for i in range(4)])['index'],r['index'])
            a=n.observe_audio(root/'0.wav');self.assertGreater(a['frames'],0)
            v=n.observe_image(root/'0.png');self.assertGreater(v['patches'],0)
    def test_invalid_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'pairs.jsonl';p.write_text('{"text":"only one modality"}\n')
            with self.assertRaises(ValueError):TBIN().train(p)
if __name__=='__main__':unittest.main()
