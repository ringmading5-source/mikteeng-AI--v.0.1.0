import copy, json, tempfile, unittest, wave
from pathlib import Path
import numpy as np
from tbin import TBIN
from tbin_v25 import TBIN25
from tbin_v26 import load_audio


def toy(seed=18):
    rng=np.random.default_rng(seed)
    colors=np.array([[220,30,30],[30,220,30],[30,30,220],[220,220,30],[220,30,220],[30,220,220]])
    freqs=[220,320,440,570,720,900]
    def image(i,v):return np.clip(colors[i]+rng.normal(0,v,(32,32,3)),0,255).astype('uint8')
    def audio(i,shift):return (.5*np.sin(2*np.pi*(freqs[i]+shift)*np.arange(2048)/8000+.1*shift)).astype('float32')
    pairs=[{'image':image(i,3),'audio':audio(i,float(j)),'group':str(i)} for j in range(3) for i in range(6)]
    held=[{'image':image(i,7),'audio':audio(i,3.5)} for i in range(6)]
    return pairs,held

def score(model,held):
    return sum(model.cross_retrieve(src,row[src],dst,[h[dst] for h in held])['index']==i
               for i,row in enumerate(held) for src,dst in [('image','audio'),('audio','image')])

class Repairs(unittest.TestCase):
    def test_consolidate_then_learn(self):
        m=TBIN();m.observe_text('cow likes grass');m.core.reasoner.consolidate()
        m.observe_text('goat likes leaves')
        self.assertEqual(m.core.reasoner.levels['words']['likes']['leaves'],1)

    def test_long_relation_and_neural_training(self):
        m=TBIN();m.teach_relation('cow likes grass','what does cow like','cow','grass')
        m.observe_text('the sheep likes green grass')
        self.assertEqual(m.answer_relation('what does sheep like','sheep'),'green grass')
        before=m.core.reasoner.W.copy()
        self.assertTrue(m.teach_relation('the goat eats fresh leaves','what does goat eat','goat','fresh leaves'))
        self.assertTrue(m.core.reasoner.neural_trained)
        self.assertFalse(np.array_equal(before,m.core.reasoner.W))
        self.assertEqual(m.answer_relation('what does goat eat','goat'),'fresh leaves')
        m.observe_fact('young goat','eats','green leaves')
        self.assertEqual(m.answer_relation('what does young goat eat','young goat'),'green leaves')

    def test_audio_time_and_aliasing(self):
        with tempfile.TemporaryDirectory() as td:
            def wav(name,sr,seconds,freq):
                path=Path(td)/name;t=np.arange(round(sr*seconds))/sr
                with wave.open(str(path),'wb') as f:
                    f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr)
                    f.writeframes((np.sin(2*np.pi*freq*t)*16000).astype('<i2').tobytes())
                return load_audio(path)
            for sr,duration in [(8000,.5),(16000,1),(44100,2)]:
                a=wav(f'{sr}.wav',sr,duration,440)
                self.assertEqual(len(a),int(8000*duration))
                dominant=np.fft.rfftfreq(len(a),1/8000)[np.argmax(abs(np.fft.rfft(a)))]
                self.assertAlmostEqual(dominant,440,delta=2)
            low=wav('low.wav',16000,1,1000);high=wav('high.wav',16000,1,6000)
            self.assertLess(np.linalg.norm(high),.05*np.linalg.norm(low))

    def test_stream_bound_and_tail(self):
        for count in (3,4,5,6,7,8,9,10,13,100):
            calls=[]
            def rows():
                calls.append(1)
                for i in range(count):yield {'text':str(i),'audio':np.sin(np.arange(256)*(i%3+1)).astype('float32')}
            m=TBIN25();m.train_unlabeled_pairs(rows,epochs=1,batch_size=6)
            self.assertLessEqual(m.training_stats['max_batch'],6)
            self.assertLessEqual(m.training_stats['max_score_elements'],36)
            self.assertEqual(m.training_stats['pairs'],count)
            self.assertGreaterEqual(len(calls),2)

    def test_manifest_is_streamed(self):
        with tempfile.TemporaryDirectory() as td:
            from PIL import Image
            root=Path(td);Image.fromarray(np.full((8,8,3),100,dtype='uint8')).save(root/'x.png')
            manifest=root/'pairs.jsonl'
            manifest.write_text('\n'.join(json.dumps({'text':str(i),'image':'x.png'}) for i in range(9)))
            m=TBIN();m.load_rows=lambda *_:self.fail('train used full-dataset loader')
            m.train(manifest,epochs=1,batch_size=6)
            self.assertEqual(m.core.reasoner.passages,[])

    def test_validation(self):
        m=TBIN25()
        with self.assertRaises(ValueError):m.train_unlabeled_pairs([],batch_size=2)
        with self.assertRaises(ValueError):m.cross_retrieve('text','hi','image',[])
        with self.assertRaises(ValueError):m.train_unlabeled_pairs([{'text':'x','audio':np.ones(64),'group':'same'}]*3)
        with self.assertRaises(ValueError):m.train_unlabeled_pairs([{'text':'x','audio':np.ones(64)}, {'text':'y','image':np.ones((4,4,3))}]*2)

    def test_accuracy_and_codebook_persistence(self):
        pairs,held=toy();m=TBIN25();before=score(m,held)
        m.train_unlabeled_pairs(pairs,epochs=150,lr=.025)
        after=score(m,held)
        self.assertGreater(after,before)
        self.assertGreaterEqual(after,8)
        feature=m.shared.image_features(held[0]['image'])
        self.assertGreater(np.linalg.norm(feature[48:]),0)
        frozen=copy.deepcopy(m.shared.visual.centers)
        m.observe_image(held[0]['image'])
        np.testing.assert_array_equal(frozen,m.shared.visual.centers)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'weights.npz';m.save_model(path);n=TBIN25();n.load_model(path)
            self.assertEqual(score(n,held),after)
            for src in ('image','audio'):
                np.testing.assert_allclose(m.shared.encode(src,held[0][src]),n.shared.encode(src,held[0][src]))

if __name__=='__main__':unittest.main()
