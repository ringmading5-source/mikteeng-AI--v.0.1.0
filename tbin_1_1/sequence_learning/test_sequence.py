import tempfile,unittest
from pathlib import Path
import numpy as np
from sequence import SequenceLearner
class Tests(unittest.TestCase):
    def test_gradient(self):
        m=SequenceLearner();_,g=m.loss_grad('x','a')
        for key,idx in [('E',(97,0)),('X',(0,0)),('H',(0,0)),('O',(0,97)),('b',(0,)),('c',(97,))]:
            original=float(m.p[key][idx]);eps=.005
            m.p[key][idx]=original+eps;plus,_=m.loss_grad('x','a')
            m.p[key][idx]=original-eps;minus,_=m.loss_grad('x','a');m.p[key][idx]=original
            self.assertAlmostEqual(float(g[key][idx]),(plus-minus)/(2*eps),delta=.0003)
    def test_roundtrip(self):
        m=SequenceLearner()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'weights.npz';m.save(p);n=m.load(p)
            self.assertEqual(m.generate('hello'),n.generate('hello'))
    def test_limits(self):
        m=SequenceLearner()
        with self.assertRaises(ValueError):m.loss_grad('x'*600,'a')
        with self.assertRaises(ValueError):m.train([])
        self.assertLessEqual(len(m.generate('x',max_bytes=3)['text']),3)
if __name__=='__main__':unittest.main()
