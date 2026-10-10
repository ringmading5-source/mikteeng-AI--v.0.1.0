import tempfile,unittest
from pathlib import Path
from copy_conditioning import CopyConditioner,CopyConditionedLearner,CopyRule
class CopyTests(unittest.TestCase):
    def test_learns_unseen_unicode_span(self):
        m=CopyConditioner();m.fit([{'prompt':'tag[Ana]','response':'<Ana>'},{'prompt':'tag[Ben]','response':'<Ben>'}])
        self.assertEqual(m.generate('tag[Nyandëng]')['text'],'<Nyandëng>')
        self.assertEqual(m.generate('other[Nyandëng]')['route'],'abstain')
    def test_contradictory_demonstrations(self):
        m=CopyConditioner();m.fit([{'prompt':'tag[Ana]','response':'<Ana>'},{'prompt':'tag[Ben]','response':'<Ben>'},{'prompt':'tag[Ben]','response':'wrong'}])
        self.assertEqual(m.generate('tag[Joy]')['route'],'abstain')
    def test_conflicting_rules_abstain(self):
        m=CopyConditioner();m.rules=[CopyRule('tag[',']','<','>',2),CopyRule('tag[',']','(',')',2)]
        self.assertEqual(m.generate('tag[Joy]')['reason'],'conflicting_templates')
    def test_save_and_limits(self):
        m=CopyConditioner();m.fit([{'prompt':'tag[Ana]','response':'<Ana>'},{'prompt':'tag[Ben]','response':'<Ben>'}])
        self.assertEqual(m.generate('tag[a\nb]')['route'],'abstain')
        self.assertEqual(m.generate('tag['+'x'*129+']')['route'],'abstain')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.json';m.save(p);self.assertEqual(m.generate('tag[Joy]'),m.load(p).generate('tag[Joy]'))
    def test_fallback_explicit(self):
        class Neural:
            def generate(self,p):return {'text':'unverified'}
        m=CopyConditionedLearner(Neural())
        self.assertEqual(m.generate('unknown')['route'],'abstain')
        self.assertEqual(m.generate('unknown',allow_neural_fallback=True)['route'],'neural_unverified')
if __name__=='__main__':unittest.main()
