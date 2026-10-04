import unittest
from pattern_self import PatternSelf, NumericPatternSelf
class Tests(unittest.TestCase):
    def test_empty(self):
        self.assertIsNone(PatternSelf().predict(['unknown'])['prediction'])
    def test_context_separation(self):
        s=PatternSelf().observe(['alice','holds','book']).observe(['brian','holds','cup'])
        self.assertEqual(s.predict(['alice','holds'])['prediction'],'book')
        self.assertEqual(s.predict(['brian','holds'])['prediction'],'cup')
    def test_conflict_retained(self):
        s=PatternSelf().observe(['switch','on']).observe(['switch','off'])
        self.assertEqual(s.predict(['switch'])['distribution'],{'on':.5,'off':.5})
    def test_feedback_adapts(self):
        s=PatternSelf().observe(['switch','on'])
        for _ in range(3):s.feedback(['switch'],'off')
        self.assertEqual(s.predict(['switch'])['prediction'],'off')
        self.assertEqual(s.patterns[('switch',)]['on'],1)
    def test_numeric_transfer(self):
        s=NumericPatternSelf()
        for step in [2,3,-2]:s.observe([10+i*step for i in range(6)])
        cases=[([101+i*step for i in range(4)],101+4*step) for step in [2,3,-2]]
        for context,expected in cases:self.assertEqual(s.predict(context),expected)
    def test_unseen_combination(self):
        s=PatternSelf().observe(['red','seed','water','grow'])
        self.assertEqual(s.generate(['new','seed','water'],1),['grow'])
    def test_wrong_extension(self):
        s=PatternSelf().observe(['seed','water','grow'])
        # Missing relationship: the suffix fallback can produce an unjustified answer.
        self.assertIsNotNone(s.predict(['seed','no','water'])['prediction'])
if __name__=='__main__':unittest.main()
