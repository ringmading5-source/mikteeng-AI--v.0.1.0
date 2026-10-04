import unittest
from pattern_self import PatternSelf,AdaptivePredictionSelf

def fixture():
    s=PatternSelf(2)
    for i in range(40):
        s.observe([f'rare{i}','signal','wrong'])
        for j in range(3):s.observe([f'common{i}_{j}','signal','right'])
    return s

class Tests(unittest.TestCase):
    def test_checked_prediction_selection(self):
        s=AdaptivePredictionSelf(fixture())
        for i in range(20):s.feedback([f'rare{i}','signal'],'right',update_patterns=False)
        for i in range(20,40):
            self.assertEqual(s.predict([f'rare{i}','signal'])['prediction'],'right')
            self.assertEqual(s.learner.predict([f'rare{i}','signal'])['prediction'],'wrong')
    def test_feedback_snapshot_before_update(self):
        s=AdaptivePredictionSelf(PatternSelf().observe(['a','wrong']))
        s.feedback(['a'],'right')
        self.assertEqual(s.history[-1]['prediction'],'wrong')
        self.assertEqual(s.history[-1]['actual'],'right')
    def test_adapts_changed_outcome(self):
        s=AdaptivePredictionSelf(fixture())
        for _ in range(30):s.feedback(['rare0','signal'],'right',update_patterns=False)
        self.assertEqual(s.predict(['rare0','signal'])['prediction'],'right')
        for _ in range(80):s.feedback(['rare0','signal'],'wrong',update_patterns=False)
        self.assertEqual(s.predict(['rare0','signal'])['prediction'],'wrong')
    def test_no_evidence(self):
        self.assertIsNone(AdaptivePredictionSelf().predict(['unknown'])['prediction'])
    def test_context_groups_independent(self):
        s=AdaptivePredictionSelf(fixture())
        for _ in range(10):s.feedback(['rare0','signal'],'right',update_patterns=False)
        self.assertNotIn(('other',1),s.reliability)
if __name__=='__main__':unittest.main()
