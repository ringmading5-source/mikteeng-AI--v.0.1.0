import unittest
from unified_self import UnifiedSelf
from evidence_self import EvidenceSelf
from evaluate_evidence import dataset
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ai=UnifiedSelf().train_evidence([{'question':q,'observations':o,'answer':a} for q,o,a in dataset(['Alice','Brian','Carol','David'],['book','cup','bag','key'])])
    def test_unfamiliar_entities_and_distractors(self):
        for q,o,a in dataset(['Nia','Omar','Pia','Ravi'],['pencil','basket','coin','bottle']):
            self.assertEqual(self.ai.respond(q,observations=o)['text'],'' if a is None else a)
    def test_changed_holder(self):
        for name in ['Nia','Omar']:
            self.assertEqual(self.ai.respond('Who holds the pencil?',observations=f'{name} holds the pencil.')['text'],name)
    def test_empty_evidence(self):
        self.assertEqual(self.ai.respond('Who holds the pencil?',observations='',task='evidence_answer')['status'],'insufficient_evidence')
    def test_invalid_label(self):
        with self.assertRaises(ValueError):EvidenceSelf().learn('Who?','Alice holds book.','Brian')
    def test_numeric_unchanged(self):
        self.assertEqual(self.ai.respond(observations=[100,105,110,115],task='numeric_next',modality='numeric')['prediction'],120)
if __name__=='__main__':unittest.main()
