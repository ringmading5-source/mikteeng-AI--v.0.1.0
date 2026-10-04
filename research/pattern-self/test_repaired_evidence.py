import unittest
from unified_self import UnifiedSelf
from evaluate_evidence import dataset
from repair_evidence import additions
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows=dataset(['Alice','Brian','Carol','David'],['book','cup','bag','key'])+additions(['Alice','Brian','Carol','David'],['book','cup','bag','key'])
        cls.ai=UnifiedSelf().train_evidence([{'question':q,'observations':o,'answer':a} for q,o,a in rows]*30,method='ranking')
    def test_passive_negative_and_retention(self):
        rows=dataset(['Elin','Farah','Goran'],['rope','stone','hat'])+additions(['Elin','Farah','Goran'],['rope','stone','hat'])
        for q,o,a in rows:self.assertEqual(self.ai.respond(q,observations=o)['text'],'' if a is None else a)
    def test_multiword_limit(self):
        self.assertNotEqual(self.ai.respond('Who holds the pencil?',observations='Mary Jane holds the pencil.')['text'],'Mary Jane')
    def test_numeric_retained(self):
        self.assertEqual(self.ai.respond(observations=[100,105,110,115],task='numeric_next',modality='numeric')['prediction'],120)
if __name__=='__main__':unittest.main()
