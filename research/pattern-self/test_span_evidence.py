import unittest
from unified_self import UnifiedSelf
from evidence_self import SpanEvidenceSelf
from train_span_evidence import examples
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows=examples(['Alice','Brian','Carol','David'],['book','cup','bag','key'])+examples(['Mary Jane','John Paul','Anna Maria Lee','James Arthur Cole'],['book','cup','bag','key'])
        cls.ai=UnifiedSelf().train_evidence([{'question':q,'observations':o,'answer':a} for q,o,a in rows]*30,method='spans')
    def test_unseen_two_word_active(self):
        self.assertEqual(self.ai.respond('Who holds the pencil?',observations='Lina Rose holds the pencil.')['text'],'Lina Rose')
    def test_unseen_three_word_passive(self):
        self.assertEqual(self.ai.respond('Who holds the pencil?',observations='The pencil is held by Pia Sera Moon.')['text'],'Pia Sera Moon')
    def test_missing_negative(self):
        self.assertEqual(self.ai.respond('Who holds the pencil?',observations='Lina Rose never holds the pencil.')['status'],'insufficient_evidence')
    def test_span_boundary(self):
        candidates=[s for s,_ in SpanEvidenceSelf().views('Who?','Mary Jane holds book.') if s]
        self.assertIn('Mary Jane',candidates)
        self.assertNotIn('Mary Jane holds book',candidates)
    def test_known_three_word_active_failure(self):
        self.assertNotEqual(self.ai.respond('Who holds the pencil?',observations='Pia Sera Moon holds the pencil.')['text'],'Pia Sera Moon')
if __name__=='__main__':unittest.main()
