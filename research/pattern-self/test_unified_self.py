import unittest,tempfile
from pathlib import Path
from unified_self import UnifiedSelf,Tokenizer
class Tests(unittest.TestCase):
    def test_tokenizer_lossless(self):
        text='def foo(x):\n    return x + 1\n'
        for level in ['word','character']:
            t=Tokenizer(level);self.assertEqual(t.decode(t.encode(text)),text)
    def test_connected_responses_and_numeric(self):
        s=UnifiedSelf().train_responses([{'question':'Who holds the book?','answer':'Alice.'},{'question':'Who holds the cup?','answer':'Brian.'}])
        self.assertEqual(s.respond('Who holds the book?')['text'],'Alice.')
        self.assertEqual(s.respond(observations=[100,105,110,115],task='numeric_next',modality='numeric')['prediction'],120)
        s.feedback(observations=[100,105,110,115],actual=120,task='numeric_next',modality='numeric')
        self.assertEqual(s.respond('Who holds the cup?')['text'],'Brian.')
    def test_feedback_corrects_response(self):
        s=UnifiedSelf().train_responses([{'question':'Who holds the book?','answer':'Alice.'}])
        for _ in range(5):s.feedback('Who holds the book?',actual='Brian.')
        self.assertEqual(s.respond('Who holds the book?')['text'],'Brian.')
    def test_separate_streams(self):
        s=UnifiedSelf().observe('hello world').observe('return 5',modality='code')
        self.assertEqual(s.respond(observations='hello ',task='continue',max_tokens=1)['text'],'world')
        self.assertEqual(s.respond(observations='return ',task='continue',modality='code',max_tokens=1)['text'],'5')
    def test_save_load(self):
        s=UnifiedSelf().train_responses([{'question':'Who?','answer':'Alice.'}])
        with tempfile.TemporaryDirectory() as d:
            p=s.save(Path(d)/'self.pkl');loaded=UnifiedSelf.load(p)
            self.assertEqual(s.respond('Who?'),loaded.respond('Who?'))
    def test_empty_and_unsupported(self):
        s=UnifiedSelf()
        self.assertEqual(s.respond('Who?')['status'],'untrained')
        with self.assertRaises(ValueError):s.observe(b'audio',modality='audio')
        with self.assertRaises(ValueError):s.respond('Who?',max_tokens=0)
if __name__=='__main__':unittest.main()
