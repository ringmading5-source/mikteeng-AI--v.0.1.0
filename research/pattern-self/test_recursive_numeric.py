import unittest
from pattern_self import RecursiveNumericSelf
class Tests(unittest.TestCase):
    def test_untrained_change(self):
        self.assertEqual(RecursiveNumericSelf().predict([100,105,110,115])['prediction'],120)
    def test_second_level(self):
        r=RecursiveNumericSelf().predict([1,4,9,16,25])
        self.assertEqual(r['prediction'],36);self.assertEqual(r['depth'],2)
    def test_third_level(self):
        r=RecursiveNumericSelf().predict([i**3 for i in range(1,8)])
        self.assertEqual(r['prediction'],512);self.assertEqual(r['depth'],3)
    def test_checked_outcomes(self):
        s=RecursiveNumericSelf();s.feedback([1,4,9,16,25],36)
        self.assertEqual(s.reliability[2],[1.,0.]);self.assertEqual(s.history[0]['proposal']['prediction'],36)
    def test_short_or_invalid(self):
        self.assertIsNone(RecursiveNumericSelf().predict([1,2])['prediction'])
        with self.assertRaises(ValueError):RecursiveNumericSelf().predict(['one','two','three'])
    def test_unlearned_geometric_failure(self):
        self.assertNotEqual(RecursiveNumericSelf().predict([2,4,8,16,32])['prediction'],64)
if __name__=='__main__':unittest.main()
