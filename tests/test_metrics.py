import unittest
from medvqa import normalize_answer, exact_match, token_f1, prompt_for

class MetricTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_answer("ANSWER:  YES."), "yes")
        self.assertEqual(normalize_answer("ANSWER: T2\nextra"), "t2")
    def test_exact_match(self):
        self.assertEqual(exact_match("ANSWER: MRI", "mri"), 1.0)
        self.assertEqual(exact_match("CT", "MRI"), 0.0)
    def test_token_f1(self):
        self.assertAlmostEqual(token_f1("left kidney", "kidney"), 2 / 3)
    def test_prompt(self):
        self.assertIn("heart", prompt_for("Is this the heart?"))

if __name__ == '__main__':
    unittest.main()
