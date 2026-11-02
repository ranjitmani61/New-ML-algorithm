import unittest

import numpy as np

import tasks as T


class Relations(unittest.TestCase):
    def test_encoding_roundtrip(self):
        for v in (0, 7, -7, 23, -81, 99):
            self.assertEqual(T.decode_answer(T.encode_answer(v)), v)
        self.assertIsNone(T.decode_answer([T.PLUS, 1]))

    def test_relations_hold(self):
        rng = np.random.default_rng(0)
        for _ in range(2000):
            terms = T.random_terms(rng, int(rng.integers(1, 9)))
            v = T.value(terms)
            for rel in (T.permute, T.shift_term, T.flip_signs):
                new, back = rel(rng, terms)
                self.assertEqual(len(new), len(terms))
                self.assertTrue(all(0 <= t <= T.TERM_MAX for _, t in new))
                self.assertEqual(back(T.value(new)), v, rel.__name__)

    def test_placebo_is_wrong(self):
        rng = np.random.default_rng(1)
        wrong = 0
        for _ in range(500):
            terms = T.random_terms(rng, 6)
            new, back = T.shift_term_wrong(rng, terms)
            wrong += back(T.value(new)) != T.value(terms)
        self.assertEqual(wrong, 500)

    def test_batch_layout(self):
        b = T.batch([[(1, 2), (-1, 7)]], [-5])
        self.assertEqual(b.shape, (1, T.SEQ))
        self.assertEqual(list(b[0, T.MAX_IN - 5:T.MAX_IN]), [T.PLUS, 3, T.MINUS, 8, T.EQ])
        self.assertTrue((b[0, :T.MAX_IN - 5] == T.PAD).all())
        self.assertEqual(T.decode_answer(b[0, T.MAX_IN:]), -5)
        self.assertEqual(int(b[0, T.MAX_IN - 1]), T.EQ)


if __name__ == "__main__":
    unittest.main()
