import unittest

import jax
import jax.numpy as jnp
import numpy as np

import core as C
import ideas as I


class Core(unittest.TestCase):
    def test_answer_roundtrip(self):
        for v in (0, 5, -5, 36, -72, 99):
            self.assertEqual(C.decode_answer(C.encode_answer(v)), v)

    def test_transforms_preserve_value_and_length(self):
        rng = np.random.default_rng(0)
        for _ in range(3000):
            x = C.random_terms(rng, int(rng.integers(1, 5)))
            for tf in (I.t_neia, I.t_npia, I.t_fill):
                y = tf(rng, x)
                self.assertEqual(C.value(y), C.value(x))
                self.assertLessEqual(len(y), 8)
                self.assertLessEqual(len(C.encode_input(y)), I.MAX_IN_LONG)
            self.assertLessEqual(C.n_terms(I.t_npia(rng, x)) - C.n_terms(x), 8 - len(x))

    def test_variants_run_and_padding_is_ignored(self):
        items = [[(1, 3), (-1, 7)], [(1, 9)]]
        for att in ("softmax", "ssmax", "sigmoid"):
            p, cfg = C.init(jax.random.PRNGKey(0), attention=att)
            a = C.batch(items, [C.value(x) for x in items], max_in=9)
            b = C.batch(items, [C.value(x) for x in items], max_in=17)
            la = C.per_example_nll(p, cfg, jnp.asarray(a))
            lb = C.per_example_nll(p, cfg, jnp.asarray(b))
            np.testing.assert_allclose(np.asarray(la), np.asarray(lb), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
