"""Tests for answer handling, the exact step verifier and the offline lab."""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchmarks as bm  # noqa: E402
import lab  # noqa: E402


class Answers(unittest.TestCase):
    def test_boxed_extraction(self):
        self.assertEqual(bm.last_boxed("so \\boxed{\\frac{1}{2}} and \\boxed{3}"), "3")
        self.assertEqual(bm.last_boxed("x = \\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}"), "\\left( 3, \\frac{\\pi}{2} \\right)")
        self.assertIsNone(bm.last_boxed("no answer here"))
        self.assertEqual(bm.gsm8k_gold("blah\n#### 1,234"), "1234")

    def test_equivalence(self):
        for a, b in (("\\frac{1}{2}", "0.5"), ("1/2", "\\dfrac12"), ("10", "10.0"), ("\\$18", "18"), ("x=5", "5"),
                     ("1,000", "1000"), ("90^\\circ", "90"), ("\\text{(C)}", "(C)"), ("50\\%", "50")):
            self.assertTrue(bm.is_equiv(a, b), (a, b))
        for a, b in (("5", "6"), ("\\frac{1}{3}", "0.33"), (None, "1")):
            self.assertFalse(bm.is_equiv(a, b), (a, b))


class StepVerifier(unittest.TestCase):
    def check(self, text):
        return bm.check_steps(text)

    def test_true_false_and_rounding(self):
        self.assertEqual(self.check("16 - 3 - 4 = 9 eggs, 9 * 2 = $18")["false"], 0)
        self.assertEqual(self.check("so 7 * 8 = 54")["false"], 1)
        self.assertEqual(self.check("10/3 = 3.33")["false"], 0)  # correct rounding
        self.assertEqual(self.check("10/3 = 3.4")["false"], 1)
        self.assertEqual(self.check("\\frac{3}{4} \\times 8 = 6")["false"], 0)
        self.assertEqual(self.check("2^{10} = 1024")["false"], 0)

    def test_skips_what_it_cannot_judge(self):
        for text in ("x = 5", "Let n = 3 apples", "x + 2 = 7 so x = 5", "3 / 4 * 2 = 1 1/2 feet",
                     "2 * 225 + 2 * 125 = 450 + 250 = 700"):
            self.assertEqual(self.check(text)["false"], 0, text)

    def test_last_claim_and_repair_source(self):
        c = self.check("First 5 + 3 = 8. Then 8 * 3 = 25. The answer is \\boxed{25}")
        self.assertEqual(c["false"], 1)
        self.assertEqual(c["last"], (False, "25", "24"))


def trace(answer, tokens=100, conf=10.0, false=0, repaired=None, finished=True):
    return lab.Trace(answer, tokens, finished, conf, conf, conf, -0.1, 3, false, repaired)


class Lab(unittest.TestCase):
    def test_vote_rules(self):
        ts = [trace("1"), trace("2"), trace("2"), trace("1")]
        self.assertEqual(lab.vote(ts)[0], "1")  # tie -> earliest
        w = lab.conf_weights([trace("1", conf=1), trace("2", conf=9), trace("2", conf=8)], eta=0.5)
        self.assertEqual(w[0], 0.0)
        bad = [trace("25", false=1, repaired="24"), trace("25", false=1, repaired="24"), trace("30")]
        self.assertEqual(lab.vote(bad, lab.vsc_weights(bad, rho=0.25), use_repair=True)[0], "24")
        self.assertEqual(lab.vote(bad)[0], "25")

    def synthetic(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        qs = []
        for i in range(n):
            p = rng.uniform(0.3, 0.95)
            q = lab.Question(f"q{i}", "dev", "A")
            for _ in range(16):
                ok = rng.random() < p
                q.traces.append(trace("A" if ok else rng.choice(["B", "C"]), tokens=int(rng.integers(50, 150)),
                                      false=0 if ok or rng.random() < 0.5 else 1))
            qs.append(q)
        return qs

    def test_simulation_sanity(self):
        qs = self.synthetic()
        one = lab.simulate(qs, "sc", {"k": 1}, repeats=20)
        many = lab.simulate(qs, "sc", {"k": 9}, repeats=20)
        self.assertGreater(many["accuracy"], one["accuracy"])
        self.assertGreater(many["tokens"], one["tokens"] * 5)
        lo = lab.simulate(qs, "asc", {"c": 0.8}, repeats=10)
        hi = lab.simulate(qs, "asc", {"c": 0.99}, repeats=10)
        self.assertLess(lo["tokens"], hi["tokens"])
        v = lab.simulate(qs, "vsc_fixed", {"k": 5, "rho": 0.0}, repeats=10)
        s = lab.simulate(qs, "sc", {"k": 5}, repeats=10)
        self.assertGreaterEqual(v["accuracy"], s["accuracy"])  # falsified wrong traces removed

    def test_uab_meets_budget(self):
        qs = self.synthetic(n=40)
        pools = [q.traces for q in qs]
        counts = lab.uab_allocate(pools, avg_samples=4)
        self.assertEqual(sum(counts), 160)
        self.assertTrue(all(2 <= c <= 16 for c in counts))
        r = lab.simulate(qs, "uab", {"avg": 4}, repeats=5)
        self.assertAlmostEqual(r["attempts"], 4.0, places=6)

    def test_interpolation_and_bootstrap(self):
        pts = [{"tokens": 100, "accuracy": 0.5}, {"tokens": 300, "accuracy": 0.7}]
        self.assertAlmostEqual(lab.accuracy_at(pts, 200), 0.6)
        self.assertIsNone(lab.accuracy_at(pts, 50))
        r = lab.paired_bootstrap([1, 1, 0, 1], [0, 1, 0, 0])
        self.assertAlmostEqual(r["mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
