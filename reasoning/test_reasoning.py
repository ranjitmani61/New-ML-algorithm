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
        self.assertTrue(bm.is_equiv("\\begin{pmatrix}\\frac{16}{49}\\\\\\frac{48}{49}\\end{pmatrix}",
                                    "\\begin{pmatrix} 16/49 \\\\ 48/49 \\end{pmatrix}"))
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
        self.assertEqual(self.check("90^3=729,\\!000")["false"], 0)
        self.assertEqual(self.check("so 3^2 + 4^2 = 25")["claims"], 1)

    def test_skips_what_it_cannot_judge(self):
        for text in ("x = 5", "Let n = 3 apples", "x + 2 = 7 so x = 5", "3 / 4 * 2 = 1 1/2 feet",
                     "2 * 225 + 2 * 125 = 450 + 250 = 700", "If $2^8=4^x$, then", "x^2 + 3^2 = 25",
                     "-11+50 = 39", "33\\frac{1}{3}=100/3", "\\frac{279}{99999} = 0.\\overline{00279}"):
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


class LearnedStopping(unittest.TestCase):
    def pools(self, n=60, seed=0):
        """Easy questions (always right) and hopeless ones (scattered wrong answers)."""
        rng = np.random.default_rng(seed)
        qs = []
        for i in range(n):
            q = lab.Question(f"q{i}", "dev", "A")
            hopeless = i % 2 == 1
            for _ in range(32):
                ans = str(rng.integers(0, 50)) if hopeless else "A"
                q.traces.append(trace(ans, tokens=100, conf=2.0 if hopeless else 9.0))
            qs.append(q)
        return qs

    def test_features_are_label_free_and_consistent(self):
        import stopping

        f, leader = stopping.state_features([trace("1"), trace("2"), trace("2")])
        self.assertEqual(leader, "2")
        self.assertAlmostEqual(f["share1"], 2 / 3)
        self.assertEqual(f["margin"], 1)
        self.assertEqual(len(stopping.FULL_FEATURES), len(set(stopping.FULL_FEATURES)))

    def test_stops_early_on_settled_and_futile_questions(self):
        import stopping

        qs = self.pools()
        policy = stopping.StoppingPolicy(lam=1e-3).fit(stopping.build_paths(qs, repeats=4, seed=1))
        eval_paths = stopping.build_paths(qs, repeats=2, seed=2)
        steps = policy.stop_steps(eval_paths)
        self.assertLessEqual(max(steps), 3)  # neither kind of question is worth more attempts
        asc = lab.simulate(qs, "asc", {"c": 0.95}, repeats=2)
        self.assertGreater(asc["attempts"], 10)  # ASC keeps sampling the futile half
        r = stopping.cross_validate(qs, [1e-3], folds=3, repeats_fit=4, repeats_eval=2)[1e-3]
        self.assertAlmostEqual(r["accuracy"], 0.5, places=6)
        self.assertLess(r["attempts"], 3)


class LatentVotes(unittest.TestCase):
    def test_mentioned_values(self):
        import selection

        v = selection.mentioned_values(r"so x = \frac{3}{4} and 2\sqrt{3}, then 10/4 = 2.5, total 1,000")
        for want in ("3/4", "2\\sqrt{3}", "5/2", "10"):
            self.assertIn(want, v)

    def test_latent_support_counts_other_attempts_only(self):
        import selection

        answers = ["26", "1", "1", "13"]
        mentions = [{"26"}, {"26", "1"}, {"1"}, {"26", "13"}]
        s = selection.latent_support(answers, mentions, problem_values={"13"})
        self.assertEqual(s["26"], 2)  # attempts 2 and 4 mention 26 but answer otherwise
        self.assertEqual(s["1"], 0)
        self.assertEqual(s["13"], 0)  # values from the problem statement get no latent support
        traces = [trace(a) for a in answers]
        self.assertEqual(selection.latent_vote(traces, mentions, {"13"}, beta=0.0)[0], "1")
        self.assertEqual(selection.latent_vote(traces, mentions, {"13"}, beta=1.0)[0], "26")


class TerminalRatio(unittest.TestCase):
    def test_waypoint_answer_loses_to_downstream_answer(self):
        import selection

        def t(answer, mentions):
            x = trace(answer)
            x.mentions = frozenset(mentions)
            return x

        # 72 is reached by every attempt; three stop there, two continue to 1152 = 72 * 16.
        pool = [t("72", {"72"}), t("72", {"72"}), t("72", {"72"}), t("1152", {"72", "1152"}), t("1152", {"72", "1152"})]
        self.assertEqual(selection.terminal_ratio_vote(pool, frozenset(), gamma=0.0)[0], "72")  # plain vote
        self.assertEqual(selection.terminal_ratio_vote(pool, frozenset(), gamma=2.0)[0], "1152")
        self.assertEqual(selection.terminal_ratio_vote(pool, frozenset({"72"}), gamma=2.0)[0], "72")  # given value
        # a settled answer that nobody passes through keeps its full vote mass
        settled = [t("5", {"5"}), t("5", {"5"}), t("6", {"6"})]
        a, scores = selection.terminal_ratio_vote(settled, frozenset(), gamma=2.0)
        self.assertEqual((a, scores["5"]), ("5", 2.0))


class Grading(unittest.TestCase):
    def test_equivalent_answers_share_a_cluster(self):
        import grading

        keys, members = grading.cluster(["\\frac{\\sqrt{6}}{2}", "\\frac{\\sqrt6}2", "(-3/2,3)", "(-1.5,3)", None, "3"], "")
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(keys[2], keys[3])
        self.assertIsNone(keys[4])
        self.assertEqual(len(members), 3)
        self.assertTrue(grading.matches_gold("5225 \\text{ cubic inches}", "5225"))
        self.assertTrue(grading.matches_gold("\\frac{35}{2} \\pi", "\\frac{35\\pi}{2}"))
        self.assertFalse(grading.matches_gold("[0,3)", "(-3/2,3)"))
        self.assertFalse(grading.matches_gold("\\frac{1}{3}", "0.33"))

    def test_multiple_choice_letters_map_to_option_values(self):
        import grading

        problem = ("How many? $\\mathrm{\\textbf{(A)} \\ }226\\qquad \\mathrm{\\textbf{(B)} \\ } 243 \\qquad "
                   "\\mathrm{\\textbf{(C)} \\ } 270 \\qquad \\mathrm{\\textbf{(D)} \\ }469\\qquad \\mathrm{\\textbf{(E)} \\ } 486$")
        options = grading.mc_options(problem)
        self.assertEqual(options, {"A": "226", "B": "243", "C": "270", "D": "469", "E": "486"})
        for raw in ("C", "(C)", "\\text{(C)}", "\\textbf{(C) }270"):
            self.assertEqual(grading.map_choice(raw, options), "270")
        self.assertEqual(grading.map_choice("12", options), "12")
        self.assertEqual(grading.mc_options("label(\"$A$\",A,NW); dot(A); (A) only"), {})
        keys, members = grading.cluster(["C", "270", "A"], problem)
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(grading.gold_key("226", members, problem), keys[2])


if __name__ == "__main__":
    unittest.main()
