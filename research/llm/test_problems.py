import random
import re
import unittest

import problems as P


class Problems(unittest.TestCase):
    def test_answers_match_independent_parse_of_the_text(self):
        """The label must equal what the text says, computed from the text alone."""
        rng = random.Random(0)
        pat = {"gain": [r"buys (\d+) more", r"gives \w+ (\d+) ", r"finds (\d+)", r"wins (\d+)"],
               "loss": [r"gives (\d+) \w+ to", r"loses (\d+)", r"sells (\d+)", r"uses (\d+)"]}
        for _ in range(3000):
            family = rng.choice(["additive", "mixed"])
            p = P.make_problem(rng, rng.randint(1, 16), family, heldout=rng.random() < 0.3)
            x = int(re.match(r"\w+ has (\d+) ", p.text()).group(1))
            for sent in re.split(r"(?<=\.) ", p.text())[1:-1]:
                if "double" in sent:
                    x *= 2
                elif "loses half" in sent:
                    x //= 2
                elif any(re.search(r, sent) for r in pat["loss"]):
                    x -= int(next(re.search(r, sent).group(1) for r in pat["loss"] if re.search(r, sent)))
                elif any(re.search(r, sent) for r in pat["gain"]):
                    x += int(next(re.search(r, sent).group(1) for r in pat["gain"] if re.search(r, sent)))
            self.assertEqual(x, p.answer(), p.text())
            self.assertTrue(p.valid())
            self.assertGreaterEqual(min([p.start] + p.amounts()), 0)

    def test_expansions_keep_the_answer_and_stay_valid(self):
        rng = random.Random(1)
        for _ in range(3000):
            family = rng.choice(["additive", "mixed"])
            p = P.make_problem(rng, rng.randint(1, 4), family)
            for name, f in P.EXPANSIONS.items():
                if name == "cancel_spread" and family == "mixed":
                    continue  # spread pairs do not cancel across doubling/halving
                q = f(rng, p, 16)
                self.assertEqual(q.answer(), p.answer(), (name, q.text()))
                self.assertTrue(q.valid(), (name, q.text()))
                self.assertLessEqual(q.n_changes(), 16)

    def test_spread_pairs_are_wrong_for_the_mixed_family(self):
        """Documents why cancel_spread is restricted to the additive family."""
        p = P.Problem("Mia", "apples", 10, [P.Event("double", 0, "")])
        q = P.Problem("Mia", "apples", 10, [P.Event("gain", 3, ""), P.Event("double", 0, ""), P.Event("loss", 3, "")])
        self.assertNotEqual(p.answer(), q.answer())

    def test_no_plural_after_one(self):
        rng = random.Random(3)
        plurals = set(P.ITEMS_TRAIN + P.ITEMS_HELDOUT + P.OTHER_THINGS)
        for _ in range(3000):
            p = P.make_problem(rng, rng.randint(1, 16), rng.choice(["additive", "mixed"]))
            q = rng.choice(list(P.EXPANSIONS.values()))(rng, p, 16) if p.n_changes() <= 4 else p
            words = q.text().replace(".", " ").replace("?", " ").split()
            for a, b in zip(words, words[1:]):
                self.assertFalse(a == "1" and b in plurals, q.text())

    def test_vocabulary_split(self):
        self.assertFalse(set(P.NAMES_TRAIN) & set(P.NAMES_HELDOUT))
        self.assertFalse(set(P.ITEMS_TRAIN) & set(P.ITEMS_HELDOUT))
        rng = random.Random(2)
        for _ in range(500):
            p = P.make_problem(rng, 8, heldout=True)
            self.assertIn(p.name, P.NAMES_HELDOUT)
            self.assertIn(p.item, P.ITEMS_HELDOUT)
            self.assertFalse(any(n in p.text().split() for n in P.NAMES_TRAIN))


if __name__ == "__main__":
    unittest.main()
