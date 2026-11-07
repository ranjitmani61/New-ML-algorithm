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
                    with self.assertRaises(ValueError):
                        f(rng, p, 16, family)
                    continue
                q = f(rng, p, 16, family)
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
            fam = "additive" if p.n_changes() > 4 or all(e.kind in ("gain", "loss") for e in p.events) else "mixed"
            q = rng.choice([f for n, f in P.EXPANSIONS.items() if not (n == "cancel_spread" and fam == "mixed")])(rng, p, 16, fam) if p.n_changes() <= 4 else p
            words = q.text().replace(".", " ").replace("?", " ").split()
            for a, b in zip(words, words[1:]):
                self.assertFalse(a == "1" and b in plurals, q.text())

    def test_mixed_adjacent_pairs_include_doubling(self):
        rng = random.Random(4)
        kinds = set()
        for _ in range(300):
            p = P.make_problem(rng, 2, "mixed")
            q = P.cancel_adjacent(rng, p, 16, "mixed")
            kinds |= {e.kind for e in q.events[len(p.events):]} | {e.kind for e in q.events}
        self.assertIn("double", kinds)
        self.assertIn("halve", kinds)

    def test_chain_label_composition_matches_the_text(self):
        """The composed label (labels only) must equal an independent parse of the chained text,
        the problem must stay valid, keep one person and one item, and respect the length cap."""
        rng = random.Random(5)
        pat = {"gain": [r"buys (\d+) more", r"gives \w+ (\d+) ", r"finds (\d+)", r"wins (\d+)"],
               "loss": [r"gives (\d+) \w+ to", r"loses (\d+)", r"sells (\d+)", r"uses (\d+)"]}
        pool = [P.make_problem(rng, rng.randint(1, 4)) for _ in range(500)]
        labels, hit = [], 0
        for _ in range(3000):
            p = P.make_problem(rng, rng.randint(1, 4), heldout=rng.random() < 0.3)
            q, label = P.chain(rng, p, 16, pool)
            x = int(re.match(r"\w+ has (\d+) ", q.text()).group(1))
            for sent in re.split(r"(?<=\.) ", q.text())[1:-1]:
                self.assertIn(p.name, sent)
                self.assertTrue(p.item in sent or P.singular(p.item) in sent, sent)
                if any(re.search(r, sent) for r in pat["loss"]):
                    x -= int(next(re.search(r, sent).group(1) for r in pat["loss"] if re.search(r, sent)))
                else:
                    x += int(next(re.search(r, sent).group(1) for r in pat["gain"] if re.search(r, sent)))
                self.assertGreaterEqual(x, 0, q.text())
            self.assertEqual(x, label, q.text())
            self.assertEqual(q.answer(), label)
            self.assertTrue(q.valid())
            self.assertLessEqual(q.n_changes(), 16)
            self.assertEqual(q.events[:len(p.events)], p.events)
            labels.append(label)
            hit += q.n_changes() > p.n_changes()
        self.assertGreater(max(labels), 70)  # composed labels go beyond the short problems' range
        self.assertGreater(hit, 2400)  # most chains grow (k = 0 pairs leaves the problem unchanged)

    def test_chain_is_additive_only(self):
        rng = random.Random(6)
        pool = [P.make_problem(rng, 2, "mixed") for _ in range(10)]
        with self.assertRaises(ValueError):
            P.chain(rng, pool[0], 16, pool, "mixed")

    def test_chain_length_matches_cancel_spread(self):
        """Same target length distribution as the method (measured, not assumed)."""
        rng = random.Random(7)
        pool = [P.make_problem(rng, rng.randint(1, 4)) for _ in range(2000)]
        base = [P.make_problem(rng, rng.randint(1, 4)) for _ in range(4000)]
        ch = [P.chain(rng, p, 16, pool)[0].n_changes() for p in base]
        cs = [P.cancel_spread(rng, p, 16).n_changes() for p in base]
        self.assertLess(abs(sum(ch) / len(ch) - sum(cs) / len(cs)), 0.15, (sum(ch) / len(ch), sum(cs) / len(cs)))
        self.assertLess(abs(max(ch) - max(cs)), 1)
        # Chain lengths can only fall short of the target, never exceed it; the shortfall is small.
        self.assertLess(sum(cs) / len(cs) - sum(ch) / len(ch), 0.15)

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
