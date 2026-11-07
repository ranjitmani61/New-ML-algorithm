import random
import re
import unittest
from unittest import mock

import multi as M
import problems as P

GAIN_PAT = [r"(\w+) buys (\d+) more", r"\w+ gives (\w+) (\d+) ", r"(\w+) finds (\d+)", r"(\w+) wins (\d+)"]
LOSS_PAT = [r"(\w+) gives (\d+) \w+ to", r"(\w+) loses (\d+)", r"(\w+) sells (\d+)", r"(\w+) uses (\d+)"]


def parse_text(text):
    """Independent simulation from the text alone: returns the final amount of each named person."""
    sents = re.split(r"(?<=[.?]) ", text)
    x = {name: int(n) for name, n in re.findall(r"(\w+) has (\d+) ", sents[0])}
    for s in sents[1:-1]:
        for pat, sign in [(p, -1) for p in LOSS_PAT] + [(p, 1) for p in GAIN_PAT]:
            m = re.search(pat, s)
            if m:
                x[m.group(1)] += sign * int(m.group(2))
                assert x[m.group(1)] >= 0, text
                break
        else:
            raise AssertionError(f"unparsed sentence: {s}")
    return x


class Multi(unittest.TestCase):
    def test_answer_matches_independent_parse(self):
        rng = random.Random(0)
        for _ in range(3000):
            p = M.make_multi(rng, rng.randint(1, 16), heldout=rng.random() < 0.3)
            x = parse_text(p.text())
            self.assertEqual(sorted(x), sorted(p.people))
            self.assertEqual(max(x.values()), p.answer(), p.text())
            self.assertEqual([x[n] for n in p.people], p.finals())
            self.assertTrue(p.valid())
            self.assertEqual(p.n_changes(), len(p.events))

    def test_cancel_spread_keeps_every_final_amount(self):
        rng = random.Random(1)
        for _ in range(3000):
            p = M.make_multi(rng, rng.randint(1, 4), heldout=rng.random() < 0.3)
            q = M.cancel_spread(rng, p, 16)
            self.assertEqual(q.finals(), p.finals())
            self.assertEqual(q.answer(), p.answer())
            self.assertTrue(q.valid())
            self.assertLessEqual(q.n_changes(), 16)
            self.assertEqual(parse_text(q.text()), parse_text(p.text()))
            # the original events stay in order
            it = iter(q.events)
            self.assertTrue(all(any(e is f for f in it) for e in p.events))

    def test_cancel_spread_never_reads_amounts(self):
        """The construction must not simulate the problem: amount lookups raise during the call."""
        rng = random.Random(2)
        p = M.make_multi(rng, 3)
        boom = mock.Mock(side_effect=AssertionError("simulated"))
        with mock.patch.object(M.Problem, "amounts", boom), mock.patch.object(M.Problem, "finals", boom), \
                mock.patch.object(M.Problem, "answer", boom), mock.patch.object(M.Problem, "valid", boom):
            for _ in range(200):
                M.cancel_spread(rng, p, 16)
        boom.assert_not_called()

    def test_gain_comes_before_its_loss(self):
        rng = random.Random(3)
        for _ in range(500):
            p = M.make_multi(rng, rng.randint(1, 4))
            q = M.cancel_spread(rng, p, 16)
            extra = [e for e in q.events if not any(e is f for f in p.events)]
            open_gains = {}
            for e in extra:
                k = (e.who, e.n)
                if e.kind == "gain":
                    open_gains[k] = open_gains.get(k, 0) + 1
                else:
                    self.assertGreater(open_gains.get(k, 0), 0, q.text())
                    open_gains[k] -= 1
            self.assertTrue(all(v == 0 for v in open_gains.values()))

    def test_answers_cannot_be_composed_from_labels(self):
        """Two problems with the same label, continued by the same event, get different labels:
        a label-only composition rule (what chaining needs) cannot exist for this question."""
        a = M.Problem(["Mia", "Leo", "Ava"], "apples", [10, 20, 5])
        b = M.Problem(["Mia", "Leo", "Ava"], "apples", [20, 10, 5])
        self.assertEqual(a.answer(), b.answer())
        e = M.Event(0, "gain", 9, "Mia finds 9 apples.")
        a.events.append(e)
        b.events.append(e)
        self.assertNotEqual(a.answer(), b.answer())

    def test_vocabulary_and_other_names(self):
        rng = random.Random(4)
        for _ in range(1000):
            held = rng.random() < 0.5
            p = M.make_multi(rng, rng.randint(1, 16), heldout=held)
            names = P.NAMES_HELDOUT if held else P.NAMES_TRAIN
            items = P.ITEMS_HELDOUT if held else P.ITEMS_TRAIN
            self.assertTrue(set(p.people) <= set(names) and len(set(p.people)) == 3)
            self.assertIn(p.item, items)
            words = set(re.findall(r"[A-Z][a-z]+", p.text())) - {"How"}
            self.assertTrue(words <= set(names), (words, p.text()))
            for e in p.events:  # every event names exactly one of the three
                self.assertEqual(sum(n in e.text.split() for n in p.people), 1, e.text)

    def test_no_plural_after_one(self):
        rng = random.Random(5)
        plurals = set(P.ITEMS_TRAIN + P.ITEMS_HELDOUT)
        for _ in range(2000):
            p = M.cancel_spread(rng, M.make_multi(rng, rng.randint(1, 4)), 16)
            w = p.text().replace(".", " ").replace(",", " ").replace("?", " ").split()
            for a, b in zip(w, w[1:]):
                self.assertFalse(a == "1" and b in plurals, p.text())


class Runner(unittest.TestCase):
    def setUp(self):
        import run_multi as RM
        self.RM = RM

    def test_self_improve_starts_from_the_plain_training_set(self):
        a = self.RM.build_train(random.Random(9), "plain", 300, 4, 16)
        b = self.RM.build_train(random.Random(9), "self_improve", 300, 4, 16)
        self.assertEqual([p.text() for p in a], [p.text() for p in b])
        self.assertTrue(all(1 <= p.n_changes() <= 4 for p in a))

    def test_condition_lengths(self):
        cs = self.RM.build_train(random.Random(9), "cancel_spread", 2000, 4, 16)
        orc = self.RM.build_train(random.Random(9), "oracle", 2000, 4, 16)
        self.assertLessEqual(max(p.n_changes() for p in cs), 16)
        self.assertGreater(max(p.n_changes() for p in cs), 12)
        self.assertEqual({p.n_changes() for p in orc}, set(range(1, 17)))

    def test_vote_needs_full_agreement(self):
        v = self.RM.vote([["12", "7", "x", "30", " 5"], ["12", "8", "3", "30", "5"], ["12", "7", "3", "30 apples", "5"]])
        self.assertEqual(v, [12, None, None, 30, 5])

    def test_sampler(self):
        s = self.RM.si_sampler(1000, 200, 16, 0)
        idx = [i for step in range(2000) for i in s(step)]
        self.assertTrue(all(0 <= i < 1200 for i in idx))
        frac_new = sum(i >= 1000 for i in idx) / len(idx)
        self.assertAlmostEqual(frac_new, 0.5, delta=0.02)
        s0 = self.RM.si_sampler(1000, 0, 16, 0)
        self.assertTrue(all(i < 1000 for step in range(100) for i in s0(step)))

    def test_schedule(self):
        self.assertEqual(self.RM.si_schedule(4, 16, 2), [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)])
        self.assertEqual(self.RM.si_schedule(4, 16, 1), [(k, k) for k in range(5, 17)])

    def test_self_generated_label_is_what_gets_trained(self):
        import run_llm as R

        class Tok:
            eos_token_id = 0

            def __call__(self, s, add_special_tokens=False):
                return {"input_ids": [ord(c) for c in s]}

            def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
                return msgs[-1]["content"] + "|"
        p = M.make_multi(random.Random(6), 3)
        ids, labels, plen = R.encode_example(Tok(), self.RM.Labeled(p.text(), p.answer() + 7), 384)
        target = "".join(chr(t) for t in labels if t not in (-100, 0))
        self.assertEqual(target, str(p.answer() + 7))

    def test_eval_sets_are_fixed_by_seed(self):
        a = self.RM.build_eval(9, [4, 12], 50)
        b = self.RM.build_eval(9, [4, 12], 50)
        self.assertEqual([p.text() for p in a[12]], [p.text() for p in b[12]])
        c = self.RM.build_eval(10, [12], 50)
        self.assertNotEqual([p.text() for p in a[12]], [p.text() for p in c[12]])


if __name__ == "__main__":
    unittest.main()
