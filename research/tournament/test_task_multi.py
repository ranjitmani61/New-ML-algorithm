"""Checks for task_multi.py: the expansions keep every final amount, keep amounts non-negative,
stay within the length cap, and never read the start amounts."""
import numpy as np

import task_multi as T


class NoRead:
    """Stands in for the start amounts; any access raises."""
    def __getitem__(self, i):
        raise AssertionError("expansion read an amount")

    def __iter__(self):
        raise AssertionError("expansion read an amount")


def test_expansions():
    rng = np.random.default_rng(0)
    grew = {"pairs": 0, "split": 0}
    for t in range(3000):
        item = T.sample(rng, int(rng.integers(1, 5)))
        for name, f in (("pairs", T.expand_pairs), ("split", T.expand_split)):
            out = f(np.random.default_rng(t), item, 16)
            assert T.finals(out) == T.finals(item) and T.valid(out) and T.length(out) <= 16, name
            blind = f(np.random.default_rng(t), (NoRead(), item[1]), 16)
            assert blind[1] == out[1], name  # same events whether or not amounts are readable
            grew[name] += T.length(out) > T.length(item)
    assert min(grew.values()) > 1500, grew


def test_encoding():
    rng = np.random.default_rng(1)
    for _ in range(500):
        item = T.sample(rng, int(rng.integers(1, 17)))
        toks = T.batch([item], [T.answer(item)])[0]
        assert T.decode_answer(toks[-T.ANS:]) == T.answer(item)


if __name__ == "__main__":
    test_expansions()
    test_encoding()
    print("ok")
