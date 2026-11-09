"""Checks for mine_multi_toy.py (run: python3 test_mine_multi_toy.py; prints "ok")."""
import numpy as np

import mine_multi_toy as MM
import task_multi as T

rng = np.random.default_rng(0)
assert len(MM.SEGMENTS) == 54 and len(MM.EDITS) == 54 + 2 * 54 * 54
assert sum(MM.exact(e) for e in MM.EDITS) == 108

# exact() agrees with brute force (all exact edits; a random 600 of the others)
others = [e for e in MM.EDITS if not MM.exact(e)]
pick = [e for e in MM.EDITS if MM.exact(e)] + [others[i] for i in rng.choice(len(others), 600, replace=False)]
for e in pick:
    changed = 0
    for _ in range(150):
        x = T.sample(rng, int(rng.integers(0, 6)))
        changed += T.answer(MM.apply(rng, x, e)) != T.answer(x)
    assert (changed == 0) == MM.exact(e), (e, changed)

# apply: length, original unchanged, block adjacency
for i in rng.choice(len(MM.EDITS), 300, replace=False):
    e = MM.EDITS[i]
    x = T.sample(rng, 3)
    before = (list(x[0]), list(x[1]))
    z = MM.apply(rng, x, e)
    assert (list(x[0]), list(x[1])) == before and T.length(z) == 3 + MM.size(e) and z[0] == x[0]
    if e[0] == "block":
        assert any(z[1][j:j + 2] == list(e[1]) for j in range(len(z[1]) - 1))

# lengthen: respects the cap; with exact edits the answer never changes
ex = [e for e in MM.EDITS if MM.exact(e)]
tf = MM.lengthen(ex, 16)
for _ in range(500):
    x = T.sample(rng, int(rng.integers(1, 5)))
    z = tf(rng, x)
    assert T.length(x) <= T.length(z) <= 16 and T.answer(z) == T.answer(x)

# certification with a perfect predictor finds exactly the exact edits (on a subset, for speed)
oracle = lambda items: [T.answer(x) for x in items]  # noqa: E731
subset = ex[:20] + [others[i] for i in rng.choice(len(others), 200, replace=False)]
ref, rows = MM.certify(oracle, np.random.default_rng(1), subset, n=40)
assert ref == {1: 1.0, 2: 1.0}
cert = [MM.as_edit(r["edit"]) for r in rows if r["certified"]]
assert set(cert) == set(ex[:20]), [e for e in cert if not MM.exact(e)][:5]
s = MM.summary(rows)
assert s["precision"] == 1.0 and s["recall"] == 1.0
print("perfect predictor: certified", len(cert), "of", len(subset), "| closest wrong", s["closest_wrong"][:2])

# a predictor blind to the last event certifies wrong edits (the certificates are only as good as M0)
blind = lambda items: [T.answer((x[0], x[1][:-1])) for x in items]  # noqa: E731
_, rows = MM.certify(blind, np.random.default_rng(2), subset, n=40)
s = MM.summary(rows)
print("blind predictor: precision", s["precision"], "certified", s["certified"])
assert s["precision"] is not None and s["precision"] < 0.5
print("ok")
