"""Checks for mine_multi.py (run: python3 test_mine_multi.py; prints "ok")."""
import random

import multi as M
import mine_multi as X

rng = random.Random(0)
assert len(X.SEGMENTS) == 54 and len(X.EDITS) == 54 + 2 * 54 * 54
assert sum(X.exact(e) for e in X.EDITS) == 108  # same person, opposite kind, same amount; spread and block

# exact() agrees with brute force (all exact edits; a random 600 of the others)
others = [e for e in X.EDITS if not X.exact(e)]
for e in [e for e in X.EDITS if X.exact(e)] + rng.sample(others, 600):
    changed = 0
    for _ in range(150):
        p = M.make_multi(rng, rng.randint(0, 5))
        changed += X.apply_edit(rng, p, e).answer() != p.answer()
    assert (changed == 0) == X.exact(e), (e, changed)

# apply_edit: lengths, the problem's own people and item, adjacency of blocks, original unchanged
for e in rng.sample(X.EDITS, 300):
    p = M.make_multi(rng, 3)
    before = p.text()
    q = X.apply_edit(rng, p, e)
    assert p.text() == before and q.n_changes() == p.n_changes() + X.size(e)
    new = [ev for ev in q.events if all(ev is not x for x in p.events)]  # by identity: _copy keeps the old objects
    assert [(ev.who, ev.kind, ev.n) for ev in new] == list(e[1]) or e[0] == "spread"
    for ev in new:
        assert q.people[ev.who] in ev.text and (q.item in ev.text or M.singular(q.item) in ev.text)
    if e[0] == "block":
        idx = [next(i for i, x in enumerate(q.events) if x is ev) for ev in new]
        assert idx[1] == idx[0] + 1

# lengthen: respects the cap; with exact edits the answer never changes
exact_edits = [e for e in X.EDITS if X.exact(e)]
for _ in range(500):
    p = M.make_multi(rng, rng.randint(1, 4))
    q = X.lengthen(rng, p, exact_edits, 16)
    assert p.n_changes() <= q.n_changes() <= 16 and q.answer() == p.answer()

# certification with a perfect predictor finds exactly the exact edits (on a subset, for speed)
oracle = lambda probs: [p.answer() for p in probs]  # noqa: E731
subset = exact_edits[:20] + rng.sample(others, 200)
rc = random.Random(1)
ref = X.reference(oracle, rc, 50)
table, cert, n2 = X.certify(oracle, rc, subset, ref, 8, 1, 32, 0.05)
assert ref == {1: 1.0, 2: 1.0}
assert set(cert) == set(exact_edits[:20]), (len(cert), [e for e in cert if not X.exact(e)][:5])
print("perfect predictor: certified", len(cert), "of", len(subset), "| stage 2:", n2)
print("ok")
