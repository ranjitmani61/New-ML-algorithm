"""Checks for mine.py (run: python3 test_mine.py; prints "ok")."""
import numpy as np

import engine as E
import mine as M

rng = np.random.default_rng(0)

for task in (E.SumTask, E.S3Task):
    edits = M.edit_space(task)
    n_seg = len(M.segments(task))
    assert len(edits) == n_seg + 2 * n_seg ** 2, task.name

    # apply: lengths, and a block stays adjacent
    for e in edits[:50] + edits[-50:]:
        x = task.sample(rng, 3)
        z = M.apply(rng, x, e)
        assert len(z) == len(x) + M.size(e)
        if e[0] == "block":
            assert any(z[i:i + 2] == list(e[1]) for i in range(len(z) - 1))

    # exact() agrees with brute force: exact edits never change an answer, the rest sometimes do
    for e in edits:
        changed = 0
        for _ in range(300):
            x = task.sample(rng, int(rng.integers(0, 6)))
            changed += task.answer(M.apply(rng, x, e)) != task.answer(x)
        assert (changed == 0) == M.exact(task, e), (task.name, e, changed)

    # certification with a perfect predictor: exactly the exact edits pass
    oracle = lambda items: [task.answer(x) for x in items]
    ref, rows = M.certify(task, oracle, edits, np.random.default_rng(1), n=200)
    assert all(v == 1.0 for v in ref.values())
    s = M.summary(rows)
    assert s["precision"] == 1.0 and s["recall"] == 1.0, (task.name, s)
    print(task.name, "edits", len(edits), "exact", s["exact"], "| perfect-predictor certification ok")

    # a predictor that ignores the last segment certifies edits it should not: the check must see it
    blind = lambda items: [task.answer(x[:-1]) if len(x) > 1 else task.answer(x) for x in items]
    _, rows_b = M.certify(task, blind, edits, np.random.default_rng(2), n=200)
    print(task.name, "blind predictor: precision", round(M.summary(rows_b)["precision"] or 0, 3))

# the sampler inherits the short label even when an edit changes the true answer
task = E.SumTask
bad = [("ins1", ((1, 5),))]                       # +5 changes every answer
tf = M.lengthen(bad, 8)
toks, _ = M.sampler(task, np.random.default_rng(3), tf, 4, task.max_in(8), batch=64, p_aug=1.0)()
r2 = np.random.default_rng(3)
for row in toks:
    x = task.sample(r2, int(r2.integers(1, 5)))
    y = task.answer(x)
    r2.random()
    z = tf(r2, x)
    assert task.decode(row[-task.ANS:]) == y and task.answer(z) == y + 5 * (len(z) - len(x))
print("sampler inherits labels: ok")

# lengthen respects the cap and only uses edits that fit
good = [e for e in M.edit_space(task) if M.exact(task, e)]
tf = M.lengthen(good, 8)
for _ in range(2000):
    x = task.sample(rng, int(rng.integers(1, 5)))
    z = tf(rng, x)
    assert len(x) <= len(z) <= 8 and task.answer(z) == task.answer(x)
print("ok")
