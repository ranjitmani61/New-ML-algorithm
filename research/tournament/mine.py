"""Certified rewrite mining (ledger #33): find answer-preserving edits with the model's own answers
on short problems, then use the certified edits to lengthen labeled short problems.

An edit never reads an answer. It is (kind, segments):
  ("ins1",   (s,))    insert segment s at a uniform random position
  ("spread", (s, t))  insert s, then t, each at an independent uniform random position
  ("block",  (s, t))  insert the adjacent block [s, t] at a uniform random position
Segments are the task's whole term vocabulary: sums (sign, value) for sign +-1 and value 0-9;
S3 the six group elements.

Certification uses only M0's predictions and the true labels of the short originals: an edit's
score is how often M0's answer on the edited problem equals the original's label. The true answer
of an edited problem (`inv`, `exact`) is computed only to measure the certificates afterwards; it
never decides anything.
"""

from __future__ import annotations

import numpy as np

import core as C
import task_s3 as S3


def segments(task):
    if task.name == "sum":
        return [(s, v) for s in (1, -1) for v in range(C.TERM_MAX + 1)]
    return list(range(len(S3.ELEMS)))


def edit_space(task):
    seg = segments(task)
    return ([("ins1", (s,)) for s in seg]
            + [(k, (s, t)) for k in ("spread", "block") for s in seg for t in seg])


def size(edit):
    return len(edit[1])


def apply(rng, item, edit):
    kind, segs = edit
    out = list(item)
    if kind == "block":
        i = int(rng.integers(0, len(out) + 1))
        out[i:i] = list(segs)
    else:
        for s in segs:
            out.insert(int(rng.integers(0, len(out) + 1)), s)
    return out


def exact(task, edit):
    """Evaluation only: does the edit keep the answer of every problem? (Uses the task's algebra.)"""
    kind, segs = edit
    if task.name == "sum":
        return sum(s * v for s, v in segs) == 0  # sums commute, so position never matters
    if kind == "ins1":
        return segs[0] == 0
    if kind == "block":
        return S3.compose(segs[1], segs[0]) == 0  # the block [s, t] contributes t o s
    return segs == (0, 0)  # spread: anything else can land around other elements (S3 has trivial centre)


def certify(task, predict, edits, rng, n=300, base=(1, 2), margin=0.05):
    """Score every edit with `predict` (a list of problems -> list of answers).

    Base problems have `base` segments, so edited problems stay at 2-4 segments. The reference for
    an edit of size k is predict's accuracy on fresh unedited problems of base + k segments.
    Returns (ref, rows)."""
    def draw(m, extra=0):
        return [task.sample(rng, int(rng.integers(base[0], base[1] + 1)) + extra) for _ in range(m)]

    ref = {}
    for k in sorted({size(e) for e in edits}):
        items = draw(4 * n, k)
        ref[k] = float(np.mean([p == task.answer(x) for p, x in zip(predict(items), items)]))
    rows = []
    for e in edits:
        xs = draw(n)
        ys = [task.answer(x) for x in xs]  # true labels of the short originals (allowed)
        ed = [apply(rng, x, e) for x in xs]
        score = float(np.mean([p == y for p, y in zip(predict(ed), ys)]))
        rows.append({"edit": [e[0], [list(s) if isinstance(s, tuple) else s for s in e[1]]],
                     "score": score, "ref": ref[size(e)], "certified": score >= ref[size(e)] - margin,
                     # evaluation only, never used above:
                     "inv": float(np.mean([task.answer(z) == y for z, y in zip(ed, ys)])),
                     "exact": exact(task, e)})
    return ref, rows


def summary(rows):
    cert = [r for r in rows if r["certified"]]
    true = [r for r in rows if r["exact"]]
    both = [r for r in cert if r["exact"]]
    return {"edits": len(rows), "certified": len(cert), "exact": len(true), "certified_and_exact": len(both),
            "precision": len(both) / len(cert) if cert else None,
            "recall": len(both) / len(true) if true else None,
            "certified_not_exact": [[r["edit"], round(r["score"], 3), round(r["inv"], 3)]
                                    for r in cert if not r["exact"]],
            "exact_not_certified": [[r["edit"], round(r["score"], 3)] for r in true if not r["certified"]]}


def as_edit(row_edit):
    kind, segs = row_edit
    return (kind, tuple(tuple(s) if isinstance(s, list) else s for s in segs))


def lengthen(certified, cap):
    """Apply random certified edits until a random length budget (0 .. cap - len) is used up."""
    def tf(rng, item):
        budget = int(rng.integers(0, cap - len(item) + 1))
        out = list(item)
        while True:
            fits = [e for e in certified if size(e) <= budget]
            if not fits:
                return out
            e = fits[int(rng.integers(len(fits)))]
            out = apply(rng, out, e)
            budget -= size(e)
    return tf


def sampler(task, rng, tf, kmax, width, batch=128, p_aug=0.5):
    """Like run2.sampler, but the label is always the short original's: an edited problem
    inherits it and is never re-solved (run2.sampler recomputes answers after expanding)."""
    def draw():
        items, labels, groups = [], [], []
        for _ in range(batch):
            x = task.sample(rng, int(rng.integers(1, kmax + 1)))
            groups.append(task.length(x))
            y = task.answer(x)
            if tf is not None and rng.random() < p_aug:
                x = tf(rng, x)
            items.append(x)
            labels.append(y)
        return task.batch(items, labels, width), np.array(groups, np.int32)
    return draw
