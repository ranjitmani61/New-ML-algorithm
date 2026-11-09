"""Certified rewrite mining (ledger #33) on the toy three-person task (task_multi.py).

The same algorithm as research/llm/mine_multi.py, for the small model:
  segments  an event (person 0-2, gain or loss, 1-9): 54 segments
  edits     ("ins1", (s,)) insert s at a uniform random position;
            ("spread", (s, t)) insert s, then t, each at an independent uniform random position;
            ("block", (s, t)) insert the adjacent block [s, t] at a uniform random position.
            54 + 2 x 54^2 = 5,886 edits; an edit never reads an answer.
  certify   apply an edit to problems with 1-2 events; score = how often M0's answer on the edited
            problem equals the original's true label; certified if the score is at least M0's
            accuracy on unedited problems of the same lengths minus `margin`.
The true answers of edited problems (`inv`, `exact`) are computed only to measure the certificates.
"""

from __future__ import annotations

import numpy as np

import task_multi as T

SEGMENTS = [(who, kind, n) for who in range(T.N) for kind in (0, 1) for n in range(1, 10)]
EDITS = ([("ins1", (s,)) for s in SEGMENTS]
         + [(k, (s, t)) for k in ("spread", "block") for s in SEGMENTS for t in SEGMENTS])


def size(edit):
    return len(edit[1])


def exact(edit):
    """Evaluation only: does the edit keep the largest final amount of every problem?"""
    if edit[0] == "ins1":
        return False
    (w1, k1, n1), (w2, k2, n2) = edit[1]
    return w1 == w2 and k1 != k2 and n1 == n2


def apply(rng, item, edit):
    starts, events = item
    ev = list(events)
    if edit[0] == "block":
        i = int(rng.integers(0, len(ev) + 1))
        ev[i:i] = list(edit[1])
    else:
        for s in edit[1]:
            ev.insert(int(rng.integers(0, len(ev) + 1)), s)
    return (list(starts), ev)


def certify(predict, rng, edits=EDITS, n=300, margin=0.05):
    """predict: list of items -> list of answers (None if unparsable). Returns (ref, rows)."""
    def draw(m, extra=0):
        return [T.sample(rng, int(rng.integers(1, 3)) + extra) for _ in range(m)]

    ref = {}
    for k in (1, 2):
        items = draw(4 * n, k)
        ref[k] = float(np.mean([p == T.answer(x) for p, x in zip(predict(items), items)]))
    rows = []
    for e in edits:
        xs = draw(n)
        ys = [T.answer(x) for x in xs]          # true labels of the short originals (allowed)
        ed = [apply(rng, x, e) for x in xs]
        score = float(np.mean([p == y for p, y in zip(predict(ed), ys)]))
        rows.append({"edit": [e[0], [list(s) for s in e[1]]], "score": score, "ref": ref[size(e)],
                     "certified": score >= ref[size(e)] - margin,
                     # evaluation only, never used above:
                     "inv": float(np.mean([T.answer(z) == y for z, y in zip(ed, ys)])),
                     "valid": float(np.mean([T.valid(z) for z in ed])), "exact": exact(e)})
    return ref, rows


def summary(rows):
    cert = [r for r in rows if r["certified"]]
    true = [r for r in rows if r["exact"]]
    both = [r for r in cert if r["exact"]]
    wrong = [r for r in rows if not r["exact"]]
    return {"edits": len(rows), "certified": len(cert), "exact": len(true), "certified_and_exact": len(both),
            "precision": len(both) / len(cert) if cert else None,
            "recall": len(both) / len(true) if true else None,
            "certified_not_exact": [[r["edit"], round(r["score"], 3), round(r["inv"], 3)] for r in cert if not r["exact"]],
            "closest_wrong": max(([round(r["score"], 3), round(r["inv"], 3), r["edit"]] for r in wrong), default=None)}


def as_edit(row_edit):
    kind, segs = row_edit
    return (kind, tuple(tuple(s) for s in segs))


def lengthen(certified, cap):
    """Random certified edits until a random length budget (0 .. cap - events) is used up."""
    def tf(rng, item, _max_k=None):
        budget = int(rng.integers(0, cap - T.length(item) + 1))
        out = item
        while True:
            fits = [e for e in certified if size(e) <= budget]
            if not fits:
                return out
            e = fits[int(rng.integers(len(fits)))]
            out = apply(rng, out, e)
            budget -= size(e)
    return tf
