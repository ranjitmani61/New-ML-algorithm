"""Idea registry for the tournament. Each idea changes one thing relative to the baseline.

kind: "baseline", "reference" (known method from the literature) or "candidate" (ours to test).
Hypotheses and predictions are written here before the idea is run; results go to LEDGER.md.
"""

from __future__ import annotations

import numpy as np

import core as C

MAX_IN_SHORT = 2 * 4 + 1
MAX_IN_LONG = 2 * 8 + 1


def _insert(rng, items, extra):
    out = list(items)
    for x in extra:
        out.insert(int(rng.integers(0, len(out) + 1)), x)
    return out


def t_neia(rng, items):
    """Identity-element insertion: add zero terms (random sign) up to 8 terms; value unchanged."""
    m = int(rng.integers(0, 8 - len(items) + 1))
    return _insert(rng, items, [(int(rng.choice([1, -1])), 0) for _ in range(m)])


def t_npia(rng, items):
    """Inverse-pair insertion: add pairs (+a, -a) at random positions up to 8 terms; value unchanged."""
    k = int(rng.integers(0, (8 - len(items)) // 2 + 1))
    extra = []
    for _ in range(k):
        a = int(rng.integers(1, C.TERM_MAX + 1))
        extra += [(1, a), (-1, a)]
    return _insert(rng, items, extra)


def t_fill(rng, items):
    """Reference: insert neutral filler tokens (like random spaces); value unchanged."""
    m = int(rng.integers(0, 8 - len(items) + 1))
    return _insert(rng, items, [None] * m)


IDEAS = {
    "baseline": dict(kind="baseline", hypothesis="vanilla softmax transformer, rotary positions",
                     prediction="8-term accuracy 5-15%"),
    "sigmoid": dict(kind="reference", attention="sigmoid",
                    hypothesis="unnormalised attention can output sums, which extrapolate linearly",
                    prediction="large gain if training finds the sum solution"),
    "fill": dict(kind="reference", transform=t_fill,
                 hypothesis="filler tokens lengthen inputs without new terms",
                 prediction="0 to +20 points"),
    "neia": dict(kind="candidate", transform=t_neia,
                 hypothesis="errors are caused by structural length (pilot probe); zero terms cover "
                            "length 5-8 with exact labels while non-zero terms stay at most 4",
                 prediction="+30 points or more at 8 terms if length is the cause"),
    "npia": dict(kind="candidate", transform=t_npia,
                 hypothesis="inverse pairs cover both structural length and the number of non-zero "
                            "terms, labels unchanged (no new sum is computed)",
                 prediction="+40 points or more at 8 terms"),
    "oracle": dict(kind="reference", max_terms=8,
                   hypothesis="upper bound: true labels on 1-8 terms (violates the short-only constraint)",
                   prediction="high; calibrates how much of the possible gain an idea recovers"),
}


def sampler(rng, idea, batch=128, max_terms=4, p_aug=0.5):
    tf = idea.get("transform")
    max_terms = idea.get("max_terms", max_terms)
    max_in = MAX_IN_LONG if (tf or max_terms > 4) else MAX_IN_SHORT

    def draw():
        items, groups, labels = [], [], []
        for _ in range(batch):
            x = C.random_terms(rng, int(rng.integers(1, max_terms + 1)))
            groups.append(len(x))
            y = C.value(x)
            if tf is not None and rng.random() < p_aug:
                if idea.get("label_changes"):
                    x, y = tf(rng, x)
                else:
                    x = tf(rng, x)
                    y = C.value(x)
            items.append(x)
            labels.append(y)
        return C.batch(items, labels, max_in=max_in), np.array(groups, np.int32)
    return draw
