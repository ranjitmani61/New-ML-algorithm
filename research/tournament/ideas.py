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


def t_dup(rng, items):
    """Answer-predictable expansion: the problem plus a shuffled copy of itself; value doubles.
    Returns (items, value) because the label changes in a known way."""
    out = list(items) + list(items)
    order = rng.permutation(len(out))
    return [out[i] for i in order], 2 * C.value(items)


def t_npia_dup(rng, items):
    """Half the time inverse-pair insertion (value kept), half the time self-duplication (doubled)."""
    if rng.random() < 0.5:
        return t_npia(rng, items), C.value(items)
    return t_dup(rng, items)


def t_fill(rng, items):
    """Reference: insert neutral filler tokens (like random spaces); value unchanged."""
    m = int(rng.integers(0, 8 - len(items) + 1))
    return _insert(rng, items, [None] * m)


IDEAS = {
    "baseline": dict(kind="baseline", hypothesis="vanilla softmax transformer, rotary positions",
                     prediction="8-term accuracy 5-15%"),
    "ssmax": dict(kind="reference", attention="ssmax",
                  hypothesis="length-scaled logits keep attention sharp at longer lengths",
                  prediction="small to moderate gain; the sum still needs a count"),
    "sigmoid": dict(kind="reference", attention="sigmoid",
                    hypothesis="unnormalised attention can output sums, which extrapolate linearly",
                    prediction="large gain if training finds the sum solution"),
    "rpe": dict(kind="reference", pos_offset_max=16,
                hypothesis="random position offsets expose larger position values",
                prediction="small gain: rotary positions are relative already"),
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
    "dup": dict(kind="candidate", transform=t_dup, label_changes=True,
                hypothesis="shuffled self-duplication (label doubled) covers long inputs with large sums, "
                           "the regime inverse pairs never show",
                prediction="better than baseline on large sums; weaker than npia on small ones"),
    "npia_dup": dict(kind="candidate", transform=t_npia_dup, label_changes=True,
                     hypothesis="inverse pairs (structure) plus duplication (large sums) together cover "
                                "8-term inputs across the value range",
                     prediction="at least npia's 8-term accuracy, and above 50% on |value| >= 20"),
    "wed": dict(kind="candidate", wed=True,
                hypothesis="weights move along a difficulty direction from 1-3 to 1-4 terms; "
                           "extrapolating theta4 + a (theta4 - theta3) continues it",
                prediction="no gain or harm (wild card)"),
    "oracle": dict(kind="reference", max_terms=8,
                   hypothesis="upper bound: true labels on 1-8 terms (violates the short-only constraint)",
                   prediction="high; calibrates how much of the possible gain an idea recovers"),
    "rex": dict(kind="candidate", loss="rex", rex_lambda=10.0,
                hypothesis="penalising the variance of risk across lengths 1-4 favours a "
                           "length-invariant mechanism (REx with length as environment)",
                prediction="0 to +10 points"),
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
