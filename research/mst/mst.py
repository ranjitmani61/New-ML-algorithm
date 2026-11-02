"""Label-free self-training conditions for the metamorphic self-training pilot.

Every condition gets the same unlabeled hard expressions, the same number of sampled answers per
expression (K) and the same training steps. They differ only in how a pseudo-label is formed:

  none         no pseudo-labels; the steps are spent on labeled short data only
  majority     K samples on the expression; label = most common answer (TTRL-style)
  invariance   K/G samples on the expression and on G-1 term permutations; votes pooled
  metamorphic  K/G samples on the expression and on G-1 single-term shifts; each transform's
               answers are mapped back by the known law (a - s_i * d); votes pooled
  placebo      as metamorphic, but mapped back with the wrong law (a + s_i * d)
  oracle       the true value (upper bound)

A pseudo-label is kept when its pooled vote share is at least `keep` (same rule for all).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

import lab
import tasks as T

CONDITIONS = ("none", "majority", "invariance", "metamorphic", "placebo", "oracle")
RELATION = {"invariance": T.permute, "metamorphic": T.shift_term, "placebo": T.shift_term_wrong}


def pseudo_label(learner, rng, exprs, condition, K=8, groups=4, keep=0.5):
    """Returns (labels, kept mask, vote share) for a list of expressions (term lists)."""
    n = len(exprs)
    if condition == "oracle":
        return [T.value(e) for e in exprs], np.ones(n, bool), np.ones(n)
    if condition == "none":
        return [None] * n, np.zeros(n, bool), np.zeros(n)
    if condition == "majority":
        variants = [[(e, lambda a: a)] * K for e in exprs]
    else:
        rel = RELATION[condition]
        per = K // groups
        variants = []
        for e in exprs:
            v = [(e, lambda a: a)] * per
            for _ in range(groups - 1):
                v += [rel(rng, e)] * per
            variants.append(v)
    flat = [x for v in variants for x, _ in v]
    answers = learner.answers(T.batch(flat), sample=True)
    labels, kept, share = [], np.zeros(n, bool), np.zeros(n)
    i = 0
    for j, v in enumerate(variants):
        mapped = []
        for (_, back), a in zip(v, answers[i:i + len(v)]):
            if a is not None:
                m = back(a)
                if abs(m) < 10 ** T.ANS_DIGITS:
                    mapped.append(m)
        i += len(v)
        if mapped:
            lab_, c = Counter(mapped).most_common(1)[0]
            labels.append(lab_)
            share[j] = c / len(v)
            kept[j] = share[j] >= keep
        else:
            labels.append(None)
    return labels, kept, share


def label_quality(exprs, labels, kept):
    truth = np.array([T.value(e) for e in exprs])
    ok = np.array([l is not None and l == t for l, t in zip(labels, truth)])
    return {"coverage": float(kept.mean()), "precision": float(ok[kept].mean()) if kept.any() else None,
            "correct_labels_per_expr": float((ok & kept).mean()), "wrong_labels_per_expr": float((~ok & kept).mean())}


def mixed_sampler(rng, short_ns, pool, batch=128):
    """Half labeled short expressions, half pseudo-labeled hard ones (or all short if pool is empty)."""
    short = lab.labeled_sampler(rng, short_ns, batch)
    if not pool:
        return short

    def draw():
        a, w = short()
        h = batch // 2
        idx = rng.integers(0, len(pool), size=h)
        exprs = [pool[i][0] for i in idx]
        b = T.batch(exprs, [pool[i][1] for i in idx])
        a[h:] = b
        return a, w
    return draw
