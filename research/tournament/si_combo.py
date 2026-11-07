"""Exploratory toy (not registered): does self-improvement go further when it starts from the
inverse-pair (npia) model instead of the plain baseline? Tournament testbed: signed sums, labels on
1-4 terms, 3-layer width-64 transformer.

    python3 si_combo.py OUT.json --start baseline|npia [--rounds 6 --models 3 --n 2000 --round-steps 500]

Self-improvement as in Lee et al. (2025): 3 models, keep a self-label only when all agree, each
round +2 terms, newest round sampled at 50%. Equal schedule and compute for both starts.
"""
import argparse
import json
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import core as C  # noqa: E402
import engine as E  # noqa: E402
import ideas as I  # noqa: E402

TASK = E.SumTask
WIDTH = TASK.max_in(16)
TEST_NS = [4, 8, 10, 12, 14, 16]

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--start", choices=("baseline", "npia"), required=True)
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--models", type=int, default=3)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--round-steps", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cap", type=int, default=8, help="npia start: pairs up to this many terms (8 = ledger #8)")
ap.add_argument("--expand-pseudo-cap", type=int, default=0,
                help="if > 0, self-labeled problems are also lengthened with pairs up to this many terms (#25)")
a = ap.parse_args()


def pairs_to(rng, items, cap):
    """Ledger #8's t_npia with the cap as a parameter (same random draws; cap 8 gives t_npia exactly)."""
    k = int(rng.integers(0, (cap - len(items)) // 2 + 1))
    extra = []
    for _ in range(k):
        b = int(rng.integers(1, C.TERM_MAX + 1))
        extra += [(1, b), (-1, b)]
    return I._insert(rng, items, extra)


expand = (lambda rng, x: pairs_to(rng, x, a.cap)) if a.start == "npia" else None


def base_draw(rng, batch):
    """The start condition's own training distribution (labels on 1-4 terms; npia: pairs to 8)."""
    items = []
    for _ in range(batch):
        x = TASK.sample(rng, int(rng.integers(1, 5)))
        if expand is not None and rng.random() < 0.5:
            x = expand(rng, x)
        items.append(x)
    return items, [TASK.answer(x) for x in items]


def sampler_round(rng, kept_old, kept_new, batch=128):
    def draw():
        items, ans = [], []
        for _ in range(batch):
            if kept_new and rng.random() < 0.5:
                x, y = kept_new[int(rng.integers(len(kept_new)))]
                if a.expand_pseudo_cap and rng.random() < 0.5:  # pairs keep the self-generated label
                    x = pairs_to(rng, x, a.expand_pseudo_cap)
            else:  # older data: the base distribution and earlier rounds, uniformly by example source
                pool_old = len(kept_old)
                if pool_old and rng.random() < pool_old / (pool_old + 20_000):
                    x, y = kept_old[int(rng.integers(pool_old))]
                    if a.expand_pseudo_cap and rng.random() < 0.5:
                        x = pairs_to(rng, x, a.expand_pseudo_cap)
                else:
                    (x,), (y,) = base_draw(rng, 1)
            items.append(x); ans.append(y)
        return TASK.batch(items, ans, WIDTH), np.zeros(batch, np.int32)
    return draw


t0 = time.time()
models = []
for m in range(a.models):
    ms = 1000 * a.seed + m
    p, cfg = E.init(jax.random.PRNGKey(ms), TASK)
    rng = np.random.default_rng(ms)

    def draw0(rng=rng):
        items, ans = base_draw(rng, 128)
        return TASK.batch(items, ans, WIDTH), np.zeros(128, np.int32)
    p = E.train(p, cfg, draw0, a.steps, 2e-3, ms)
    models.append(p)
g = jax.jit(lambda pp, x: E.greedy(pp, cfg, x))
log = {"start": a.start, "args": vars(a), "round0": E.evaluate(models[0], cfg, TASK, 7, TEST_NS, 500), "rounds": []}
print(json.dumps({"round0": log["round0"], "t": round(time.time() - t0)}), flush=True)
kept_old, kept_new = [], []
for r in range(1, a.rounds + 1):
    lo, hi = 4 + 2 * r - 1, 4 + 2 * r
    prng = np.random.default_rng(50_000 + 100 * a.seed + r)
    pool = [TASK.sample(prng, int(prng.integers(lo, hi + 1))) for _ in range(a.n)]
    X = jnp.asarray(TASK.batch(pool, None, WIDTH))
    preds = [[TASK.decode(o) for o in np.asarray(g(p, X))] for p in models]
    kept_old += kept_new
    kept_new = [(x, v[0]) for x, *v in zip(pool, *preds) if v[0] is not None and all(u == v[0] for u in v)]
    acc_kept = float(np.mean([y == TASK.answer(x) for x, y in kept_new])) if kept_new else None
    for m in range(a.models):
        rng = np.random.default_rng(10_000 * (m + 1) + r)
        models[m] = E.train(models[m], cfg, sampler_round(rng, kept_old, kept_new), a.round_steps, 2e-3, m, warmup=20)
    ev = E.evaluate(models[0], cfg, TASK, 7, TEST_NS, 500)
    row = {"round": r, "terms": [lo, hi], "kept": len(kept_new), "kept_label_acc": acc_kept,
           "model_acc_on_pool": [float(np.mean([pp == TASK.answer(x) for pp, x in zip(pr, pool)])) for pr in preds],
           "eval_model0": ev, "t": round(time.time() - t0)}
    log["rounds"].append(row)
    print(json.dumps(row), flush=True)
    json.dump(log, open(a.out, "w"), indent=1)
json.dump(log, open(a.out, "w"), indent=1)
