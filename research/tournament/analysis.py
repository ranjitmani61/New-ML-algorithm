"""Mechanism checks for a kept sum-task idea, from saved parameters (no retraining).

    python3 analysis.py PARAMS.pkl SEED

1. accuracy at 8 terms by |true value| (are sums beyond the training range, |v| > 36, solved?)
2. lengths beyond the augmentation range (10, 12, 16 terms)
3. random inputs vs inputs that contain cancelling pairs (does the model only handle the
   augmentation's structure?)
"""

import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

import core as C
import engine as E

p = pickle.load(open(sys.argv[1], "rb"))
seed = int(sys.argv[2])
_, cfg = E.init(jax.random.PRNGKey(0), E.SumTask)
g = jax.jit(lambda pp, x: E.greedy(pp, cfg, x))
rng = np.random.default_rng(777_000 + seed)


def acc(items, width):
    out = np.asarray(g(p, jnp.asarray(C.batch(items, max_in=width))))
    return np.array([C.decode_answer(o) == C.value(x) for o, x in zip(out, items)])


items = [C.random_terms(rng, 8) for _ in range(4000)]
ok = acc(items, 33)
v = np.abs([C.value(x) for x in items])
print("8 terms by |value|:", {f"[{lo},{hi})": (round(float(ok[(v >= lo) & (v < hi)].mean()), 3), int(((v >= lo) & (v < hi)).sum()))
                              for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 37), (37, 73))})
for n in (8, 10, 12, 16):
    print(f"{n} terms:", round(float(acc([C.random_terms(rng, n) for _ in range(2000)], 33).mean()), 3))
struct = []
for _ in range(2000):
    core = C.random_terms(rng, 4)
    extra = []
    for _ in range(2):
        a = int(rng.integers(1, 10))
        extra += [(1, a), (-1, a)]
    x = list(core)
    for e in extra:
        x.insert(int(rng.integers(0, len(x) + 1)), e)
    struct.append(x)
print("8 terms with two cancelling pairs:", round(float(acc(struct, 33).mean()), 3))
