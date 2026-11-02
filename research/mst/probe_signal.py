"""Before any self-training: how good is each condition's pseudo-label on unlabeled hard inputs?

Also measures the consensus trap: of the expressions the model gets wrong (greedy), how often the
majority of K samples agrees on that same wrong answer with share >= keep.
"""

import json
import pickle
import sys

import numpy as np

import lab
import mst
import tasks as T

params, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
L = lab.Learner(seed, lr=1e-3, d=64, layers=3, heads=4, mlp=256)
L.params = pickle.load(open(params, "rb"))
rng = np.random.default_rng(10_000 + seed)
result = {}
for n in (5, 6, 7, 8):
    exprs = [T.random_terms(rng, n) for _ in range(1000)]
    truth = [T.value(e) for e in exprs]
    greedy = L.answers(T.batch(exprs))
    row = {"greedy_acc": float(np.mean([g == t for g, t in zip(greedy, truth)]))}
    for cond in ("majority", "invariance", "metamorphic", "placebo"):
        labels, kept, share = mst.pseudo_label(L, rng, exprs, cond)
        row[cond] = mst.label_quality(exprs, labels, kept)
        if cond == "majority":
            wrong = [i for i, (g, t) in enumerate(zip(greedy, truth)) if g != t]
            row["trap_share_of_errors"] = float(np.mean([kept[i] and labels[i] == greedy[i] for i in wrong])) if wrong else None
    result[n] = row
    print(n, json.dumps(row), flush=True)
json.dump(result, open(out, "w"), indent=1)
