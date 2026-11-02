"""One seed of the pilot: shared short-only pretraining, then each label-free condition.

    python3 run.py SEED PRETRAINED_PARAMS OUT.json [--rounds 3 --unlabeled 3000 --steps 800]

All conditions start from the same pretrained parameters and a fresh optimizer, see the same
unlabeled expressions in each round (same generator seed), sample the same number of answers and
train the same number of steps. Evaluation uses fresh expressions from a separate generator seed.
"""

import argparse
import json
import pickle
import time

import numpy as np

import lab
import mst
import tasks as T

ap = argparse.ArgumentParser()
ap.add_argument("seed", type=int)
ap.add_argument("params")
ap.add_argument("out")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--unlabeled", type=int, default=3000)
ap.add_argument("--steps", type=int, default=800)
ap.add_argument("--conditions", nargs="+", default=list(mst.CONDITIONS))
ap.add_argument("--hard", nargs="+", type=int, default=[5, 6, 7, 8])
ap.add_argument("--eval", nargs="+", type=int, default=[4, 5, 6, 7, 8])
args = ap.parse_args()

pre = pickle.load(open(args.params, "rb"))
results = {"seed": args.seed, "args": vars(args), "conditions": {}}
for cond in args.conditions:
    t0 = time.time()
    L = lab.Learner(args.seed, lr=1e-3, d=64, layers=3, heads=4, mlp=256)
    L.params = pre
    L.state = L.opt.init(pre)
    L.key = __import__("jax").random.PRNGKey(77_000 + args.seed)
    data_rng = np.random.default_rng(1_000 + args.seed)  # same unlabeled inputs for every condition
    rel_rng = np.random.default_rng(2_000 + args.seed)
    train_rng = np.random.default_rng(3_000 + args.seed)
    rows = [{"round": 0, "eval": lab.evaluate(L, np.random.default_rng(50_000 + args.seed), args.eval, 500)}]
    for r in range(args.rounds):
        exprs = [T.random_terms(data_rng, int(data_rng.choice(args.hard))) for _ in range(args.unlabeled)]
        labels, kept, share = mst.pseudo_label(L, rel_rng, exprs, cond)
        quality = mst.label_quality(exprs, labels, kept)
        pool = [(e, l) for e, l, k in zip(exprs, labels, kept) if k]
        L.train(mst.mixed_sampler(train_rng, [1, 2, 3, 4], pool), args.steps)
        ev = lab.evaluate(L, np.random.default_rng(50_000 + args.seed + r + 1), args.eval, 500)
        rows.append({"round": r + 1, "labels": quality, "eval": ev})
        print(cond, r + 1, json.dumps(quality), json.dumps(ev), f"{time.time() - t0:.0f}s", flush=True)
    results["conditions"][cond] = rows
    json.dump(results, open(args.out, "w"), indent=1)
