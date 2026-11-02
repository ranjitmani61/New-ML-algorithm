"""Run one idea on the given seeds and write a JSON result.

    python3 run.py IDEA OUT.json --seeds 0 1 [--steps 4000]
"""

import argparse
import json
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

import core as C
import ideas as I

ap = argparse.ArgumentParser()
ap.add_argument("idea")
ap.add_argument("out")
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--lr", type=float, default=2e-3)
args = ap.parse_args()

idea = I.IDEAS[args.idea]
res = {"idea": args.idea, "spec": {k: v for k, v in idea.items() if k != "transform"}, "steps": args.steps,
       "lr": args.lr, "seeds": {}}
for seed in args.seeds:
    t0 = time.time()
    p0, cfg = C.init(jax.random.PRNGKey(seed), attention=idea.get("attention", "softmax"))
    kw = {k: idea[k] for k in ("loss", "rex_lambda", "pos_offset_max") if k in idea}
    p = C.train(p0, cfg, I.sampler(np.random.default_rng(seed), idea), args.steps, args.lr, seed, **kw)
    row = {"eval": C.evaluate(p, cfg, seed), "seconds": round(time.time() - t0)}
    if idea.get("wed"):
        p3 = C.train(p0, cfg, I.sampler(np.random.default_rng(seed), idea, max_terms=3), args.steps, args.lr, seed)
        for a in (0.5, 1.0):
            pe = jax.tree_util.tree_map(lambda x4, x3: x4 + a * (x4 - x3), p, p3)
            row[f"eval_alpha{a}"] = C.evaluate(pe, cfg, seed)
        row["eval_theta3"] = C.evaluate(p3, cfg, seed)
    res["seeds"][seed] = row
    pickle.dump(jax.device_get(p), open(args.out.replace(".json", f"_s{seed}.pkl"), "wb"))
    print(args.idea, seed, json.dumps(row), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
