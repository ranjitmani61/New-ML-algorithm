"""Batch 5 runner (ledger #33-#36): certified rewrite mining.

    python3 run4.py sum baseline OUT.json --seed 0        # M0 for mining; gate: 1-4 terms >= 90%
    python3 run4.py sum npia OUT.json --seed 0
    python3 run4.py sum mined OUT.json --seed 0 --m0 M0.pkl
    python3 run4.py s3 certify OUT.json --seed 0          # trains its own M0 on 1-6 elements

baseline and npia use run2.sampler (identical to batch 1). mined certifies edits with M0 and
trains a fresh model whose edited problems inherit the short labels (mine.sampler).
"""
import argparse
import json
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

import engine as E
import ideas as I
import mine as M
import run2 as R

ap = argparse.ArgumentParser()
ap.add_argument("task", choices=("sum", "s3"))
ap.add_argument("cond", choices=("baseline", "npia", "mined", "certify"))
ap.add_argument("out")
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--m0")
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--lr", type=float, default=2e-3)
a = ap.parse_args()

task = E.TASKS[a.task]
kmax, long = R.TRAIN_MAX[a.task], R.LONG[a.task]
res = {"task": a.task, "cond": a.cond, "seed": a.seed, "steps": a.steps, "lr": a.lr}
t0 = time.time()


def predictor(p, cfg, width):
    g = jax.jit(lambda pp, x: E.greedy(pp, cfg, x))
    return lambda items: [task.decode(o) for o in np.asarray(g(p, jnp.asarray(task.batch(items, max_in=width))))]


def train_plain(idea):
    p, cfg = E.init(jax.random.PRNGKey(a.seed), task)
    return E.train(p, cfg, R.sampler(task, np.random.default_rng(a.seed), idea), a.steps, a.lr, a.seed), cfg


def finish(p, cfg):
    res["eval"] = E.evaluate(p, cfg, task, a.seed, R.TEST[a.task])
    if a.task == "sum":
        res["eval_long"] = E.evaluate(p, cfg, task, a.seed, [12, 16])
    res["in_dist"] = float(np.mean([res["eval"][k] for k in range(1, kmax + 1)]))
    res["seconds"] = round(time.time() - t0)
    pickle.dump(jax.device_get(p), open(a.out.replace(".json", ".pkl"), "wb"))


if a.cond in ("baseline", "npia"):
    assert a.task == "sum"
    p, cfg = train_plain(I.IDEAS[a.cond])
    finish(p, cfg)
    if a.cond == "baseline":
        res["gate_passed"] = res["in_dist"] >= 0.90
elif a.cond == "mined":
    assert a.task == "sum" and a.m0
    _, cfg = E.init(jax.random.PRNGKey(a.seed), task)
    p0 = pickle.load(open(a.m0, "rb"))
    gate = E.evaluate(p0, cfg, task, a.seed, list(range(1, kmax + 1)))
    res["m0_in_dist"] = float(np.mean(list(gate.values())))
    if res["m0_in_dist"] < 0.90:
        res["stopped"] = "gate failed: M0 below 90% on 1-4 terms"
    else:
        ref, rows = M.certify(task, predictor(p0, cfg, task.max_in(4)), M.edit_space(task),
                              np.random.default_rng(700_000 + a.seed))
        res["ref"], res["certification"], res["rows"] = ref, M.summary(rows), rows
        cert = [M.as_edit(r["edit"]) for r in rows if r["certified"]]
        res["n_certified"], res["t_certify"] = len(cert), round(time.time() - t0)
        print(json.dumps({"ref": ref, "certification": res["certification"]}), flush=True)
        p, cfg = E.init(jax.random.PRNGKey(a.seed), task)
        draw = M.sampler(task, np.random.default_rng(a.seed), M.lengthen(cert, long), kmax, task.max_in(long))
        p = E.train(p, cfg, draw, a.steps, a.lr, a.seed)
        finish(p, cfg)
else:  # certify (S3): M0 on 1-6 elements, certification only
    assert a.task == "s3"
    p0, cfg = train_plain(R.S3_IDEAS["baseline"])
    res["m0_eval"] = E.evaluate(p0, cfg, task, a.seed, list(range(1, 7)))
    ref, rows = M.certify(task, predictor(p0, cfg, task.max_in(4)), M.edit_space(task),
                          np.random.default_rng(700_000 + a.seed))
    res["ref"], res["certification"], res["rows"] = ref, M.summary(rows), rows
    res["seconds"] = round(time.time() - t0)

print(a.task, a.cond, a.seed, json.dumps({k: v for k, v in res.items() if k != "rows"}), flush=True)
json.dump(res, open(a.out, "w"), indent=1)
