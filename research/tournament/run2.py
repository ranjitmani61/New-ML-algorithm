"""Run one idea on a task with the generic engine.

    python3 run2.py TASK IDEA OUT.json --seeds 0 1 [--steps 4000]

S3 task: train words of length 1-6, test 7-16, chance 1/6. Only the baseline remains; every S3
idea was discarded (see LEDGER.md; code in git history).
"""

import argparse
import json
import pickle
import time

import jax
import numpy as np

import engine as E
import ideas as I
import task_s3 as S3

S3_IDEAS = {
    "baseline": dict(kind="baseline"),
}
TRAIN_MAX = {"sum": 4, "s3": 6}
TEST = {"sum": list(range(1, 9)), "s3": list(range(1, 17))}
LONG = {"sum": 8, "s3": 16}


def sampler(task, rng, idea, batch=128, p_aug=0.5):
    kmax, long = TRAIN_MAX[task.name], LONG[task.name]
    tf = idea.get("transform") or idea.get("expand")
    max_in = task.max_in(long) if tf else task.max_in(kmax)

    def draw():
        items, groups = [], []
        for _ in range(batch):
            x = task.sample(rng, int(rng.integers(1, kmax + 1)))
            groups.append(task.length(x))
            if tf is not None and rng.random() < p_aug:
                x = tf(rng, x) if task.name == "sum" else tf(rng, x, long)
            items.append(x)
        return task.batch(items, [task.answer(x) for x in items], max_in), np.array(groups, np.int32)
    return draw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("idea")
    ap.add_argument("out")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    task = E.TASKS[args.task]
    idea = (I.IDEAS if args.task == "sum" else S3_IDEAS)[args.idea]
    res = {"task": args.task, "idea": args.idea, "steps": args.steps, "lr": args.lr,
           "spec": {k: str(v) for k, v in idea.items()}, "seeds": {}}
    for seed in args.seeds:
        t0 = time.time()
        p, cfg = E.init(jax.random.PRNGKey(seed), task, attention=idea.get("attention", "softmax"))
        kw = {k: idea[k] for k in ("loss", "rex_lambda", "pos_offset_max") if k in idea}
        p = E.train(p, cfg, sampler(task, np.random.default_rng(seed), idea), args.steps, args.lr, seed, **kw)
        row = {"eval": E.evaluate(p, cfg, task, seed, TEST[args.task]), "seconds": round(time.time() - t0)}
        res["seeds"][seed] = row
        pickle.dump(jax.device_get(p), open(args.out.replace(".json", f"_s{seed}.pkl"), "wb"))
        print(args.task, args.idea, seed, json.dumps(row), flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
