"""Run one condition on the three-person toy task (task_multi.py) with the generic engine.

    python3 run3.py COND OUT.json --seeds 0 1 [--steps 4000]

Train on 1-4 events (oracle: 1-16); expansions apply with probability 0.5, up to 16 events.
Test: 1,000 fresh problems at each of 1-4, 8, 12 and 16 events; at 12 events also accuracy by the
number of events that touched the winner (the failure found in research/llm/diagnose_multi.py).
"""
import argparse
import json
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

import engine as E
import mine_multi_toy as MM
import task_multi as T


class Task:
    name, VOCAB, ANS, PAD = "multi", T.VOCAB, T.ANS, T.PAD


CONDS = {
    "baseline": dict(),
    "pairs": dict(expand=T.expand_pairs),
    "split": dict(expand=T.expand_split),
    "split_pairs": dict(expand=lambda rng, x, m: T.expand_pairs(rng, T.expand_split(rng, x, m), m)),
    "oracle": dict(train_max=16),
    "mined": dict(mined=True),  # ledger #33: certified edits, labels inherited (needs --m0)
}
LONG, MAX_IN = 16, 9 + 3 * 16 + 1


def sampler(rng, cond, batch=128, p_aug=0.5):
    kmax, tf = cond.get("train_max", 4), cond.get("expand")

    def draw():
        items, groups = [], []
        for _ in range(batch):
            x = T.sample(rng, int(rng.integers(1, kmax + 1)))
            if tf is not None and rng.random() < p_aug:
                x = tf(rng, x, LONG)
            items.append(x)
            groups.append(T.length(x))
        width = MAX_IN if (tf is not None or kmax > 4) else 9 + 3 * kmax + 1  # padding is masked
        return T.batch(items, [T.answer(x) for x in items], width), np.array(groups, np.int32)
    return draw


def sampler_inherit(rng, tf, batch=128, p_aug=0.5):
    """For mined: an edited problem keeps the short original's label; it is never re-solved."""
    def draw():
        items, labels, groups = [], [], []
        for _ in range(batch):
            x = T.sample(rng, int(rng.integers(1, 5)))
            y = T.answer(x)
            if rng.random() < p_aug:
                x = tf(rng, x)
            items.append(x)
            labels.append(y)
            groups.append(T.length(x))
        return T.batch(items, labels, MAX_IN), np.array(groups, np.int32)
    return draw


def predictor(p, cfg):
    g = jax.jit(lambda pp, x: E.greedy(pp, cfg, x))
    return lambda items: [T.decode_answer(o) for o in np.asarray(g(p, jnp.asarray(T.batch(items, max_in=MAX_IN))))]


def run_mined(seed, args, cfg):
    """M0 = the baseline model of this seed (--m0, with {seed}); gate; certify; train fresh."""
    info = {}
    p0 = pickle.load(open(args.m0.format(seed=seed), "rb"))
    pred = predictor(p0, cfg)
    grng = np.random.default_rng(800_000 + seed)
    gate_items = [T.sample(grng, k) for k in (1, 2, 3, 4) for _ in range(500)]
    info["m0_in_dist"] = float(np.mean([a == T.answer(x) for a, x in zip(pred(gate_items), gate_items)]))
    if info["m0_in_dist"] < args.gate:
        info["stopped"] = f"gate failed: M0 below {args.gate} on 1-4 events"
        return None, info
    t = time.time()
    ref, rows = MM.certify(pred, np.random.default_rng(700_000 + seed))
    info.update(ref=ref, certification=MM.summary(rows), t_certify=round(time.time() - t))
    cert = [MM.as_edit(r["edit"]) for r in rows if r["certified"]]
    print(json.dumps({"seed": seed, "ref": ref, "certification": info["certification"]}), flush=True)
    if not cert:
        info["stopped"] = "no edit certified"
        return None, info
    tf = MM.lengthen(cert, LONG)
    srng = np.random.default_rng(10_000 + seed)  # evaluation only: label error and validity of the data
    probe = []
    for _ in range(2000):
        x = T.sample(srng, int(srng.integers(1, 5)))
        z = tf(srng, x) if srng.random() < 0.5 else x
        probe.append((T.answer(z) == T.answer(x), T.valid(z), T.length(z)))
    info["train_probe"] = {"label_true": float(np.mean([a for a, _, _ in probe])),
                           "valid": float(np.mean([b for _, b, _ in probe])),
                           "events_mean": float(np.mean([c for _, _, c in probe]))}
    return sampler_inherit(np.random.default_rng(seed), tf), info


def evaluate(p, cfg, seed, per_n=1000):
    rng = np.random.default_rng(900_000 + seed)
    g = jax.jit(lambda pp, x: E.greedy(pp, cfg, x))
    acc, by_w = {}, {}
    for n in (1, 2, 3, 4, 8, 12, 16):
        items = [T.sample(rng, n) for _ in range(per_n)]
        out = np.asarray(g(p, jnp.asarray(T.batch(items, max_in=MAX_IN))))
        ok = [T.decode_answer(o) == T.answer(x) for o, x in zip(out, items)]
        acc[n] = float(np.mean(ok))
        if n == 12:
            for x, c in zip(items, ok):
                f = T.finals(x)
                w = f.index(max(f))
                k = sum(1 for e in x[1] if e[0] == w)
                b = "0-2" if k <= 2 else "3-4" if k <= 4 else "5-6" if k <= 6 else "7+"
                by_w.setdefault(b, []).append(c)
    return acc, {b: [float(np.mean(v)), len(v)] for b, v in sorted(by_w.items())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cond")
    ap.add_argument("out")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d", type=int, default=64)  # model size; the defaults are batch 4's
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mlp", type=int, default=256)
    ap.add_argument("--m0", help="mined only: the baseline pickle, with {seed}")
    ap.add_argument("--gate", type=float, default=0.90, help="mined only; lower it for smoke tests only")
    args = ap.parse_args()
    cond = CONDS[args.cond]
    res = {"task": "multi", "cond": args.cond, "steps": args.steps, "lr": args.lr,
           "model": {"d": args.d, "layers": args.layers, "heads": args.heads, "mlp": args.mlp}, "seeds": {}}
    for seed in args.seeds:
        t0 = time.time()
        p, cfg = E.init(jax.random.PRNGKey(seed), Task, d=args.d, layers=args.layers, heads=args.heads, mlp=args.mlp)
        info = {}
        if cond.get("mined"):
            draw, info = run_mined(seed, args, cfg)
            if draw is None:
                res["seeds"][seed] = {**info, "seconds": round(time.time() - t0)}
                print(args.cond, seed, json.dumps(res["seeds"][seed]), flush=True)
                json.dump(res, open(args.out, "w"), indent=1)
                continue
        else:
            draw = sampler(np.random.default_rng(seed), cond)
        p = E.train(p, cfg, draw, args.steps, args.lr, seed)
        acc, by_w = evaluate(p, cfg, seed)
        res["seeds"][seed] = {**info, "eval": acc, "by_winner_events_12": by_w, "seconds": round(time.time() - t0)}
        pickle.dump(jax.device_get(p), open(args.out.replace(".json", f"_s{seed}.pkl"), "wb"))
        print(args.cond, seed, json.dumps(res["seeds"][seed]), flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
