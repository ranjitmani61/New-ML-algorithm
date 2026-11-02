"""Final test of inverse-pair insertion, as registered in SPEC.md (single file, runs on CPU or TPU).

    python3 final.py --out results.json [--conditions ...] [--seeds ...] [--smoke]

Task: signed sums of single-digit terms; answer = sign + three reversed digits. Labeled training
problems have 1-8 terms. Expansions (half of each batch) lengthen them to at most 32 items without
changing the label: cancelling pairs (+a, -a), zero terms, or neutral filler tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

PAD, PLUS, MINUS, EQ, FILL = 0, 11, 12, 13, 14
VOCAB = 15
DIGITS = 3
ANS = 1 + DIGITS
LABEL_MAX = 8
EXPAND_MAX = 32
TEST_LENGTHS = (8, 16, 24, 32, 48)
CONDITIONS = ("baseline", "sigmoid", "fill", "zeros", "pairs", "oracle")
SEEDS = (100, 101, 102, 103, 104)

# ---------------------------------------------------------------- task


def random_terms(rng, n):
    s = rng.choice([1, -1], size=n)
    v = rng.integers(0, 10, size=n)
    return [(int(a), int(b)) for a, b in zip(s, v)]


def value(items):
    return sum(t[0] * t[1] for t in items if t is not None)


def encode_input(items):
    toks = []
    for t in items:
        if t is None:
            toks.append(FILL)
        else:
            toks += [PLUS if t[0] > 0 else MINUS, 1 + t[1]]
    return toks + [EQ]


def encode_answer(v):
    a = abs(v)
    if a >= 10 ** DIGITS:
        raise ValueError(v)
    return [PLUS if v >= 0 else MINUS] + [1 + (a // 10 ** i) % 10 for i in range(DIGITS)]


def decode_answer(toks):
    toks = [int(t) for t in toks]
    if toks[0] not in (PLUS, MINUS) or any(not 1 <= t <= 10 for t in toks[1:]):
        return None
    v = sum((t - 1) * 10 ** i for i, t in enumerate(toks[1:]))
    return v if toks[0] == PLUS else -v


def batch(items_list, answers, max_in):
    out = np.zeros((len(items_list), max_in + ANS), dtype=np.int32)
    for i, items in enumerate(items_list):
        e = encode_input(items)
        out[i, max_in - len(e):max_in] = e
        if answers is not None:
            out[i, max_in:] = encode_answer(answers[i])
    return out


def _insert(rng, items, extra):
    out = list(items)
    for x in extra:
        out.insert(int(rng.integers(0, len(out) + 1)), x)
    return out


def expand_pairs(rng, items):
    k = int(rng.integers(0, (EXPAND_MAX - len(items)) // 2 + 1))
    extra = []
    for _ in range(k):
        a = int(rng.integers(1, 10))
        extra += [(1, a), (-1, a)]
    return _insert(rng, items, extra)


def expand_zeros(rng, items):
    m = int(rng.integers(0, EXPAND_MAX - len(items) + 1))
    return _insert(rng, items, [(int(rng.choice([1, -1])), 0) for _ in range(m)])


def expand_fill(rng, items):
    m = int(rng.integers(0, EXPAND_MAX - len(items) + 1))
    return _insert(rng, items, [None] * m)


EXPAND = {"pairs": expand_pairs, "zeros": expand_zeros, "fill": expand_fill}


def sampler(rng, condition, batch_size):
    tf = EXPAND.get(condition)
    n_max = EXPAND_MAX if condition == "oracle" else LABEL_MAX
    max_in = 2 * EXPAND_MAX + 1 if (tf or condition == "oracle") else 2 * LABEL_MAX + 1

    def draw():
        items = []
        for _ in range(batch_size):
            x = random_terms(rng, int(rng.integers(1, n_max + 1)))
            if tf is not None and rng.random() < 0.5:
                x = tf(rng, x)
            items.append(x)
        return batch(items, [value(x) for x in items], max_in)
    return draw


def self_check():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        x = random_terms(rng, int(rng.integers(1, LABEL_MAX + 1)))
        for f in EXPAND.values():
            y = f(rng, x)
            assert value(y) == value(x) and len(y) <= EXPAND_MAX
    for v in (0, 7, -7, 123, -432, 999):
        assert decode_answer(encode_answer(v)) == v

# ---------------------------------------------------------------- model


def init(key, d, layers, heads, mlp, attention):
    keys = iter(jax.random.split(key, 4 + 6 * layers))
    nrm = lambda k, shape, scale: jax.random.normal(k, shape) * scale
    p = {"emb": nrm(next(keys), (VOCAB, d), 0.02), "out": nrm(next(keys), (d, VOCAB), 0.02),
         "ln_f": jnp.ones(d), "layers": []}
    for _ in range(layers):
        L = {"ln1": jnp.ones(d), "ln2": jnp.ones(d),
             "qkv": nrm(next(keys), (d, 3 * d), d ** -0.5),
             "o": nrm(next(keys), (d, d), (2 * layers * d) ** -0.5),
             "w1": nrm(next(keys), (d, mlp), d ** -0.5),
             "w2": nrm(next(keys), (mlp, d), (2 * layers * mlp) ** -0.5)}
        if attention == "sigmoid":
            L["b"] = jnp.full((heads,), -np.log(10.0))
        p["layers"].append(L)
    return p


def _rms(x, g):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6) * g


def _rope(x, pos):
    half = x.shape[-1] // 2
    freq = 1.0 / (10000 ** (jnp.arange(half) / half))
    ang = pos[:, :, None] * freq[None, None, :]
    c, s = jnp.cos(ang)[:, :, None, :], jnp.sin(ang)[:, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)


def forward(p, toks, heads, attention):
    B, T = toks.shape
    real = (toks != PAD) | (jnp.arange(T)[None, :] >= T - ANS)
    pos = jnp.cumsum(real, axis=1).astype(jnp.float32) - 1.0
    x = p["emb"][toks]
    mask = jnp.tril(jnp.ones((T, T), bool))[None, None] & real[:, None, None, :]
    for L in p["layers"]:
        y = _rms(x, L["ln1"])
        q, k, v = jnp.split(y @ L["qkv"], 3, -1)
        sh = lambda t: t.reshape(B, T, heads, -1)
        q, k, v = _rope(sh(q), pos), _rope(sh(k), pos), sh(v)
        att = jnp.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(q.shape[-1])
        if attention == "sigmoid":
            w = jax.nn.sigmoid(att + L["b"][None, :, None, None]) * mask
        else:
            w = jax.nn.softmax(jnp.where(mask, att, -1e9), -1)
        y = jnp.einsum("bhqk,bkhd->bqhd", w, v).reshape(B, T, -1)
        x = x + y @ L["o"]
        y = _rms(x, L["ln2"])
        x = x + jax.nn.gelu(y @ L["w1"]) @ L["w2"]
    return _rms(x, p["ln_f"]) @ p["out"]


def run(condition, seed, steps, batch_size, lr, warmup, d=128, layers=4, heads=4, mlp=512, per_len=1000):
    attention = "sigmoid" if condition == "sigmoid" else "softmax"
    p = init(jax.random.PRNGKey(seed), d, layers, heads, mlp, attention)
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, steps, lr * 0.1)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    state = opt.init(p)

    def loss_fn(p, toks):
        T = toks.shape[1]
        logits = forward(p, toks, heads, attention)[:, T - ANS - 1:T - 1]
        lp = jax.nn.log_softmax(logits, -1)
        return -jnp.take_along_axis(lp, toks[:, T - ANS:, None], -1)[..., 0].sum(-1).mean()

    @jax.jit
    def step(p, state, toks):
        l, g = jax.value_and_grad(loss_fn)(p, toks)
        upd, state = opt.update(g, state, p)
        return optax.apply_updates(p, upd), state, l

    @jax.jit
    def greedy(p, toks):
        T = toks.shape[1]
        toks = toks.at[:, T - ANS:].set(PAD)
        for i in range(ANS):
            nxt = jnp.argmax(forward(p, toks, heads, attention)[:, T - ANS - 1 + i], -1)
            toks = toks.at[:, T - ANS + i].set(nxt.astype(toks.dtype))
        return toks[:, T - ANS:]

    draw = sampler(np.random.default_rng(seed), condition, batch_size)
    t0 = time.time()
    for i in range(steps):
        p, state, l = step(p, state, jnp.asarray(draw()))
        if (i + 1) % max(steps // 5, 1) == 0:
            print(json.dumps({"condition": condition, "seed": seed, "step": i + 1, "loss": round(float(l), 4),
                              "elapsed": round(time.time() - t0)}), flush=True)
    train_s = time.time() - t0
    rng = np.random.default_rng(900_000 + seed)
    width = 2 * max(TEST_LENGTHS) + 1
    acc, by_sum = {}, {}
    for n in TEST_LENGTHS:
        items = [random_terms(rng, n) for _ in range(per_len)]
        out = np.asarray(greedy(p, jnp.asarray(batch(items, None, width))))
        ok = np.array([decode_answer(o) == value(x) for o, x in zip(out, items)])
        acc[n] = float(ok.mean())
        if n == 24:
            v = np.abs([value(x) for x in items])
            for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 40), (40, 217)):
                m = (v >= lo) & (v < hi)
                by_sum[f"{lo}-{hi - 1}"] = [float(ok[m].mean()) if m.any() else None, int(m.sum())]
    return {"eval": acc, "by_sum_24": by_sum, "train_seconds": round(train_s), "total_seconds": round(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the pipeline (not a result)")
    args = ap.parse_args()
    self_check()
    steps, per_len = (120, 100) if args.smoke else (20_000, 1000)
    warmup = 20 if args.smoke else 500
    size = dict(d=32, layers=2, heads=4, mlp=64) if args.smoke else dict(d=128, layers=4, heads=4, mlp=512)
    bs = 64 if args.smoke else 256
    info = {"devices": [str(d) for d in jax.devices()], "jax": jax.__version__, "smoke": args.smoke,
            "steps": steps, "spec": "research/final/SPEC.md"}
    print(json.dumps(info), flush=True)
    results = {"info": info, "runs": {}}
    if os.path.exists(args.out):
        results = json.load(open(args.out))
    for cond in args.conditions:
        for seed in args.seeds:
            key = f"{cond}/{seed}"
            if key in results["runs"]:
                continue
            r = run(cond, seed, steps, bs, 1e-3, warmup, per_len=per_len, **size)
            results["runs"][key] = r
            print(json.dumps({"done": key, **r}), flush=True)
            json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
