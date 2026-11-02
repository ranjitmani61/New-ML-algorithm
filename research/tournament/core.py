"""Tournament core: length-extrapolation testbed, model variants, training and evaluation.

Task: signed sums of single-digit terms ("+3-7+5="), answer = sign + two reversed digits.
Train on 1-4 terms, test on 5-8 terms. An input item is a term (sign, value) or None (a filler
token used by one of the reference augmentations).

Model: small pre-norm causal transformer with rotary positions; attention variants:
  softmax  standard
  ssmax    scalable softmax: logits * s_h * log(number of visible keys)   (known, 2501.19399)
  sigmoid  unnormalised sigmoid attention with bias b_h = -log(10) at init (known, 2409.04431)
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optax

PAD, PLUS, MINUS, EQ, FILL = 0, 11, 12, 13, 14
VOCAB = 15
ANS = 3
TERM_MAX = 9


def random_terms(rng, n):
    s = rng.choice([1, -1], size=n)
    v = rng.integers(0, TERM_MAX + 1, size=n)
    return [(int(a), int(b)) for a, b in zip(s, v)]


def value(items):
    return sum(t[0] * t[1] for t in items if t is not None)


def n_terms(items):
    return sum(t is not None for t in items)


def encode_input(items):
    toks = []
    for t in items:
        if t is None:
            toks.append(FILL)
        else:
            toks += [PLUS if t[0] > 0 else MINUS, 1 + t[1]]
    return toks + [EQ]


def encode_answer(v):
    if abs(v) >= 100:
        raise ValueError(v)
    return [PLUS if v >= 0 else MINUS, 1 + abs(v) % 10, 1 + abs(v) // 10]


def decode_answer(toks):
    toks = [int(t) for t in toks]
    if toks[0] not in (PLUS, MINUS) or any(not 1 <= t <= 10 for t in toks[1:]):
        return None
    v = (toks[1] - 1) + 10 * (toks[2] - 1)
    return v if toks[0] == PLUS else -v


def batch(items_list, answers=None, max_in=None):
    enc = [encode_input(x) for x in items_list]
    max_in = max_in or max(len(e) for e in enc)
    out = np.zeros((len(enc), max_in + ANS), dtype=np.int32)
    for i, e in enumerate(enc):
        out[i, max_in - len(e):max_in] = e
        if answers is not None:
            out[i, max_in:] = encode_answer(answers[i])
    return out


# ---------------------------------------------------------------- model


def init(key, d=64, layers=3, heads=4, mlp=256, attention="softmax"):
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
        if attention == "ssmax":
            L["s"] = jnp.full((heads,), 1.0 / np.log(10.0))
        if attention == "sigmoid":
            L["b"] = jnp.full((heads,), -np.log(10.0))
        p["layers"].append(L)
    return p, {"heads": heads, "attention": attention}


def _rms(x, g):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6) * g


def _rope(x, pos):
    half = x.shape[-1] // 2
    freq = 1.0 / (10000 ** (jnp.arange(half) / half))
    ang = pos[:, :, None] * freq[None, None, :]
    c, s = jnp.cos(ang)[:, :, None, :], jnp.sin(ang)[:, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)


def forward(p, cfg, toks, pos_offset=None):
    B, T = toks.shape
    h = cfg["heads"]
    real = (toks != PAD) | (jnp.arange(T)[None, :] >= T - ANS)
    pos = jnp.cumsum(real, axis=1).astype(jnp.float32) - 1.0
    if pos_offset is not None:
        pos = pos + pos_offset[:, None]
    x = p["emb"][toks]
    mask = jnp.tril(jnp.ones((T, T), bool))[None, None] & real[:, None, None, :]
    n_keys = jnp.maximum(jnp.cumsum(real, axis=1), 1).astype(jnp.float32)  # visible real keys per query
    for L in p["layers"]:
        y = _rms(x, L["ln1"])
        q, k, v = jnp.split(y @ L["qkv"], 3, -1)
        sh = lambda t: t.reshape(B, T, h, -1)
        q, k, v = _rope(sh(q), pos), _rope(sh(k), pos), sh(v)
        att = jnp.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(q.shape[-1])
        if cfg["attention"] == "ssmax":
            att = att * (L["s"][None, :, None, None] * jnp.log(n_keys)[:, None, :, None])
        if cfg["attention"] == "sigmoid":
            w = jax.nn.sigmoid(att + L["b"][None, :, None, None]) * mask
        else:
            w = jax.nn.softmax(jnp.where(mask, att, -1e9), -1)
        y = jnp.einsum("bhqk,bkhd->bqhd", w, v).reshape(B, T, -1)
        x = x + y @ L["o"]
        y = _rms(x, L["ln2"])
        x = x + jax.nn.gelu(y @ L["w1"]) @ L["w2"]
    return _rms(x, p["ln_f"]) @ p["out"]


def per_example_nll(p, cfg, toks, pos_offset=None):
    T = toks.shape[1]
    logits = forward(p, cfg, toks, pos_offset)[:, T - ANS - 1:T - 1]
    lp = jax.nn.log_softmax(logits, -1)
    return -jnp.take_along_axis(lp, toks[:, T - ANS:, None], -1)[..., 0].sum(-1)


def greedy(p, cfg, toks):
    T = toks.shape[1]
    toks = toks.at[:, T - ANS:].set(PAD)
    for i in range(ANS):
        nxt = jnp.argmax(forward(p, cfg, toks)[:, T - ANS - 1 + i], -1)
        toks = toks.at[:, T - ANS + i].set(nxt.astype(toks.dtype))
    return toks[:, T - ANS:]


# ---------------------------------------------------------------- training and evaluation


def make_trainer(cfg, lr, steps, warmup=200, loss="mean", rex_lambda=0.0, pos_offset_max=0):
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, steps, lr * 0.1)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))

    def total_loss(p, toks, group, key):
        off = None
        if pos_offset_max:
            off = jax.random.randint(key, (toks.shape[0],), 0, pos_offset_max + 1).astype(jnp.float32)
        nll = per_example_nll(p, cfg, toks, off)
        if loss == "rex":  # risk extrapolation across lengths: mean + lambda * variance of per-length risks
            onehot = jax.nn.one_hot(group, 9)
            cnt = onehot.sum(0)
            risk = (onehot * nll[:, None]).sum(0) / jnp.maximum(cnt, 1)
            present = cnt > 0
            mean_r = (risk * present).sum() / present.sum()
            var_r = (((risk - mean_r) ** 2) * present).sum() / present.sum()
            return nll.mean() + rex_lambda * var_r
        return nll.mean()

    @jax.jit
    def step(p, state, toks, group, key):
        l, g = jax.value_and_grad(total_loss)(p, toks, group, key)
        upd, state = opt.update(g, state, p)
        return optax.apply_updates(p, upd), state, l

    return opt, step


def train(p, cfg, sampler, steps, lr=2e-3, seed=0, **kw):
    opt, step = make_trainer(cfg, lr, steps, **kw)
    state = opt.init(p)
    key = jax.random.PRNGKey(seed + 123)
    for _ in range(steps):
        toks, group = sampler()
        key, sub = jax.random.split(key)
        p, state, _ = step(p, state, jnp.asarray(toks), jnp.asarray(group), sub)
    return p


_greedy_cache = {}


def evaluate(p, cfg, seed, ns=range(1, 9), per_n=1000):
    rng = np.random.default_rng(900_000 + seed)
    key = (cfg["heads"], cfg["attention"])
    if key not in _greedy_cache:
        _greedy_cache[key] = jax.jit(lambda pp, x: greedy(pp, cfg, x))
    g = _greedy_cache[key]
    acc = {}
    for n in ns:
        items = [random_terms(rng, n) for _ in range(per_n)]
        out = np.asarray(g(p, jnp.asarray(batch(items, max_in=2 * 8 + 1))))
        acc[n] = float(np.mean([decode_answer(o) == value(x) for o, x in zip(out, items)]))
    return acc
