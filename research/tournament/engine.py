"""Task-generic engine (model, training, evaluation) for the tournament.

A task adapter provides: VOCAB, ANS, PAD, sample(rng, k) -> item, answer(item) -> int,
batch(items, answers, max_in) -> int32 [B, max_in + ANS], decode(tokens) -> int or None,
length(item) -> difficulty k, max_in(k) -> encoded input length bound.
The model is the same pre-norm rotary transformer as core.py with attention variants.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optax

import core as C
import task_s3 as S3


class SumTask:
    name, VOCAB, ANS, PAD = "sum", C.VOCAB, C.ANS, C.PAD
    sample = staticmethod(C.random_terms)
    answer = staticmethod(C.value)
    length = staticmethod(C.n_terms)

    @staticmethod
    def batch(items, answers=None, max_in=None):
        return C.batch(items, answers, max_in)

    @staticmethod
    def decode(t):
        return C.decode_answer(t)

    @staticmethod
    def max_in(k):
        return 2 * k + 1


class S3Task:
    name, VOCAB, ANS, PAD = "s3", S3.VOCAB, S3.ANS, S3.PAD
    sample = staticmethod(S3.random_word)
    answer = staticmethod(S3.product)
    length = staticmethod(len)

    @staticmethod
    def batch(items, answers=None, max_in=None):
        return S3.batch(items, answers, max_in)

    @staticmethod
    def decode(t):
        return S3.decode_answer(t)

    @staticmethod
    def max_in(k):
        return k + 1


TASKS = {"sum": SumTask, "s3": S3Task}


def init(key, task, d=64, layers=3, heads=4, mlp=256, attention="softmax"):
    keys = iter(jax.random.split(key, 4 + 6 * layers))
    nrm = lambda k, shape, scale: jax.random.normal(k, shape) * scale
    V = task.VOCAB
    p = {"emb": nrm(next(keys), (V, d), 0.02), "out": nrm(next(keys), (d, V), 0.02),
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
    return p, {"heads": heads, "attention": attention, "ans": task.ANS, "pad": task.PAD}


def forward(p, cfg, toks, pos_offset=None):
    B, T = toks.shape
    h, A = cfg["heads"], cfg["ans"]
    real = (toks != cfg["pad"]) | (jnp.arange(T)[None, :] >= T - A)
    pos = jnp.cumsum(real, axis=1).astype(jnp.float32) - 1.0
    if pos_offset is not None:
        pos = pos + pos_offset[:, None]
    x = p["emb"][toks]
    mask = jnp.tril(jnp.ones((T, T), bool))[None, None] & real[:, None, None, :]
    n_keys = jnp.maximum(jnp.cumsum(real, axis=1), 1).astype(jnp.float32)
    for L in p["layers"]:
        y = C._rms(x, L["ln1"])
        q, k, v = jnp.split(y @ L["qkv"], 3, -1)
        sh = lambda t: t.reshape(B, T, h, -1)
        q, k, v = C._rope(sh(q), pos), C._rope(sh(k), pos), sh(v)
        att = jnp.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(q.shape[-1])
        if cfg["attention"] == "ssmax":
            att = att * (L["s"][None, :, None, None] * jnp.log(n_keys)[:, None, :, None])
        if cfg["attention"] == "sigmoid":
            w = jax.nn.sigmoid(att + L["b"][None, :, None, None]) * mask
        else:
            w = jax.nn.softmax(jnp.where(mask, att, -1e9), -1)
        y = jnp.einsum("bhqk,bkhd->bqhd", w, v).reshape(B, T, -1)
        x = x + y @ L["o"]
        y = C._rms(x, L["ln2"])
        x = x + jax.nn.gelu(y @ L["w1"]) @ L["w2"]
    return C._rms(x, p["ln_f"]) @ p["out"]


def per_example_nll(p, cfg, toks, pos_offset=None):
    T, A = toks.shape[1], cfg["ans"]
    logits = forward(p, cfg, toks, pos_offset)[:, T - A - 1:T - 1]
    lp = jax.nn.log_softmax(logits, -1)
    return -jnp.take_along_axis(lp, toks[:, T - A:, None], -1)[..., 0].sum(-1)


def greedy(p, cfg, toks):
    T, A = toks.shape[1], cfg["ans"]
    toks = toks.at[:, T - A:].set(cfg["pad"])
    for i in range(A):
        nxt = jnp.argmax(forward(p, cfg, toks)[:, T - A - 1 + i], -1)
        toks = toks.at[:, T - A + i].set(nxt.astype(toks.dtype))
    return toks[:, T - A:]


def make_step(cfg, lr, steps, warmup=200, loss="mean", rex_lambda=0.0, pos_offset_max=0, n_groups=17):
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, steps, lr * 0.1)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))

    def total_loss(p, toks, group, key):
        off = None
        if pos_offset_max:
            off = jax.random.randint(key, (toks.shape[0],), 0, pos_offset_max + 1).astype(jnp.float32)
        nll = per_example_nll(p, cfg, toks, off)
        if loss == "rex":
            onehot = jax.nn.one_hot(group, n_groups)
            cnt = onehot.sum(0)
            risk = (onehot * nll[:, None]).sum(0) / jnp.maximum(cnt, 1)
            present = cnt > 0
            mean_r = (risk * present).sum() / present.sum()
            return nll.mean() + rex_lambda * (((risk - mean_r) ** 2) * present).sum() / present.sum()
        return nll.mean()

    @jax.jit
    def step(p, state, toks, group, key):
        l, g = jax.value_and_grad(total_loss)(p, toks, group, key)
        upd, state = opt.update(g, state, p)
        return optax.apply_updates(p, upd), state, l

    return opt, step


def train(p, cfg, sampler, steps, lr=2e-3, seed=0, **kw):
    opt, step = make_step(cfg, lr, steps, **kw)
    state = opt.init(p)
    key = jax.random.PRNGKey(seed + 123)
    for _ in range(steps):
        toks, group = sampler()
        key, sub = jax.random.split(key)
        p, state, _ = step(p, state, jnp.asarray(toks), jnp.asarray(group), sub)
    return p


_cache = {}


def evaluate(p, cfg, task, seed, ns, per_n=1000):
    rng = np.random.default_rng(900_000 + seed)
    key = (task.name, cfg["attention"], cfg["heads"])
    if key not in _cache:
        _cache[key] = jax.jit(lambda pp, x: greedy(pp, cfg, x))
    g = _cache[key]
    width = task.max_in(max(ns))
    acc = {}
    for n in ns:
        items = [task.sample(rng, n) for _ in range(per_n)]
        out = np.asarray(g(p, jnp.asarray(task.batch(items, max_in=width))))
        acc[n] = float(np.mean([task.decode(o) == task.answer(x) for o, x in zip(out, items)]))
    return acc
