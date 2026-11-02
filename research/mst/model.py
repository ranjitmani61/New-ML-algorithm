"""A small causal transformer (rotary positions, pre-norm) in plain JAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tasks import ANS, MAX_IN, PAD, SEQ, VOCAB


def init(key, d=128, layers=4, heads=4, mlp=512):
    keys = iter(jax.random.split(key, 4 + 6 * layers))
    norm = lambda k, shape, scale: jax.random.normal(k, shape) * scale
    p = {"emb": norm(next(keys), (VOCAB, d), 0.02), "out": norm(next(keys), (d, VOCAB), 0.02),
         "ln_f": jnp.ones(d), "layers": []}
    for _ in range(layers):
        p["layers"].append({
            "ln1": jnp.ones(d), "ln2": jnp.ones(d),
            "qkv": norm(next(keys), (d, 3 * d), d ** -0.5),
            "o": norm(next(keys), (d, d), (2 * layers * d) ** -0.5),
            "w1": norm(next(keys), (d, mlp), d ** -0.5),
            "w2": norm(next(keys), (mlp, d), (2 * layers * mlp) ** -0.5),
        })
    return p, {"heads": heads}


def _rms(x, g):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6) * g


def _rope(x, pos):
    half = x.shape[-1] // 2
    freq = 1.0 / (10000 ** (jnp.arange(half) / half))
    ang = pos[:, :, None] * freq[None, None, :]
    c, s = jnp.cos(ang)[:, :, None, :], jnp.sin(ang)[:, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)


def forward(p, cfg, toks):
    """Logits [B, T, VOCAB] for left-padded tokens [B, T]; positions count from the first real token."""
    B, T = toks.shape
    h = cfg["heads"]
    real = toks != PAD
    # answer slots are PAD while decoding; they still get positions
    real = real | (jnp.arange(T)[None, :] >= MAX_IN)
    pos = jnp.cumsum(real, axis=1).astype(jnp.float32) - 1.0
    x = p["emb"][toks]
    causal = jnp.tril(jnp.ones((T, T), bool))
    mask = causal[None, None] & real[:, None, None, :]
    for L in p["layers"]:
        y = _rms(x, L["ln1"])
        q, k, v = jnp.split(y @ L["qkv"], 3, -1)
        sh = lambda t: t.reshape(B, T, h, -1)
        q, k, v = _rope(sh(q), pos), _rope(sh(k), pos), sh(v)
        att = jnp.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(q.shape[-1])
        att = jnp.where(mask, att, -1e9)
        y = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(att, -1), v).reshape(B, T, -1)
        x = x + y @ L["o"]
        y = _rms(x, L["ln2"])
        x = x + jax.nn.gelu(y @ L["w1"]) @ L["w2"]
    return _rms(x, p["ln_f"]) @ p["out"]


def loss_fn(p, cfg, toks, weight):
    """Cross-entropy on the ANS answer tokens; weight [B] (0 drops an example)."""
    logits = forward(p, cfg, toks)[:, MAX_IN - 1:SEQ - 1]
    tgt = toks[:, MAX_IN:]
    lp = jax.nn.log_softmax(logits, -1)
    nll = -jnp.take_along_axis(lp, tgt[..., None], -1)[..., 0].sum(-1)
    return (nll * weight).sum() / jnp.maximum(weight.sum(), 1.0)


def decode(p, cfg, toks, key, temperature):
    """Fill the answer slots; temperature 0 is greedy. Returns tokens [B, ANS]."""
    toks = toks.at[:, MAX_IN:].set(PAD)
    for i in range(ANS):
        logits = forward(p, cfg, toks)[:, MAX_IN - 1 + i]
        if temperature == 0:
            nxt = jnp.argmax(logits, -1)
        else:
            key, sub = jax.random.split(key)
            nxt = jax.random.categorical(sub, logits / temperature)
        toks = toks.at[:, MAX_IN + i].set(nxt.astype(toks.dtype))
    return toks[:, MAX_IN:]
