"""Training, evaluation and label-free self-training for the metamorphic self-training pilot."""

from __future__ import annotations

import time
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import optax

import model as M
import tasks as T


class Learner:
    def __init__(self, seed, lr=3e-4, d=128, layers=4, heads=4, mlp=512):
        self.params, self.cfg = M.init(jax.random.PRNGKey(seed), d, layers, heads, mlp)
        self.opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=0.01))
        self.state = self.opt.init(self.params)
        cfg, opt = self.cfg, self.opt

        @jax.jit
        def step(params, state, toks, w):
            loss, g = jax.value_and_grad(M.loss_fn)(params, cfg, toks, w)
            upd, state = opt.update(g, state, params)
            return optax.apply_updates(params, upd), state, loss

        self._step = step
        self._greedy = jax.jit(lambda p, x: M.decode(p, cfg, x, jax.random.PRNGKey(0), 0.0))
        self._sample = jax.jit(lambda p, x, k: M.decode(p, cfg, x, k, 1.0))
        self.key = jax.random.PRNGKey(seed + 10_000)

    def train(self, sampler, steps, log_every=0):
        losses = []
        for i in range(steps):
            toks, w = sampler()
            self.params, self.state, loss = self._step(self.params, self.state, jnp.asarray(toks), jnp.asarray(w))
            losses.append(float(loss)) if log_every and (i + 1) % log_every == 0 else None
        return losses

    def answers(self, toks, sample=False, chunk=2048):
        out = []
        for i in range(0, len(toks), chunk):
            x = jnp.asarray(toks[i:i + chunk])
            if sample:
                self.key, k = jax.random.split(self.key)
                y = self._sample(self.params, x, k)
            else:
                y = self._greedy(self.params, x)
            out.extend(T.decode_answer(r) for r in np.asarray(y))
        return out


def labeled_sampler(rng, ns, batch=256):
    def draw():
        terms = [T.random_terms(rng, int(rng.choice(ns))) for _ in range(batch)]
        return T.batch(terms, [T.value(t) for t in terms]), np.ones(batch, np.float32)
    return draw


def evaluate(learner, rng, ns, per_n=1000):
    acc = {}
    for n in ns:
        terms = [T.random_terms(rng, n) for _ in range(per_n)]
        pred = learner.answers(T.batch(terms))
        acc[n] = float(np.mean([p == T.value(t) for p, t in zip(pred, terms)]))
    return acc


if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    L = Learner(seed)
    rng = np.random.default_rng(seed)
    t = time.time()
    L.train(labeled_sampler(rng, [1, 2, 3, 4]), 50)
    print("50 steps", round(time.time() - t, 1), "s (includes compile)")
    t = time.time()
    L.train(labeled_sampler(rng, [1, 2, 3, 4]), 200)
    print("200 steps", round(time.time() - t, 1), "s")
    t = time.time()
    print(evaluate(L, np.random.default_rng(999), range(1, 9), 500), round(time.time() - t, 1), "s eval")
