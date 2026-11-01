"""Learned optimal stopping for repeated sampling, fitted by least-squares Monte Carlo.

The decision after n attempts is: stop and return the current majority answer, or draw one
more attempt. The objective is expected correctness minus lam * generated tokens, so the
policy may stop early for two different reasons: the answer is settled, or more attempts are
unlikely to turn a wrong answer into a right one.

The value of continuing is learned by Longstaff-Schwartz backward induction on replayed
attempt pools: at each step n, least-squares regress the realised value-to-go of the paths on
a polynomial basis of the state features, decide along each path with that regression and
propagate the realised value. The value of stopping is a calibrated P(majority answer correct
| state) from a logistic model on the same basis. Ground-truth labels are used only on
development questions to fit the two regressions.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

import lab

CAP = 32


@lru_cache(maxsize=None)
def _beta_prob(top, second):
    return lab.beta_mode_prob(top, second)


COUNT_FEATURES = ["n", "share1", "share2", "distinct", "none", "beta", "margin"]
FULL_FEATURES = COUNT_FEATURES + ["lead_conf_tail", "lead_conf_mean", "lead_conf_low", "all_conf_tail",
                                  "rest_conf_tail", "lead_log_tokens", "all_log_tokens", "lead_false",
                                  "unfinished", "answer_len"]


def state_features(used):
    """Features of the attempts drawn so far (all observable without labels)."""
    n = len(used)
    answers = [t.answer for t in used]
    counts = {}
    for a in answers:
        if a is not None:
            counts[a] = counts.get(a, 0) + 1
    ordered = sorted(counts.values(), reverse=True) + [0, 0]
    v1, v2 = ordered[0], ordered[1]
    leader, _ = lab.vote(used)
    lead = [t for t, a in zip(used, answers) if a == leader and a is not None]
    rest = [t for t, a in zip(used, answers) if a != leader or a is None]
    mean = lambda xs, k, d=0.0: sum(getattr(t, k) for t in xs) / len(xs) if xs else d
    f = {
        "n": n, "share1": v1 / n, "share2": v2 / n, "distinct": len(counts) / n,
        "none": sum(a is None for a in answers) / n, "beta": _beta_prob(v1, v2), "margin": v1 - v2,
        "lead_conf_tail": mean(lead, "conf_tail"), "lead_conf_mean": mean(lead, "conf_mean"),
        "lead_conf_low": min((t.conf_low for t in lead), default=0.0),
        "all_conf_tail": mean(used, "conf_tail"), "rest_conf_tail": mean(rest, "conf_tail", -1.0),
        "lead_log_tokens": math.log1p(mean(lead, "tokens")), "all_log_tokens": math.log1p(mean(used, "tokens")),
        "lead_false": sum(t.false > 0 for t in lead) / len(lead) if lead else 0.0,
        "unfinished": sum(not t.finished for t in used) / n,
        "answer_len": math.log1p(len(leader)) if leader else 0.0,
    }
    return f, leader


def build_paths(questions, repeats=16, seed=0, cap=CAP):
    """Replayed orderings: per path and step, features, correctness, majority agreement, tokens."""
    rng = np.random.default_rng(seed)
    rows = []
    for qi, q in enumerate(questions):
        full_major = lab.vote(q.traces)[0]
        for _ in range(repeats):
            pool = [q.traces[j] for j in rng.permutation(len(q.traces))[:cap]]
            feats, correct, agree = [], [], []
            for n in range(1, len(pool) + 1):
                f, leader = state_features(pool[:n])
                feats.append([f[k] for k in FULL_FEATURES])
                correct.append(leader is not None and leader == q.gold)
                agree.append(leader is not None and leader == full_major)
            rows.append({"q": qi, "X": np.array(feats, float), "correct": np.array(correct, float),
                         "agree": np.array(agree, float), "tok": np.array([t.tokens for t in pool], float)})
    return rows


class Basis:
    """Standardise the chosen features, then expand to all degree<=2 monomials."""

    def __init__(self, feature_set="full"):
        names = FULL_FEATURES if feature_set == "full" else COUNT_FEATURES
        self.cols = [FULL_FEATURES.index(k) for k in names]

    def fit(self, X):
        Z = X[:, self.cols]
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        return self

    def __call__(self, X):
        Z = (X[:, self.cols] - self.mu) / self.sd
        i, j = np.triu_indices(Z.shape[1])
        return np.hstack([np.ones((len(Z), 1)), Z, Z[:, i] * Z[:, j]])


def _ridge(B, y, alpha=1.0):
    A = B.T @ B + alpha * np.eye(B.shape[1])
    A[0, 0] -= alpha  # do not shrink the intercept
    return np.linalg.solve(A, B.T @ y)


class StoppingPolicy:
    """Stop at the first n with  P(correct | s_n) >= E[value of continuing | s_n]."""

    def __init__(self, lam, target="correct", feature_set="full", cap=CAP):
        self.lam, self.target, self.feature_set, self.cap = lam, target, feature_set, cap
        self.basis, self.p_model, self.cont = None, None, {}

    def fit_stop_value(self, paths, logistic=True):
        """Feature basis, and (for the two-value variant) a logistic P(majority correct | state)."""
        from sklearn.linear_model import LogisticRegression

        cap = min(self.cap, min(len(p["tok"]) for p in paths))
        X = np.concatenate([p["X"][:cap] for p in paths])
        self.basis = Basis(self.feature_set).fit(X)
        if logistic:
            y = np.concatenate([p[self.target][:cap] for p in paths])
            self.p_model = LogisticRegression(C=1.0, max_iter=2000).fit(self.basis(X)[:, 1:], y)
        self.cap = cap
        return self

    def stop_value(self, X):
        return self.p_model.predict_proba(self.basis(X)[:, 1:])[:, 1]

    def fit(self, paths, stop_model=None, advantage=True):
        """Backward induction. With ``advantage`` (default) each step regresses the realised gain of
        continuing over stopping *on the same path*, G_{n+1} - lam*t_{n+1} - r_n, and stops when the
        fitted gain is <= 0. The two values share most of their noise, so their difference is far
        less noisy than either; comparing two separately fitted values stops on noise."""
        if stop_model is None:
            self.fit_stop_value(paths, logistic=not advantage)
        else:
            self.basis, self.p_model, self.cap = stop_model.basis, stop_model.p_model, stop_model.cap
        self.advantage = advantage
        cap = self.cap
        steps = np.stack([p["X"][:cap] for p in paths])  # paths x steps x features
        tok = np.stack([p["tok"][:cap] for p in paths])
        reward = np.stack([p[self.target][:cap] for p in paths])
        if not advantage:
            s_hat = self.stop_value(steps.reshape(-1, steps.shape[2])).reshape(len(paths), cap)
        G = reward[:, cap - 1].copy()  # forced stop at the cap
        for n in range(cap - 1, 0, -1):  # decision after n attempts (column n-1)
            B = self.basis(steps[:, n - 1, :])
            target = G - self.lam * tok[:, n]
            if advantage:
                beta = _ridge(B, target - reward[:, n - 1])
                stop = B @ beta <= 0.0
            else:
                beta = _ridge(B, target)
                stop = s_hat[:, n - 1] >= B @ beta
            self.cont[n] = beta
            G = np.where(stop, reward[:, n - 1], target)
        return self

    def _stop_matrix(self, X):
        """Boolean paths x steps: would the policy stop at this step?"""
        P, T = X.shape[0], X.shape[1]
        stop = np.ones((P, T), bool)  # the last column always stops
        if not getattr(self, "advantage", True):
            s_hat = self.stop_value(X.reshape(-1, X.shape[2])).reshape(P, T)
        for n in range(1, T):
            fitted = self.basis(X[:, n - 1, :]) @ self.cont[n]
            stop[:, n - 1] = fitted <= 0.0 if getattr(self, "advantage", True) else s_hat[:, n - 1] >= fitted
        return stop

    def stop_steps(self, paths):
        """First stopping step (1-based) on each prepared path, vectorised over steps and paths."""
        T = min(self.cap, min(len(p["tok"]) for p in paths))
        X = np.stack([p["X"][:T] for p in paths])
        return self._stop_matrix(X).argmax(axis=1) + 1

    def run(self, traces):
        """Online use: draw attempts in order until the policy stops."""
        used = []
        for t in traces[: self.cap]:
            used.append(t)
            f, _ = state_features(used)
            x = np.array([[f[k] for k in FULL_FEATURES]], float)
            n = len(used)
            if n >= self.cap:
                break
            fitted = (self.basis(x) @ self.cont[n])[0]
            if (fitted <= 0.0) if getattr(self, "advantage", True) else self.stop_value(x)[0] >= fitted:
                break
        answer, _ = lab.vote(used)
        return answer, sum(t.tokens for t in used), len(used)


def cross_validate(questions, lams, folds=5, repeats_fit=16, repeats_eval=20, target="correct",
                   feature_set="full", seed=0, configs=None, advantage=True):
    """Per-question accuracy/tokens where each question is decided by a policy not fitted on it.

    ``configs`` is a list of (target, feature_set); without it the single (target, feature_set)
    is run and the result is keyed by lam only. Paths are built once per fold and shared.
    """
    single = configs is None
    configs = [(target, feature_set)] if single else list(configs)
    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(len(questions)) % folds
    shape = len(questions)
    res = {c: {lam: {"acc": np.zeros(shape), "tok": np.zeros(shape), "att": np.zeros(shape)} for lam in lams}
           for c in configs}
    for k in range(folds):
        train = [q for q, f in zip(questions, fold_of) if f != k]
        test_idx = [i for i, f in enumerate(fold_of) if f == k]
        paths = build_paths(train, repeats=repeats_fit, seed=seed + 100 + k)
        eval_paths = build_paths([questions[i] for i in test_idx], repeats=repeats_eval, seed=seed + 1000 + k)
        for c in configs:
            tgt, fs = c
            stop_model = StoppingPolicy(0.0, target=tgt, feature_set=fs).fit_stop_value(paths, logistic=not advantage)
            for lam in lams:
                policy = StoppingPolicy(lam, target=tgt, feature_set=fs).fit(paths, stop_model, advantage=advantage)
                r = res[c][lam]
                for p, n in zip(eval_paths, policy.stop_steps(eval_paths)):
                    i = test_idx[p["q"]]
                    r["acc"][i] += p["correct"][n - 1] / repeats_eval
                    r["tok"][i] += p["tok"][:n].sum() / repeats_eval
                    r["att"][i] += n / repeats_eval
    out = {c: {lam: {"accuracy": float(r["acc"].mean()), "tokens": float(r["tok"].mean()),
                     "attempts": float(r["att"].mean()), "per_question": r["acc"], "per_question_tokens": r["tok"]}
               for lam, r in by_lam.items()} for c, by_lam in res.items()}
    return out[configs[0]] if single else out
