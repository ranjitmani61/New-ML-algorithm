"""Offline lab for inference-time reasoning algorithms.

Pools hold N independent attempts per question (from kaggle_runner.py). A method
is evaluated by replaying attempts in random orders (without replacement), so
every method sees identical model behaviour; cost is the number of generated
tokens actually consumed. Results are accuracy-vs-tokens curves averaged over
many orderings, compared at matched token budgets with paired bootstrap
intervals over questions.

Methods
  sc            majority vote over the first k attempts (self-consistency)
  conf          confidence-weighted vote with top-eta filtering (DeepConf offline)
  asc           Adaptive-Consistency: stop when P(leader is the mode) >= c (Beta posterior)
  esc           Early-stopping SC: windows of w attempts, stop when a window agrees
  uab           Uncertainty-Aware Budget allocation across questions (vote entropy, greedy)
  vsc           Verified-Step Self-Consistency (new): exact zero-token checking of each
                attempt's arithmetic steps; falsified attempts are down-weighted, and a
                final answer that is the written result of a false final computation is
                repaired to the exact value; optional confidence weighting; sequential
                stopping on the weighted posterior.
"""

from __future__ import annotations

import gzip
import glob
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

import benchmarks as bm


@dataclass
class Trace:
    answer: str | None
    tokens: int
    finished: bool
    conf_mean: float
    conf_tail: float
    conf_low: float
    lp_mean: float
    claims: int
    false: int
    repaired: str | None  # answer after final-step repair (None if no repair applies)


@dataclass
class Question:
    qid: str
    set: str
    gold: str
    traces: list = field(default_factory=list)


def trace_from_summary(s):
    """Trace from a benchmarks.summarize_attempt record."""
    raw, tokens, finished, mean, tail_mean, low, lp, claims, false, last = s
    answer = bm.normalize(raw) if finished else None
    repaired = None
    if answer is not None and last is not None and not last[0] and bm.normalize(last[1]) == answer:
        repaired = bm.normalize(last[2])
    return Trace(answer, int(tokens), bool(finished), mean, tail_mean, low, lp, int(claims), int(false), repaired)


def load_features(path, sets=None):
    """Questions from a features.jsonl(.gz) produced by the runner's log transport."""
    opener = gzip.open if path.endswith(".gz") else open
    questions = []
    with opener(path, "rt") as fh:
        for line in fh:
            item = json.loads(line)
            if sets and item["set"] not in sets:
                continue
            q = Question(item["id"], item["set"], bm.normalize(item["gold"]))
            q.traces = [trace_from_summary(s) for s in item["samples"]]
            questions.append(q)
    return sorted(questions, key=lambda q: q.qid)


def trace_features(sample, conf=None, logprob=None, window=64, tail=256):
    text = sample["text"]
    raw = bm.model_answer(text) if sample["finished"] else None
    answer = bm.normalize(raw)
    steps = bm.check_steps(text) if sample["finished"] else {"claims": 0, "false": 0, "last": None}
    repaired = None
    last = steps.get("last")
    if answer is not None and last is not None and not last[0] and bm.normalize(last[1]) == answer:
        repaired = bm.normalize(last[2])
    if conf is not None and len(conf):
        c = np.asarray(conf, dtype=np.float64)
        tail_mean = float(c[-tail:].mean())
        if len(c) >= window:
            csum = np.concatenate([[0.0], np.cumsum(c)])
            low = float(((csum[window:] - csum[:-window]) / window).min())
        else:
            low = float(c.mean())
        mean = float(c.mean())
        lp = float(np.mean(logprob)) if logprob is not None and len(logprob) else 0.0
    else:
        mean = tail_mean = low = sample.get("mean_conf", 0.0)
        lp = sample.get("sum_logprob", 0.0) / max(1, sample["tokens"])
    return Trace(answer, int(sample["tokens"]), bool(sample["finished"]), mean, tail_mean, low, lp,
                 int(steps["claims"]), int(steps["false"]), repaired)


def load_pools(directory, sets=None):
    questions = []
    for part in sorted(glob.glob(os.path.join(directory, "**", "part*.jsonl.gz"), recursive=True)):
        arrays = np.load(part.replace(".jsonl.gz", ".npz"))
        conf, lp, off = arrays["conf"], arrays["logprob"], arrays["offsets"]
        for line in gzip.open(part, "rt"):
            item = json.loads(line)
            if sets and item["set"] not in sets:
                continue
            q = Question(item["id"], item["set"], bm.normalize(bm.gold_answer(item)))
            for s in item["samples"]:
                a = s["array"]
                q.traces.append(trace_features(s, conf[off[a]:off[a + 1]], lp[off[a]:off[a + 1]]))
            questions.append(q)
    return sorted(questions, key=lambda q: q.qid)


# ---------------------------------------------------------------- voting rules


def vote(traces, weights=None, use_repair=False):
    scores = {}
    first = {}
    for i, t in enumerate(traces):
        a = t.repaired if (use_repair and t.repaired is not None) else t.answer
        if a is None:
            continue
        w = 1.0 if weights is None else weights[i]
        scores[a] = scores.get(a, 0.0) + w
        first.setdefault(a, i)
    if not scores:
        return None, {}
    best = max(scores, key=lambda a: (scores[a], -first[a]))
    return best, scores


def conf_weights(traces, measure="conf_tail", eta=1.0):
    values = np.array([getattr(t, measure) for t in traces], dtype=np.float64)
    w = values.copy()
    if eta < 1.0 and len(values) > 1:
        cut = np.quantile(values, 1.0 - eta)
        w[values < cut] = 0.0
    return list(w)


def vsc_weights(traces, rho=0.25, alpha=0.0, measure="conf_tail", repair=True):
    """Verification weight rho**(uncorrected false steps), optionally times confidence**alpha.

    A repaired attempt's final false step has been corrected, so it no longer counts.
    """
    out = []
    for t in traces:
        false = t.false - (1 if repair and t.repaired is not None else 0)
        w = rho ** min(max(false, 0), 4)
        if alpha:
            w *= max(getattr(t, measure), 1e-6) ** alpha
        out.append(w)
    return out


# ---------------------------------------------------------------- per-question policies


def beta_mode_prob(top, second):
    """P(p_top > p_second) under a Beta(top+1, second+1) posterior on their split."""
    from scipy.stats import beta

    return float(beta.sf(0.5, top + 1.0, second + 1.0))


def run_fixed(traces, k, rule):
    used = traces[:k]
    return rule(used), sum(t.tokens for t in used), len(used)


def run_sequential(traces, rule_scores, stop, cap, minimum=1):
    """Draw attempts one at a time; stop(scores, n) decides; rule_scores -> (answer, scores)."""
    used = []
    for t in traces[:cap]:
        used.append(t)
        answer, scores = rule_scores(used)
        if len(used) >= minimum and stop(scores, len(used)):
            break
    answer, _ = rule_scores(used)
    return answer, sum(t.tokens for t in used), len(used)


def asc_stop(threshold):
    def stop(scores, n):
        if not scores:
            return False
        v = sorted(scores.values(), reverse=True) + [0.0]
        return beta_mode_prob(v[0], v[1]) >= threshold
    return stop


def esc_run(traces, window, cap):
    used = []
    for start in range(0, min(cap, len(traces)), window):
        block = traces[start:start + window]
        used.extend(block)
        answers = [t.answer for t in block]
        if len(block) == window and answers[0] is not None and len(set(answers)) == 1:
            break
    answer, _ = vote(used)
    return answer, sum(t.tokens for t in used), len(used)


# ---------------------------------------------------------------- batch allocation


def uab_allocate(pools, avg_samples, k0=2, tau=1.0, cap=None):
    """UAB (Nguyen et al. 2026): k0 probes per question, then greedy p(1-p)^m by vote entropy."""
    cap = cap or min(len(p) for p in pools)
    counts = [min(k0, len(p)) for p in pools]
    p_hat = []
    for p, c in zip(pools, counts):
        ans = [t.answer for t in p[:c]]
        freq = defaultdict(int)
        for a in ans:
            freq[a] += 1
        h = -sum((f / c) * math.log(f / c) for f in freq.values()) / math.log(c) if c > 1 else 0.0
        p_hat.append(math.exp(-h / tau))
    budget = int(round(avg_samples * len(pools))) - sum(counts)
    import heapq

    heap = [(-(ph * (1 - ph) ** c), i) for i, (ph, c) in enumerate(zip(p_hat, counts)) if c < cap]
    heapq.heapify(heap)
    while budget > 0 and heap:
        _, i = heapq.heappop(heap)
        counts[i] += 1
        budget -= 1
        if counts[i] < cap:
            heapq.heappush(heap, (-(p_hat[i] * (1 - p_hat[i]) ** counts[i]), i))
    return counts


# ---------------------------------------------------------------- evaluation


def simulate(questions, method, params, repeats=40, seed=0, cap=None):
    """Mean accuracy, tokens/question and attempts/question over random orderings."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(len(questions))
    tokens = np.zeros(len(questions))
    used = np.zeros(len(questions))
    for _ in range(repeats):
        orders = [rng.permutation(len(q.traces)) for q in questions]
        pools = [[q.traces[j] for j in o] for q, o in zip(questions, orders)]
        if method == "uab":
            counts = uab_allocate(pools, params["avg"], cap=cap)
            for i, (q, p, c) in enumerate(zip(questions, pools, counts)):
                a, _ = vote(p[:c])
                acc[i] += a == q.gold
                tokens[i] += sum(t.tokens for t in p[:c])
                used[i] += c
            continue
        for i, (q, p) in enumerate(zip(questions, pools)):
            a, tk, n = run_method(p, method, params, cap or len(p))
            acc[i] += a == q.gold
            tokens[i] += tk
            used[i] += n
    return {"per_question": acc / repeats, "accuracy": float(acc.mean() / repeats),
            "tokens": float(tokens.mean() / repeats), "attempts": float(used.mean() / repeats)}


def run_method(p, method, params, cap):
    if method == "sc":
        return run_fixed(p, params["k"], lambda u: vote(u)[0])
    if method == "conf":
        return run_fixed(p, params["k"], lambda u: vote(u, conf_weights(u, params.get("measure", "conf_tail"),
                                                                          params.get("eta", 1.0)))[0])
    if method == "vsc_fixed":
        return run_fixed(p, params["k"], lambda u: vote(u, vsc_weights(u, params["rho"], params.get("alpha", 0.0),
                                                                     repair=params.get("repair", True)),
                                                         params.get("repair", True))[0])
    if method == "asc":
        return run_sequential(p, lambda u: vote(u), asc_stop(params["c"]), cap)
    if method == "esc":
        return esc_run(p, params["w"], cap)
    if method == "vsc":
        rule = lambda u: vote(u, vsc_weights(u, params["rho"], params.get("alpha", 0.0),
                                             repair=params.get("repair", True)), params.get("repair", True))
        return run_sequential(p, rule, asc_stop(params["c"]), cap, minimum=params.get("min", 1))
    raise ValueError(method)


def curve(questions, method, grid, **kw):
    return [{"params": g, **{k: v for k, v in simulate(questions, method, g, **kw).items() if k != "per_question"}}
            for g in grid]


def accuracy_at(curve_points, budget):
    """Linear interpolation of accuracy at a token budget along a method's curve."""
    pts = sorted((p["tokens"], p["accuracy"]) for p in curve_points)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if budget <= xs[0]:
        return ys[0] if budget == xs[0] else None
    if budget >= xs[-1]:
        return ys[-1] if budget == xs[-1] else None
    return float(np.interp(budget, xs, ys))


def paired_bootstrap(a, b, reps=4000, seed=0):
    """CI for mean(a - b) over questions (per-question accuracies averaged over orderings)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    boots = rng.choice(d, size=(reps, len(d)), replace=True).mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_le_0": float(np.mean(boots <= 0))}


# ---------------------------------------------------------------- development driver

DEV_GRIDS = {
    "sc": [{"k": k} for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)],
    "conf": [{"k": k, "measure": m, "eta": e} for k in (4, 8, 16) for m in ("conf_tail", "conf_mean", "conf_low")
             for e in (1.0, 0.9, 0.5)],
    "asc": [{"c": c} for c in (0.6, 0.75, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)],
    "esc": [{"w": w} for w in (2, 3, 4, 5, 6, 8)],
    "uab": [{"avg": a} for a in (2, 3, 4, 6, 8, 12, 16)],
    "vsc_fixed": [{"k": k, "rho": r, "repair": rep} for k in (4, 8, 16) for r in (0.0, 0.1, 0.25, 0.5, 1.0)
                  for rep in (True, False)],
}


def describe(questions):
    """Pool statistics: accuracy, length, step-check coverage and how errors relate to correctness."""
    traces = [(t, t.answer == q.gold, q.set) for q in questions for t in q.traces]
    out = {"questions": len(questions), "attempts": len(traces)}
    for name, sel in (("all", lambda s: True), ("gsm8k", lambda s: s.startswith("gsm8k")),
                      ("math", lambda s: s.startswith("math"))):
        group = [(t, ok) for t, ok, s in traces if sel(s)]
        if not group:
            continue
        with_claims = [(t, ok) for t, ok in group if t.claims]
        falsified = [(t, ok) for t, ok in group if t.false]
        repaired = [(t, ok) for t, ok in group if t.repaired is not None]
        qs = [q for q in questions if sel(q.set)]
        out[name] = {
            "pass@1": float(np.mean([ok for _, ok in group])),
            "mean_tokens": float(np.mean([t.tokens for t, _ in group])),
            "unfinished": float(np.mean([not t.finished for t, _ in group])),
            "with_checked_claims": len(with_claims) / len(group),
            "claims_per_attempt": float(np.mean([t.claims for t, _ in group])),
            "falsified_attempts": len(falsified) / len(group),
            "accuracy_if_falsified": float(np.mean([ok for _, ok in falsified])) if falsified else None,
            "accuracy_if_not_falsified": float(np.mean([ok for t, ok in group if not t.false])),
            "repairable_attempts": len(repaired) / len(group),
            "repairs_wrong_to_right": sum(1 for q in qs for t in q.traces
                                          if t.repaired is not None and t.answer != q.gold and t.repaired == q.gold),
            "repairs_right_to_wrong": sum(1 for q in qs for t in q.traces
                                          if t.repaired is not None and t.answer == q.gold and t.repaired != q.gold),
            "oracle_any_correct": float(np.mean([any(t.answer == q.gold for t in q.traces) for q in qs])),
            "majority_all": float(np.mean([vote(q.traces)[0] == q.gold for q in qs])),
        }
    return out


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Development experiments on attempt pools")
    ap.add_argument("pools")
    ap.add_argument("--sets", nargs="+", default=["gsm8k_dev", "math_dev"])
    ap.add_argument("--repeats", type=int, default=40)
    ap.add_argument("--output")
    args = ap.parse_args(argv)
    if any(s.endswith("_test") for s in args.sets):
        raise SystemExit("test sets are sealed: evaluate them only through the preregistered final run")
    questions = load_pools(args.pools, set(args.sets))
    result = {"sets": args.sets, "describe": describe(questions), "curves": {}}
    for method, grid in DEV_GRIDS.items():
        result["curves"][method] = curve(questions, method, grid, repeats=args.repeats)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
    print(text[:20000])
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
