"""Check SPEC_CHAIN.md's recorded predictions on the chaining runs, with paired bootstrap intervals,
and describe the training data of each condition.

    python3 -I check_chain.py results/chain

Descriptive only: the decision is applied by `summarize.py --gate chain`. The intervals resample
test problems within each seed (the same problems are used for every condition of a seed) and do
not include seed-to-seed variation; per-seed values are printed for that. The training sets are
regenerated locally with the registered code and checked against the statistics each run recorded.
"""
import collections
import glob
import gzip
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this repo's modules (-I drops it)
import problems as P  # noqa: E402
import run_llm as R  # noqa: E402

SEEDS = (6, 7, 8)
SETS = ("train_vocab/4", "train_vocab/8", "train_vocab/12", "train_vocab/16", "heldout_vocab/12")
CONDS = ("plain", "cancel_spread", "chain", "oracle")
runs, stats = {}, {}
for root in sys.argv[1:]:
    for mp in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        m = json.load(open(mp))
        g = os.path.join(os.path.dirname(mp), "generations.jsonl")
        rows = [json.loads(line) for line in (open(g) if os.path.exists(g) else gzip.open(g + ".gz", "rt"))]
        key = (m["args"]["condition"], m["args"]["seed"])
        runs[key] = {s: [r for r in rows if r["set"] == s] for s in m["metrics"]}
        stats[key] = m["train_stats"]


def acc(c, s, k):
    return float(np.mean([r["correct"] for r in runs[(c, s)][k]]))


def mean(c, k):
    return float(np.mean([acc(c, s, k) for s in SEEDS]))


print("== means over seeds 6-8")
for k in SETS:
    print(f"  {k:17s} " + "  ".join(f"{c} {mean(c, k):.3f}" for c in CONDS))
for s in SEEDS:
    same = all([r["text"] for r in runs[(c, s)][k]] == [r["text"] for r in runs[("plain", s)][k]]
               for c in CONDS for k in SETS)
    print(f"  seed {s}: the same test problems for every condition, in every set: {same}")

rng = np.random.default_rng(0)


def paired(c1, c2, k, n_boot=10000):
    per = [np.array([r["correct"] for r in runs[(c1, s)][k]], float)
           - np.array([r["correct"] for r in runs[(c2, s)][k]], float) for s in SEEDS]
    boot = [np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in per]) for _ in range(n_boot)]
    return float(np.mean([d.mean() for d in per])), [float(d.mean()) for d in per], np.percentile(boot, [2.5, 97.5])


print("\n== paired differences (mean over seeds; per seed; 95% bootstrap over problems)")
for c1, c2, k in [("chain", "cancel_spread", k) for k in SETS] + [
        ("oracle", "chain", "train_vocab/12"), ("cancel_spread", "plain", "train_vocab/12"),
        ("chain", "plain", "train_vocab/4")]:
    m, per, (lo, hi) = paired(c1, c2, k)
    print(f"  {c1} - {c2} at {k}: {m:+.3f}  per seed {[round(x, 3) for x in per]}  [{lo:+.3f}, {hi:+.3f}]")


def big(c, k, lo=60):
    """(correct, total) on test answers >= lo, pooled over seeds, and per seed."""
    per = [[r["correct"] for r in runs[(c, s)][k] if r["answer"] >= lo] for s in SEEDS]
    return sum(map(sum, per)), sum(map(len, per)), [f"{sum(x)}/{len(x)}" for x in per]


print("\n== test answers of 60 or more (12 events is the registered measure; 16 events is descriptive)")
for k in ("train_vocab/12", "train_vocab/16"):
    for c in CONDS:
        n_ok, n, per = big(c, k)
        print(f"  {k:15s} {c:14s} {n_ok}/{n} = {n_ok / max(n, 1):.3f}  per seed {per}")

print("\n== recorded predictions (SPEC_CHAIN.md)")
d = [acc("chain", s, "train_vocab/12") - acc("cancel_spread", s, "train_vocab/12") for s in SEEDS]
verdict = ("CHAINING BETTER" if np.mean(d) >= 0.05 and min(d) > 0 else
           "METHOD BETTER" if np.mean(d) <= -0.05 and max(d) < 0 else "NO CLEAR DIFFERENCE")
print(f"  1. verdict CHAINING BETTER: d per seed {[round(x, 3) for x in d]}, mean {np.mean(d):+.3f}"
      f" -> {verdict} -> {'held' if verdict == 'CHAINING BETTER' else 'missed'}")
c12, m12 = mean("chain", "train_vocab/12"), mean("cancel_spread", "train_vocab/12")
print(f"  2. chain >= 65% at 12: {c12:.3f} -> {'held' if c12 >= 0.65 else 'missed'};"
      f" the method {m12:.3f} (final: 0.582)")
n_ok, n, _ = big("chain", "train_vocab/12")
m_ok, m_n, _ = big("cancel_spread", "train_vocab/12")
print(f"  3. chain > 10% on answers >= 60 at 12: {n_ok}/{n} = {n_ok / n:.3f} pooled -> "
      f"{'held' if n_ok / n > 0.10 else 'missed'}; the method {m_ok}/{m_n} -> {'held' if m_ok == 0 else 'missed'}")
rep = [acc("cancel_spread", s, "train_vocab/12") - acc("plain", s, "train_vocab/12") for s in SEEDS]
print(f"  4. replication (cancel_spread - plain >= 15 points at 12 on every seed): {[round(x, 3) for x in rep]}"
      f" -> {'held' if min(rep) >= 0.15 else 'missed'}")
cost = mean("plain", "train_vocab/4") - mean("chain", "train_vocab/4")
print(f"  5. chain costs <= 3 points at 4 events (plain - chain): {cost:+.3f} -> {'held' if cost <= 0.03 else 'missed'}")

print("\n== training data, regenerated with the registered code (seed 6)")
train = {c: R.build_train(random.Random(6), c, "additive", 24000, 4, 16) for c in ("cancel_spread", "chain", "oracle")}
for c, ps in train.items():
    rec = stats[(c, 6)]
    ch, am = float(np.mean([p.n_changes() for p in ps])), max(p.answer() for p in ps)
    print(f"  {c:14s} changes mean {ch:.4f} (run recorded {rec['changes_mean']:.4f}), answer max {am}"
          f" (run recorded {rec['answer_max']}) -> {'same' if abs(ch - rec['changes_mean']) < 1e-9 and am == rec['answer_max'] else 'DIFFERENT'}")
print("  answer distribution by number of changes (mean; share of answers >= 30; count)")
print("  changes   " + "  ".join(f"{c:>30s}" for c in train))
for lo, hi in ((1, 4), (5, 8), (9, 12), (13, 16)):
    cells = []
    for c, ps in train.items():
        a = [p.answer() for p in ps if lo <= p.n_changes() <= hi]
        cells.append(f"{np.mean(a):6.1f}  {np.mean([x >= 30 for x in a]):.2f}  n={len(a):5d}" if a else f"{'-':>30s}")
    print(f"  {lo:2d}-{hi:<2d}     " + "  ".join(f"{x:>30s}" for x in cells))
tests = R.build_eval(6, "additive", [12], 500)[12]
print(f"  test, 12 events (seed 6): answer mean {np.mean([p.answer() for p in tests]):.1f},"
      f" share >= 30 {np.mean([p.answer() >= 30 for p in tests]):.2f}")
