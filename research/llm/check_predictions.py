"""Check SPEC.md's recorded predictions on the final runs, with paired bootstrap intervals.

    python3 -I check_predictions.py results/final

Descriptive only: the pass criteria are applied by `summarize.py --gate final`. The intervals
resample test problems within each seed (the same problems are used for every condition of a
seed) and do not include seed-to-seed variation; per-seed values are printed for that.
"""
import glob
import gzip
import json
import os
import sys

import numpy as np

SEEDS = (3, 4, 5)
runs = {}
for root in sys.argv[1:]:
    for mp in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        m = json.load(open(mp))
        g = os.path.join(os.path.dirname(mp), "generations.jsonl")
        rows = [json.loads(line) for line in (open(g) if os.path.exists(g) else gzip.open(g + ".gz", "rt"))]
        runs[(m["args"]["condition"], m["args"]["seed"])] = {s: [r for r in rows if r["set"] == s] for s in m["metrics"]}


def acc(c, s, k):
    return float(np.mean([r["correct"] for r in runs[(c, s)][k]]))


def mean(c, k):
    return float(np.mean([acc(c, s, k) for s in SEEDS]))


print("means over seeds 3-5; share = (cancel_spread - plain) / (oracle - plain)")
for k in ("train_vocab/4", "train_vocab/8", "train_vocab/12", "train_vocab/16", "heldout_vocab/12"):
    pl, cs, ca, orc = (mean(c, k) for c in ("plain", "cancel_spread", "cancel_adjacent", "oracle"))
    print(f"  {k:17s} plain {pl:.3f}  cancel_spread {cs:.3f}  cancel_adjacent {ca:.3f}  oracle {orc:.3f}"
          f"  gain {cs - pl:+.3f}  share {(cs - pl) / (orc - pl):.1%}")

for s in SEEDS:
    ref = [r["text"] for r in runs[("plain", s)]["train_vocab/12"]]
    same = all([r["text"] for r in runs[(c, s)]["train_vocab/12"]] == ref
               for c in ("irrelevant", "inactive", "cancel_spread", "cancel_adjacent", "oracle"))
    print(f"  seed {s}: the same 12-event test problems for every condition: {same}")

rng = np.random.default_rng(0)


def paired(c1, c2, k, n_boot=10000):
    per = [np.array([r["correct"] for r in runs[(c1, s)][k]], float)
           - np.array([r["correct"] for r in runs[(c2, s)][k]], float) for s in SEEDS]
    boot = [np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in per]) for _ in range(n_boot)]
    return float(np.mean([d.mean() for d in per])), [float(d.mean()) for d in per], np.percentile(boot, [2.5, 97.5])


print("\npaired differences (mean over seeds; per seed; 95% bootstrap over problems)")
for c1, c2, k in (("cancel_spread", "plain", "train_vocab/12"), ("cancel_spread", "irrelevant", "train_vocab/12"),
                  ("cancel_spread", "plain", "train_vocab/4"), ("cancel_adjacent", "cancel_spread", "train_vocab/12"),
                  ("cancel_spread", "plain", "heldout_vocab/12"), ("oracle", "cancel_spread", "train_vocab/12")):
    m, per, (lo, hi) = paired(c1, c2, k)
    print(f"  {c1} - {c2} at {k}: {m:+.3f}  per seed {[round(x, 3) for x in per]}  [{lo:+.3f}, {hi:+.3f}]")

print("\nrecorded predictions")
g12 = mean("cancel_spread", "train_vocab/12") - mean("plain", "train_vocab/12")
g16 = mean("cancel_spread", "train_vocab/16") - mean("plain", "train_vocab/16")
print(f"  1. gains shrink at 16 events: gain at 12 {g12:+.3f}, at 16 {g16:+.3f} -> {'held' if g16 < g12 else 'missed'}"
      " (large answers: see audit_runs.py)")
d = mean("cancel_adjacent", "train_vocab/12") - mean("cancel_spread", "train_vocab/12")
print(f"  2. cancel_adjacent close to cancel_spread at 12: difference {d:+.3f}")
dh = [acc("cancel_spread", s, "heldout_vocab/12") - acc("cancel_spread", s, "train_vocab/12") for s in SEEDS]
print(f"  3. held-out vocabulary within 5 points at 12: per seed {[round(x, 3) for x in dh]}, mean {np.mean(dh):+.3f}"
      f" -> {'held' if abs(np.mean(dh)) <= 0.05 else 'missed'}")
for c in ("plain", "oracle"):
    dc = [acc(c, s, "heldout_vocab/12") - acc(c, s, "train_vocab/12") for s in SEEDS]
    print(f"     for comparison, {c}: per seed {[round(x, 3) for x in dc]}, mean {np.mean(dc):+.3f}")
