"""Check SPEC_MULTI.md's recorded predictions on the three-person final, with paired bootstrap
intervals and self-improvement's round diagnostics. Written before the final's results existed.

    python3 -I check_multi.py results/multi_final

Descriptive only: the verdict comes from `summarize_multi.py --gate final`. The intervals resample
test problems within each seed (every condition of a seed has the same test problems) and do not
include seed-to-seed variation; per-seed values are printed for that.
"""
import glob
import gzip
import json
import os
import sys

import numpy as np

SEEDS = (9, 10, 11)
SETS = ("train_vocab/4", "train_vocab/8", "train_vocab/12", "train_vocab/16", "heldout_vocab/12")
CONDS = ("plain", "cancel_spread", "self_improve", "oracle")
runs, metas = {}, {}
for root in sys.argv[1:]:
    for mp in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        m = json.load(open(mp))
        if m["args"].get("family") != "multi":
            continue
        g = os.path.join(os.path.dirname(mp), "generations.jsonl")
        rows = [json.loads(line) for line in (open(g) if os.path.exists(g) else gzip.open(g + ".gz", "rt"))]
        key = (m["args"]["condition"], m["args"]["seed"])
        runs[key] = {s: [r for r in rows if r["set"] == s] for s in m["metrics"]}
        metas[key] = m


def acc(c, s, k):
    return float(np.mean([r["correct"] for r in runs[(c, s)][k]]))


def mean(c, k):
    return float(np.mean([acc(c, s, k) for s in SEEDS]))


print("== means over seeds 9-11")
for k in SETS:
    print(f"  {k:17s} " + "  ".join(f"{c} {mean(c, k):.3f}" for c in CONDS))
for s in SEEDS:
    same = all([r["text"] for r in runs[(c, s)][k]] == [r["text"] for r in runs[("plain", s)][k]] for c in CONDS for k in SETS)
    print(f"  seed {s}: the same test problems for every condition, in every set: {same}")

rng = np.random.default_rng(0)


def paired(c1, c2, k, n_boot=10000):
    per = [np.array([r["correct"] for r in runs[(c1, s)][k]], float)
           - np.array([r["correct"] for r in runs[(c2, s)][k]], float) for s in SEEDS]
    boot = [np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in per]) for _ in range(n_boot)]
    return float(np.mean([d.mean() for d in per])), [float(d.mean()) for d in per], np.percentile(boot, [2.5, 97.5])


print("\n== paired differences (mean over seeds; per seed; 95% bootstrap over problems)")
for c1, c2, k in [("cancel_spread", "self_improve", k) for k in SETS] + [
        ("cancel_spread", "plain", "train_vocab/12"), ("self_improve", "plain", "train_vocab/12"),
        ("oracle", "cancel_spread", "train_vocab/12"), ("plain", "cancel_spread", "train_vocab/4"),
        ("plain", "self_improve", "train_vocab/4")]:
    m, per, (lo, hi) = paired(c1, c2, k)
    print(f"  {c1} - {c2} at {k}: {m:+.3f}  per seed {[round(x, 3) for x in per]}  [{lo:+.3f}, {hi:+.3f}]")

print("\n== self-improvement rounds (kept by the vote; label accuracy is a diagnostic only)")
first_below = {}
for s in SEEDS:
    si = metas[("self_improve", s)]["self_improve"]
    cells = [f"r{r['round']} {r['events'][0]}-{r['events'][1]}: kept {r['kept']}/{r['pool']}, right "
             f"{r['kept_label_accuracy_diagnostic']:.3f}" if r["kept_label_accuracy_diagnostic"] is not None else
             f"r{r['round']}: kept 0" for r in si["rounds"]]
    print(f"  seed {s}: " + "; ".join(cells))
    below = [r["round"] for r in si["rounds"] if (r["kept_label_accuracy_diagnostic"] or 0) < 0.85]
    first_below[s] = below[0] if below else None
    others = si.get("final_other_models", {})
    print(f"    models 1 and 2 at 12 events: {[round(v['train_vocab/12']['accuracy'], 3) for _, v in sorted(others.items())]}")

print("\n== recorded predictions (SPEC_MULTI.md)")
v = [acc("cancel_spread", s, "train_vocab/12") - acc("plain", s, "train_vocab/12") for s in SEEDS]
d = [acc("cancel_spread", s, "train_vocab/12") - acc("self_improve", s, "train_vocab/12") for s in SEEDS]
if min(v) < 0.15:
    verdict = "METHOD FAILS HERE"
elif np.mean(d) >= 0.05 and min(d) > 0:
    verdict = "METHOD BETTER"
elif np.mean(d) <= -0.05 and max(d) < 0:
    verdict = "SELF-IMPROVEMENT BETTER"
else:
    verdict = "NO CLEAR DIFFERENCE"
print(f"  1. verdict METHOD BETTER: d per seed {[round(x, 3) for x in d]}, mean {np.mean(d):+.3f} -> {verdict}"
      f" -> {'held' if verdict == 'METHOD BETTER' else 'missed'}")
cm, sm = mean("cancel_spread", "train_vocab/12"), mean("self_improve", "train_vocab/12")
print(f"  2. at 12 events: the method {cm:.3f} (predicted 0.60-0.75: {'held' if 0.60 <= cm <= 0.75 else 'missed'}),"
      f" self-improvement {sm:.3f} (predicted 0.40-0.60: {'held' if 0.40 <= sm <= 0.60 else 'missed'}),"
      f" plain {mean('plain', 'train_vocab/12'):.3f} (about 0.28), oracle {mean('oracle', 'train_vocab/12'):.3f} (about 0.89)")
print(f"  3. validity (method - plain >= 15 points on every seed): {[round(x, 3) for x in v]}"
      f" -> {'held' if min(v) >= 0.15 else 'missed'}")
ok4 = all(b is not None and b <= 4 for b in first_below.values())
print(f"  4. kept labels below 85% accuracy by round 4: first such round per seed {first_below}"
      f" -> {'held' if ok4 else 'missed'}")
cost = mean("plain", "train_vocab/4") - mean("cancel_spread", "train_vocab/4")
print(f"  5. the method costs at most 3 points at 4 events: plain - method {cost:+.3f} -> {'held' if cost <= 0.03 else 'missed'}")

print("\n== compute: training seconds per run (mean over seeds)")
for c in CONDS:
    print(f"  {c:14s} {np.mean([metas[(c, s)]['train_seconds'] for s in SEEDS]):8.0f}")
