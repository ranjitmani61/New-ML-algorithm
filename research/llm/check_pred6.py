"""Check prediction 6 of SPEC_MULTI.md (addendum), written before the final's results existed.

    python3 -I check_pred6.py RUN_ROOT [RUN_ROOT ...]

Prediction 6: at 12 events (training vocabulary), pooled over seeds 9-11, the method's accuracy on
answers of 45 or more is at least 20 points below its accuracy on answers of 29 or less, and the
oracle's drop between the same answer sizes is smaller than the method's.

Uses audit_multi.py's bins and rows (section 2). Pooling weights each bin by its problem count,
which is the same as counting correct answers over all pooled problems; this script counts them
exactly instead of re-weighting audit's rounded accuracies.
"""
import glob
import gzip
import json
import os
import sys

SEEDS = (9, 10, 11)
LOW, HIGH = (0, 30), (45, 10**9)  # audit_multi.py bins: 0-29, and 45-59 plus 60-max


def rows_of(run):
    g = os.path.join(run, "generations.jsonl")
    return [json.loads(line) for line in (open(g) if os.path.exists(g) else gzip.open(g + ".gz", "rt"))]


runs = {}
for root in sys.argv[1:]:
    for mp in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        m = json.load(open(mp))
        if m["args"].get("family") != "multi":
            continue
        runs[(m["args"]["condition"], m["args"]["seed"])] = rows_of(os.path.dirname(mp))


def cell(cond, seeds, lo, hi):
    b = [r for s in seeds for r in runs[(cond, s)] if r["set"] == "train_vocab/12" and lo <= r["answer"] < hi]
    return sum(r["correct"] for r in b), len(b)


drops = {}
for cond in ("plain", "cancel_spread", "self_improve", "oracle"):
    per = []
    for s in SEEDS:
        (cl, nl), (ch, nh) = cell(cond, [s], *LOW), cell(cond, [s], *HIGH)
        per.append(f"s{s} {cl / nl:.3f} (n={nl}) vs {ch / max(nh, 1):.3f} (n={nh})")
    (cl, nl), (ch, nh) = cell(cond, SEEDS, *LOW), cell(cond, SEEDS, *HIGH)
    drops[cond] = cl / nl - ch / nh
    print(f"{cond:14s} pooled: answers 0-29 {cl}/{nl} = {cl / nl:.3f}; answers 45+ {ch}/{nh} = {ch / nh:.3f};"
          f" drop {drops[cond]:+.3f} | " + "; ".join(per))

m, o = drops["cancel_spread"], drops["oracle"]
print(f"\nprediction 6: method's drop {m:+.3f} (needs >= +0.20): {'held' if m >= 0.20 else 'missed'};"
      f" oracle's drop {o:+.3f} smaller than the method's: {'held' if o < m else 'missed'}"
      f" -> {'HELD' if m >= 0.20 and o < m else 'MISSED'}")
