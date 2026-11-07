"""Answer sizes in each condition's training data vs the final's test sets (generator only, no model).

The method never changes a label, so its training answers are those of 1-4 event problems.
Generator only: no model output is read. This measures how often a test answer is larger than (almost) any answer the method trains on.
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_multi as RM  # noqa: E402

BINS = [(0, 29), (30, 44), (45, 59), (60, 10**9)]  # audit_multi.py's bins
for seed in (9, 10, 11):
    tr = {c: np.array([p.answer() for p in RM.build_train(random.Random(seed), c, 24000, 4, 16)])
          for c in ("plain", "cancel_spread", "oracle")}
    ev = {k: np.array([p.answer() for p in v]) for k, v in RM.build_eval(seed, [4, 8, 12, 16], 500).items()}
    q99, q999, mx = np.percentile(tr["cancel_spread"], 99), np.percentile(tr["cancel_spread"], 99.9), tr["cancel_spread"].max()
    print(f"seed {seed}: method's training answers: median {np.median(tr['cancel_spread']):.0f}, 99th pct {q99:.0f},"
          f" 99.9th {q999:.0f}, max {mx}; identical to plain's: {np.array_equal(np.sort(tr['plain']), np.sort(tr['cancel_spread']))}")
    for name, a in [("train plain", tr["plain"]), ("train method", tr["cancel_spread"]), ("train oracle", tr["oracle"])] + \
                   [(f"test {k} events", ev[k]) for k in (4, 8, 12, 16)]:
        bins = " ".join(f"{lo}-{hi if hi < 10**9 else '':<2}: {np.mean((a >= lo) & (a <= hi)):5.1%}" for lo, hi in BINS)
        print(f"  {name:15s} median {np.median(a):4.0f} | {bins} | above method's 99th pct {np.mean(a > q99):5.1%},"
              f" 99.9th {np.mean(a > q999):5.1%}, max {np.mean(a > mx):5.1%}")
