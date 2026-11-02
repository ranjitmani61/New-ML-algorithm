"""Apply the ledger's keep rule to tournament results.

    python3 judge.py RESULTS_DIR

Keep rule: mean 8-term accuracy >= baseline + 0.10, every seed above the baseline's same seed,
and in-distribution (mean of 1-4 terms) >= 0.95 on every seed.
"""

import glob
import json
import os
import sys

import numpy as np

d = sys.argv[1]
res = {}
for path in sorted(glob.glob(os.path.join(d, "*.json"))):
    r = json.load(open(path))
    res[r["idea"]] = r


def rows(r, key="eval"):
    out = {}
    for seed, row in r["seeds"].items():
        if key in row:
            e = {int(k): v for k, v in row[key].items()}
            out[seed] = e
    return out


base = rows(res["baseline"])
b8 = {s: e[8] for s, e in base.items()}
print(f"{'idea':22s} {'ID(1-4)':>8s} {'5':>6s} {'6':>6s} {'7':>6s} {'8':>6s}  {'d8 vs base':>10s}  decision")
for name, r in res.items():
    variants = [("eval", name)]
    for key in ("eval_alpha0.5", "eval_alpha1.0", "eval_theta3"):
        if any(key in row for row in r["seeds"].values()):
            variants.append((key, f"{name}:{key.replace('eval_', '')}"))
    for key, label in variants:
        e = rows(r, key)
        if not e:
            continue
        seeds = sorted(e)
        idm = [np.mean([e[s][n] for n in (1, 2, 3, 4)]) for s in seeds]
        acc = {n: np.mean([e[s][n] for s in seeds]) for n in (5, 6, 7, 8)}
        d8 = [e[s][8] - b8[s] for s in seeds if s in b8]
        keep = (name != "baseline" and np.mean(d8) >= 0.10 and all(x > 0 for x in d8) and min(idm) >= 0.95)
        dec = "baseline" if name == "baseline" else ("KEEP" if keep else "discard")
        print(f"{label:22s} {np.mean(idm):8.3f} {acc[5]:6.3f} {acc[6]:6.3f} {acc[7]:6.3f} {acc[8]:6.3f}  "
              f"{np.mean(d8):+10.3f}  {dec}   per-seed 8-term: {[round(e[s][8], 3) for s in seeds]}")
