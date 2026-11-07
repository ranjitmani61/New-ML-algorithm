"""Tabulate fetched language-model runs and apply the gates written in SPEC_DRAFT.md.

    python3 summarize.py --gate pilot|discovery|final DIR [DIR ...]

Reads every DIR/**/metrics.json. Refuses to judge (prints INCOMPLETE) if a run the gate needs is
missing, and flags any run whose code hashes differ from the local files.
"""

import argparse
import glob
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = {n: hashlib.sha256(open(os.path.join(HERE, n), "rb").read()).hexdigest() for n in ("problems.py", "run_llm.py")}
SETS = ("train_vocab/4", "train_vocab/8", "train_vocab/12", "train_vocab/16", "heldout_vocab/12")

ap = argparse.ArgumentParser()
ap.add_argument("--gate", choices=("pilot", "discovery", "final"), required=True)
ap.add_argument("dirs", nargs="+")
a = ap.parse_args()

runs, hash_issues = {}, []
for d in a.dirs:
    for path in glob.glob(os.path.join(d, "**", "metrics.json"), recursive=True):
        m = json.load(open(path))
        key = (m["args"]["condition"], m["args"]["family"], m["args"]["seed"])
        if key in runs:
            raise SystemExit(f"duplicate run {key}: {path}")
        runs[key] = m
        if m["generator_sha256"] != LOCAL["problems.py"] or m["script_sha256"] != LOCAL["run_llm.py"]:
            hash_issues.append(key)

print("runs:", len(runs), "| code hashes differ from local for:", hash_issues or "none")
print(f"{'condition':16s} {'family':9s} {'seed':>4s}", *(f"{s.split('/')[1]:>6s}" for s in SETS[:4]), " held12  parse-fail  steps")
for key in sorted(runs):
    m = runs[key]["metrics"]
    cells = [f"{m[s]['accuracy']:6.3f}" if s in m else "     -" for s in SETS]
    pf = sum(v["parse_failures"] for v in m.values())
    print(f"{key[0]:16s} {key[1]:9s} {key[2]:4d}", *cells[:4], f" {cells[4]}  {pf:10d}  {runs[key]['args']['steps']}")


def acc(cond, seed, s="train_vocab/12", family="additive"):
    return runs[(cond, family, seed)]["metrics"][s]["accuracy"]


def need(keys):
    missing = [k for k in keys if k not in runs]
    if missing:
        print("\nverdict: INCOMPLETE, missing", missing)
        raise SystemExit(0)


print()
if a.gate == "pilot":
    need([("plain", "additive", 0), ("oracle", "additive", 0)])
    p4, p12, o12 = acc("plain", 0, "train_vocab/4"), acc("plain", 0), acc("oracle", 0)
    g1, g2 = p4 >= 0.90, o12 - p12 >= 0.20
    print(f"pilot gate 1: plain at 4 events >= 90%: {p4:.3f} -> {'PASS' if g1 else 'FAIL'}")
    print(f"pilot gate 2: oracle - plain at 12 events >= 20 points: {o12:.3f} - {p12:.3f} = {o12 - p12:+.3f} -> {'PASS' if g2 else 'FAIL'}")
    print("verdict:", "PASS (proceed to discovery)" if g1 and g2 else "FAIL (apply the pre-set adjustment once, or stop)")
elif a.gate == "discovery":
    seeds = (1, 2)
    need([(c, "additive", s) for c in ("plain", "irrelevant", "inactive", "cancel_spread") for s in seeds])
    gains = [acc("cancel_spread", s) - acc("plain", s) for s in seeds]
    ctrl = {c: np.mean([acc(c, s) for s in seeds]) for c in ("irrelevant", "inactive")}
    best = max(ctrl, key=ctrl.get)
    margin = np.mean([acc("cancel_spread", s) for s in seeds]) - ctrl[best]
    c1, c2 = all(g >= 0.15 for g in gains), margin >= 0.10
    print(f"go/no-go 1: cancel_spread - plain at 12 events >= 15 points on both seeds: {[round(g, 3) for g in gains]} -> {'PASS' if c1 else 'FAIL'}")
    print(f"go/no-go 2: cancel_spread - better control ({best}) >= 10 points on average: {margin:+.3f} -> {'PASS' if c2 else 'FAIL'}")
    print("verdict:", "GO (register and run the final on seeds 3-5)" if c1 and c2 else "NO-GO (stop; report the negative result)")
else:
    seeds = (3, 4, 5)
    need([(c, "additive", s) for c in ("plain", "irrelevant", "inactive", "cancel_spread", "oracle") for s in seeds])
    gains = [acc("cancel_spread", s) - acc("plain", s) for s in seeds]
    ctrl = {c: np.mean([acc(c, s) for s in seeds]) for c in ("irrelevant", "inactive")}
    best = max(ctrl, key=ctrl.get)
    cs = np.mean([acc("cancel_spread", s) for s in seeds])
    pl = np.mean([acc("plain", s) for s in seeds])
    orc = np.mean([acc("oracle", s) for s in seeds])
    share = (cs - pl) / (orc - pl) if orc > pl else float("nan")
    cost = np.mean([acc("plain", s, "train_vocab/4") - acc("cancel_spread", s, "train_vocab/4") for s in seeds])
    crit = [all(g >= 0.15 for g in gains), cs - ctrl[best] >= 0.10, share >= 0.50, cost <= 0.03]
    print(f"1. cancel_spread - plain at 12 >= 15 points on every seed: {[round(g, 3) for g in gains]} -> {'PASS' if crit[0] else 'FAIL'}")
    print(f"2. cancel_spread - better control ({best}) >= 10 points: {cs - ctrl[best]:+.3f} -> {'PASS' if crit[1] else 'FAIL'}")
    print(f"3. share of the oracle's gain >= 50%: {share:.1%} -> {'PASS' if crit[2] else 'FAIL'}")
    print(f"4. in-distribution cost at 4 events <= 3 points: {cost:+.3f} -> {'PASS' if crit[3] else 'FAIL'}")
    print("verdict:", "PASS" if all(crit) else "FAIL")
