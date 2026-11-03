"""Inspect fetched run outputs: recompute metrics from raw generations and show outputs by eye.

    python3 inspect_run.py OUT_DIR [--show 10]

For every run folder (metrics.json + generations.jsonl):
  * recompute strict and lenient accuracy per set from generations.jsonl, independently of
    run_llm.py's code, and flag any disagreement with metrics.json;
  * check that the code hashes recorded by the run match the local files;
  * print train statistics, leakage counts and a few raw generations for reading by eye.
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
local = {n: hashlib.sha256(open(os.path.join(HERE, n), "rb").read()).hexdigest() for n in ("problems.py", "run_llm.py")}

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--show", type=int, default=10)
a = ap.parse_args()
bad = 0
for mpath in sorted(glob.glob(os.path.join(a.out, "**", "metrics.json"), recursive=True)):
    run = os.path.dirname(mpath)
    m = json.load(open(mpath))
    print(f"\n=== {os.path.basename(run)}  device={m['device']} torch={m['torch']}")
    for n, h in (("problems.py", m["generator_sha256"]), ("run_llm.py", m["script_sha256"])):
        same = h == local[n]
        bad += not same
        print(f"  code {n}: {'matches local' if same else 'DIFFERS from local'}")
    print("  train stats:", m["train_stats"] or "(no training)")
    print("  leakage (exact overlap with training):", m["leakage_exact_overlap"])
    rows = [json.loads(line) for line in open(os.path.join(run, "generations.jsonl"))]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["set"]].append(r)
    for s, rs in by.items():
        strict = sum(re.match(r"^\s*(-?\d+)", r["output"]) is not None
                     and int(re.match(r"^\s*(-?\d+)", r["output"]).group(1)) == r["answer"] for r in rs) / len(rs)
        first = [re.search(r"-?\d+", r["output"]) for r in rs]
        lenient = sum(f is not None and int(f.group(0)) == r["answer"] for f, r in zip(first, rs)) / len(rs)
        rep = m["metrics"][s]
        agree = abs(strict - rep["accuracy"]) < 1e-9 and abs(lenient - rep["accuracy_lenient"]) < 1e-9
        bad += not agree
        print(f"  {s:18s} n={len(rs)} strict {strict:.3f} lenient {lenient:.3f} parse failures {rep['parse_failures']}"
              f"  {'(matches metrics.json)' if agree else '(DISAGREES with metrics.json)'}")
    for r in rows[: a.show]:
        print(f"    [{r['set']}] answer {r['answer']:>4}  output {r['output']!r}")
print(f"\n{'all checks consistent' if bad == 0 else f'{bad} inconsistencies found'}")
