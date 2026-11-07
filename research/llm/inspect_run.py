"""Inspect fetched run outputs: recompute metrics from raw generations and show outputs by eye.

    python3 inspect_run.py OUT_DIR [--show 10]

For every run folder (metrics.json + generations.jsonl, or generations.jsonl.gz as committed):
  * recompute strict and lenient accuracy per set from generations.jsonl, independently of
    run_llm.py's code, and flag any disagreement with metrics.json;
  * check that the code hashes recorded by the run match the local files or a registered version;
  * print train statistics, leakage counts and a few raw generations for reading by eye.
"""

import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
local = {n: hashlib.sha256(open(os.path.join(HERE, n), "rb").read()).hexdigest() for n in ("problems.py", "run_llm.py")}
REGISTERED = {  # earlier registered versions, so committed results still check out after code changes
    "d4d3128bd64043ac3449b5bc04a0bc995c3cdb6aa9525236f9140697e63b61a9": "the version registered in SPEC.md",
    "5a684fec81f3bcad5b85db2e56acae8dd63440dcc2f02f2457e6b21f3c1be08a": "the version registered in SPEC.md",
    "d8cb98f513f14c3ce41ab088d3ff3023f2827b75b69a83b99d56bb8c364e325a": "the version registered in SPEC_CHAIN.md",
    "2fe0b2afe62ef404ccb163d087842dc8892b2cebb5ae34204fbf19b24e4db6b7": "the version registered in SPEC_CHAIN.md",
}

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
        known = h == local[n] or h in REGISTERED
        bad += not known
        where = "local" if h == local[n] else REGISTERED.get(h, "")
        print(f"  code {n}: {('matches ' + where) if known else 'DIFFERS from local and from every registered version'}")
    print("  train stats:", m["train_stats"] or "(no training)")
    print("  leakage (exact overlap with training):", m["leakage_exact_overlap"])
    g = os.path.join(run, "generations.jsonl")
    rows = [json.loads(line) for line in (open(g) if os.path.exists(g) else gzip.open(g + ".gz", "rt"))]
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
