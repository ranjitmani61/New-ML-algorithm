"""Descriptive audit of fetched or committed runs (not a pass criterion; written after discovery).

    python3 -I audit_runs.py RUN_ROOT [RUN_ROOT ...]

For the additive family:
  1. trivial shortcut rules on the test sets (the start amount, the last number in the text, the
     most common answer), to check that no shortcut explains a condition's accuracy;
  2. accuracy at 12 events by answer size, and the shape of the errors (within 2 of the answer,
     too low);
  3. 20 raw outputs per condition at 12 events (first seed), for reading by eye.
Reads generations.jsonl or generations.jsonl.gz.
"""
import collections
import glob
import gzip
import json
import os
import random
import re
import sys

NUM = re.compile(r"\d+")
BINS = ((0, 30), (30, 60), (60, 90), (90, 10**9))
CONDS = ("plain", "irrelevant", "inactive", "cancel_spread", "cancel_adjacent", "oracle", "chain")


def rows_of(run):
    g = os.path.join(run, "generations.jsonl")
    return [json.loads(line) for line in (open(g) if os.path.exists(g) else gzip.open(g + ".gz", "rt"))]


runs = {}
for root in sys.argv[1:]:
    for mp in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        m = json.load(open(mp))
        runs[(m["args"]["condition"], m["args"]["family"], m["args"]["seed"])] = rows_of(os.path.dirname(mp))
seeds = sorted({k[2] for k in runs if k[1] == "additive"})


def label(lo, hi):
    return f"{lo}-{hi if hi < 10**9 else 'max'}"


print("== 1. shortcut rules (additive; fraction of test problems a trivial rule gets right)")
for seed in seeds:
    any_run = next(v for k, v in runs.items() if k[1] == "additive" and k[2] == seed)
    for s in ("train_vocab/4", "train_vocab/12", "train_vocab/16"):
        rs = [r for r in any_run if r["set"] == s]
        nums = [list(map(int, NUM.findall(r["text"]))) for r in rs]
        start = sum(n[0] == r["answer"] for n, r in zip(nums, rs)) / len(rs)
        last = sum(n[-1] == r["answer"] for n, r in zip(nums, rs)) / len(rs)
        ans = [r["answer"] for r in rs]
        mode = collections.Counter(ans).most_common(1)[0]
        print(f"  seed {seed} {s:15s} start amount {start:.3f}  last number {last:.3f}  "
              f"most common answer {mode[0]} ({mode[1] / len(rs):.3f})  answers {min(ans)}-{max(ans)}")

print("\n== 2. at 12 events (train vocab): accuracy by answer size; shape of the wrong answers")
for key in sorted(k for k in runs if k[1] == "additive"):
    rs = [r for r in runs[key] if r["set"] == "train_vocab/12"]
    cells = []
    for lo, hi in BINS:
        b = [r for r in rs if lo <= r["answer"] < hi]
        cells.append(f"{label(lo, hi)}: {sum(r['correct'] for r in b) / max(len(b), 1):.2f} (n={len(b)})")
    d = [r["parsed"] - r["answer"] for r in rs if not r["correct"] and r["parsed"] is not None]
    print(f"  {key[0]:16s} s{key[2]}  " + "  ".join(cells)
          + f"  | wrong {len(d)}: within 2 {sum(abs(x) <= 2 for x in d) / max(len(d), 1):.2f},"
            f" too low {sum(x < 0 for x in d) / max(len(d), 1):.2f}")

print(f"\n== 3. raw outputs by eye (seed {seeds[0]}, 12 events, 20 per condition; answer:output)")
idx = sorted(random.Random(0).sample(range(500), 20))
for cond in CONDS:
    if (cond, "additive", seeds[0]) in runs:
        rs = [r for r in runs[(cond, "additive", seeds[0])] if r["set"] == "train_vocab/12"]
        print(f"  {cond:16s}", " ".join(f"{rs[i]['answer']}:{rs[i]['output']!r}" for i in idx))
