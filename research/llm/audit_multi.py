"""Descriptive audit of three-person runs (not a decision criterion).

    python3 -I audit_multi.py RUN_ROOT [RUN_ROOT ...]

1. trivial shortcut rules on the test sets: the largest start amount, the last number in the
   text, the largest number in the text, the most common answer;
2. accuracy at 12 events by answer size, and the shape of the wrong answers;
3. self-improvement, per round: problems kept by the vote, their label accuracy (diagnostic, from
   the logged true answers) and each model's accuracy on the round's pool;
4. 20 raw outputs per condition at 12 events (first seed), for reading by eye.
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
BINS = ((0, 30), (30, 45), (45, 60), (60, 10**9))
CONDS = ("plain", "cancel_spread", "self_improve", "oracle")


def rows_of(run):
    g = os.path.join(run, "generations.jsonl")
    return [json.loads(line) for line in (open(g) if os.path.exists(g) else gzip.open(g + ".gz", "rt"))]


runs, metas = {}, {}
for root in sys.argv[1:]:
    for mp in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        m = json.load(open(mp))
        if m["args"].get("family") != "multi":
            continue
        k = (m["args"]["condition"], m["args"]["seed"])
        runs[k], metas[k] = rows_of(os.path.dirname(mp)), m
seeds = sorted({k[1] for k in runs})


def label(lo, hi):
    return f"{lo}-{hi - 1 if hi < 10**9 else 'max'}"


print("== 1. shortcut rules (fraction of test problems a trivial rule gets right)")
for seed in seeds:
    any_run = next(v for k, v in runs.items() if k[1] == seed)
    for s in sorted({r["set"] for r in any_run}, key=lambda x: (x.startswith("held"), int(x.split("/")[1]))):
        rs = [r for r in any_run if r["set"] == s]
        nums = [list(map(int, NUM.findall(r["text"]))) for r in rs]
        rules = {"largest start": [max(n[:3]) for n in nums], "last number": [n[-1] for n in nums],
                 "largest number": [max(n) for n in nums]}
        ans = [r["answer"] for r in rs]
        mode = collections.Counter(ans).most_common(1)[0]
        cells = "  ".join(f"{name} {sum(x == y for x, y in zip(v, ans)) / len(rs):.3f}" for name, v in rules.items())
        print(f"  seed {seed} {s:17s} {cells}  most common answer {mode[0]} ({mode[1] / len(rs):.3f})  answers {min(ans)}-{max(ans)}")

print("\n== 2. at 12 events (train vocab): accuracy by answer size; shape of the wrong answers")
for key in sorted(runs):
    rs = [r for r in runs[key] if r["set"] == "train_vocab/12"]
    cells = []
    for lo, hi in BINS:
        b = [r for r in rs if lo <= r["answer"] < hi]
        cells.append(f"{label(lo, hi)}: {sum(r['correct'] for r in b) / max(len(b), 1):.2f} (n={len(b)})")
    d = [r["parsed"] - r["answer"] for r in rs if not r["correct"] and r["parsed"] is not None]
    print(f"  {key[0]:14s} s{key[1]}  " + "  ".join(cells)
          + f"  | wrong {len(d)}: within 2 {sum(abs(x) <= 2 for x in d) / max(len(d), 1):.2f},"
            f" too low {sum(x < 0 for x in d) / max(len(d), 1):.2f}")

print("\n== 3. self-improvement rounds (kept by the vote; label accuracy is a diagnostic only)")
for key in sorted(k for k in metas if k[0] == "self_improve"):
    si = metas[key]["self_improve"]
    print(f"  seed {key[1]}: schedule {si['schedule'][:len(si['rounds'])] or '(no rounds)'}")
    for r in si["rounds"]:
        print(f"    round {r['round']} events {r['events']}: kept {r['kept']}/{r['pool']}"
              f" ({r['kept'] / r['pool']:.0%}), kept-label accuracy {r['kept_label_accuracy_diagnostic']},"
              f" per-model pool accuracy {[round(x, 3) for x in r['per_model_pool_accuracy_diagnostic']]}")
    for mi, met in sorted(si.get("final_other_models", {}).items()):
        print(f"    final model {mi}: " + " ".join(f"{s} {v['accuracy']:.3f}" for s, v in met.items()))

print(f"\n== 4. raw outputs by eye (seed {seeds[0]}, 12 events, 20 per condition; answer:output)")
idx = sorted(random.Random(0).sample(range(500), 20))
for cond in CONDS:
    if (cond, seeds[0]) in runs:
        rs = [r for r in runs[(cond, seeds[0])] if r["set"] == "train_vocab/12"]
        if len(rs) == 500:
            print(f"  {cond:14s}", " ".join(f"{rs[i]['answer']}:{rs[i]['output']!r}" for i in idx))
