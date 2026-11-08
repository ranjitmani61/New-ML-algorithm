"""Apply SPEC_MINE.md's decision rule (written before the mining runs finished).

    python3 -I summarize_mine.py MINE_ROOT FINAL_ROOT

MINE_ROOT holds the mined runs (one folder per seed with metrics.json); FINAL_ROOT holds the
final's runs (results/multi_final: plain, cancel_spread, self_improve, oracle). The primary metric
is accuracy at 12 events (training vocabulary).
"""
import glob
import json
import os
import sys

SEEDS, KEY = (9, 10, 11), "train_vocab/12"


def runs(root):
    out = {}
    for mp in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        m = json.load(open(mp))
        a = m.get("args", {})
        if a.get("family") == "multi":
            out[(a["condition"], a["seed"])] = m
    return out


mine, final = runs(sys.argv[1]), runs(sys.argv[2])
acc = lambda m, k=KEY: m["metrics"][k]["accuracy"]  # noqa: E731

print("== gate and certification")
passed = []
for s in SEEDS:
    m = mine.get(("mined", s))
    if m is None:
        print(f"seed {s}: no run")
        continue
    if "stopped" in m:
        print(f"seed {s}: stopped: {m['stopped']}  ref={m.get('ref')}")
        continue
    c, t = m["certification"], m["train_stats"]
    passed.append(s)
    print(f"seed {s}: ref {m['ref']}; edits {c['edits']}, to stage 2 {c['stage2']}, certified {c['certified']}, "
          f"exact {c['exact']}, both {c['certified_and_exact']}; precision {c['precision']:.3f}, recall {c['recall']:.3f}; "
          f"{c['generations']} generations, {c['seconds']} s")
    print(f"         certified but not exact: {c['certified_not_exact'][:10]}")
    print(f"         training labels true {t['label_correct_rate']:.4f}, valid problems {t['valid_rate']:.3f}, "
          f"events mean {t['changes_mean']:.2f}, truncated {t['truncated']}; train {m['train_seconds']} s, total {m['total_seconds']} s")

print("\n== accuracy (training vocabulary; held-out at 12)")
conds = ("plain", "cancel_spread", "self_improve", "oracle")
for s in SEEDS:
    cells = []
    for c in conds:
        f = final.get((c, s))
        cells.append(f"{c} {100 * acc(f):.1f}" if f else f"{c} -")
    m = mine.get(("mined", s))
    mined = ("mined " + " / ".join(f"{100 * acc(m, f'train_vocab/{k}'):.1f}" for k in (4, 8, 12, 16))
             + f" | held-out 12: {100 * acc(m, 'heldout_vocab/12'):.1f}") if s in passed else "mined -"
    print(f"seed {s}: at 12: " + ", ".join(cells) + f"  ||  {mined}")

print("\n== decision rule (SPEC_MINE.md)")
if len(passed) < 2:
    print("VERDICT: NO TEST (fewer than two seeds passed the gate)")
    sys.exit(0)
d, v = [], []
for s in passed:
    mi, pl, cs, si = acc(mine[("mined", s)]), acc(final[("plain", s)]), acc(final[("cancel_spread", s)]), acc(final[("self_improve", s)])
    v.append(mi - pl)
    d.append(mi - cs)
    print(f"seed {s}: mined - plain {100 * (mi - pl):+.1f}; d = mined - cancel_spread {100 * (mi - cs):+.1f}; "
          f"mined - self_improve {100 * (mi - si):+.1f}")
md = sum(d) / len(d)
print(f"mean d = {100 * md:+.2f} points over seeds {passed}")
if min(v) < 0.15:
    verdict = "MINING FAILS HERE (validity: mined - plain below 15 points on a seed)"
elif md >= 0.05 and min(d) > 0:
    verdict = "MINING BETTER"
elif md <= -0.05 and max(d) < 0:
    verdict = "HAND RULE BETTER"
else:
    verdict = "MATCHES"
print("VERDICT:", verdict)
