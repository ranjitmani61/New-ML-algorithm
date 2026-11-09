"""Apply SPEC_REPLICATE.md's decision rule (written before the replication ran).

    python3 -I summarize_replicate.py ROOT [ROOT ...] [--seeds 12 13 14]

Each ROOT is searched for metrics.json files of the three-person test (family "multi"). The rule is
SPEC_MINE.md's, unchanged: gate, validity (mined - plain >= 15 on every seed that passed the gate),
then d = mined - cancel_spread at 12 events (training vocabulary). REPLICATED if the verdict is
MATCHES or MINING BETTER; NOT REPLICATED if HAND RULE BETTER or MINING FAILS HERE.
Run on seeds 9-11 (results/mine_multi results/multi_final) it reproduces SPEC_MINE.md's result.
"""
import argparse
import glob
import gzip
import json
import os

KEY = "train_vocab/12"
ap = argparse.ArgumentParser()
ap.add_argument("roots", nargs="+")
ap.add_argument("--seeds", nargs="+", type=int, default=[12, 13, 14])
a = ap.parse_args()

runs = {}
for root in a.roots:
    for mp in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        m = json.load(open(mp))
        g = m.get("args", {})
        if g.get("family") == "multi":
            key = (g["condition"], g["seed"])
            if key in runs:
                raise SystemExit(f"two runs for {key}: {runs[key]['_path']} and {mp}")
            m["_path"] = mp
            runs[key] = m
acc = lambda m, k=KEY: m["metrics"][k]["accuracy"]  # noqa: E731

print("== gate and certification")
passed, no_rules, infra = [], [], []  # gate passed and trained; gate passed but nothing certified; run missing
for s in a.seeds:
    m = runs.get(("mined", s))
    stop = m.get("stopped", "") if m else "no mined run"
    if stop:
        print(f"seed {s}: stopped: {stop}  ref={m.get('ref') if m else None}")
        if stop.startswith("gate failed"):
            continue
        (no_rules if stop == "no edit certified" else infra).append(s)
        continue
    passed.append(s)
    c, t = m["certification"], m["train_stats"]
    rows = [json.loads(x) for x in gzip.open(os.path.join(os.path.dirname(m["_path"]), "certification.jsonl.gz"), "rt")]
    thr = m["ref"]["2"] - m["args"]["cert_margin"]
    wrong = [r for r in rows if not r["exact"] and len(r["edit"][1]) == 2]
    best = max(wrong, key=lambda r: r["score"])
    print(f"seed {s}: M0 {m['m0_adapter_sha256'][:12]}, ref {m['ref']}; certified {c['certified']} (exact {c['certified_and_exact']}) "
          f"of {c['edits']}; precision {c['precision']:.3f}, recall {c['recall']:.3f}")
    print(f"         closest wrong two-event edit: score {best['score']:.3f} vs threshold {thr:.3f} "
          f"(true keep-rate {best['inv']:.3f}); certified but not exact: {c['certified_not_exact'][:10]}")
    print(f"         training labels true {t['label_correct_rate']:.4f}, valid problems {t['valid_rate']:.3f}; "
          f"certification {c['seconds']} s, training {m['train_seconds']} s")

print("\n== accuracy, % (training vocabulary at 4/8/12/16 events | held-out vocabulary at 12)")
for s in a.seeds:
    for cond in ("plain", "cancel_spread", "mined"):
        m = runs.get((cond, s))
        if m is None or "stopped" in m:
            print(f"seed {s} {cond:13s} -")
            continue
        print(f"seed {s} {cond:13s} " + " ".join(f"{100 * acc(m, f'train_vocab/{k}'):5.1f}" for k in (4, 8, 12, 16))
              + f" | {100 * acc(m, 'heldout_vocab/12'):5.1f}   train {m.get('train_seconds')} s")

print("\n== decision rule (SPEC_MINE.md, unchanged)")
missing = [(c, s) for s in passed + no_rules for c in ("plain", "cancel_spread") if (c, s) not in runs]
if infra or missing:
    print(f"VERDICT: INCOMPLETE (infrastructure; rerun once): mined seeds {infra}, missing runs {missing}")
    raise SystemExit(0)
if len(passed) + len(no_rules) < 2:
    print("VERDICT: NO TEST (fewer than two seeds passed the gate)")
    raise SystemExit(0)
if no_rules:
    print(f"VERDICT: MINING FAILS HERE (M0 passed the gate but no edit was certified on seeds {no_rules})")
    print("REPLICATION: NOT REPLICATED")
    raise SystemExit(0)
d, v = [], []
for s in passed:
    mi, pl, cs = acc(runs[("mined", s)]), acc(runs[("plain", s)]), acc(runs[("cancel_spread", s)])
    v.append(mi - pl)
    d.append(mi - cs)
    print(f"seed {s}: mined {100 * mi:.1f}, cancel_spread {100 * cs:.1f}, plain {100 * pl:.1f}; "
          f"mined - plain {100 * (mi - pl):+.1f}; d = mined - cancel_spread {100 * (mi - cs):+.1f}")
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
print("REPLICATION:", "REPLICATED" if verdict in ("MATCHES", "MINING BETTER") else "NOT REPLICATED")
