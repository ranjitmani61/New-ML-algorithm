"""Gates for the three-person test (PLAN_MULTI.md, SPEC_MULTI.md), applied mechanically.

    python3 summarize_multi.py --gate pilot RUN_ROOT [RUN_ROOT ...]
    python3 summarize_multi.py --gate final RUN_ROOT [RUN_ROOT ...]

Reads every metrics.json under the roots (run_multi.py outputs) and checks that the runs used the
code registered for the gate.
"""
import argparse
import glob
import json
import os

import numpy as np

PILOT_CODE = {"multi.py": "43c22ac06782e270facbb8fd6463ea44bd62100d480114776dee310787a9e79d",
              "run_multi.py": "fbbc6d3eba216dc7393af1b4171d4f299897545eae120869b57ff9d10f1ab3e4"}
REGISTERED = {"pilot": PILOT_CODE, "final": PILOT_CODE}  # final: set when SPEC_MULTI.md is registered
SETS = ("train_vocab/4", "train_vocab/6", "train_vocab/8", "train_vocab/12", "train_vocab/16", "heldout_vocab/12")

ap = argparse.ArgumentParser()
ap.add_argument("--gate", choices=("pilot", "final"), required=True)
ap.add_argument("roots", nargs="+")
a = ap.parse_args()
runs = {}
for root in a.roots:
    for mp in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        m = json.load(open(mp))
        if m["args"].get("family") != "multi":
            continue
        key = (m["args"]["condition"], m["args"]["seed"])
        assert key not in runs, f"duplicate run {key}"
        runs[key] = m
reg = REGISTERED.get(a.gate)
bad = [k for k, m in runs.items() if reg is None or any(m["code_sha256"].get(f) != h for f, h in reg.items())]
print(f"runs: {len(runs)} | code hashes differ from the code registered for this gate: {bad or 'none'}")
print(f"{'condition':14s} seed " + " ".join(f"{s.replace('train_vocab/', '').replace('heldout_vocab/', 'held'):>7s}" for s in SETS)
      + "  parse-fail  train_s")
for (c, s), m in sorted(runs.items()):
    cells = " ".join(f"{m['metrics'][x]['accuracy']:7.3f}" if x in m["metrics"] else f"{'-':>7s}" for x in SETS)
    pf = sum(v["parse_failures"] for v in m["metrics"].values())
    print(f"{c:14s} {s:4d} {cells}  {pf:10d}  {m['train_seconds']:7d}")


def acc(c, s, k="train_vocab/12"):
    return runs[(c, s)]["metrics"][k]["accuracy"]


if a.gate == "pilot":
    need = [("plain", 0), ("oracle", 0), ("self_improve", 0)]
    missing = [k for k in need if k not in runs]
    if missing:
        raise SystemExit(f"missing runs: {missing}")
    si = runs[("self_improve", 0)]["self_improve"]
    print("\nself_improve round-0 models (each trained like plain; model 0 exactly like plain):")
    for mi, met in sorted(si["round0"].items()):
        print(f"  model {mi}: " + " ".join(f"{x.split('/')[1] if x.startswith('train') else 'held12'} {met[x]['accuracy']:.3f}"
                                         for x in SETS if x in met))
    d0 = {x: si["round0"]["0"][x]["accuracy"] - acc("plain", 0, x) for x in runs[("plain", 0)]["metrics"]}
    print(f"  model 0 minus the separate plain run (GPU nondeterminism only): "
          f"max |difference| {max(abs(v) for v in d0.values()):.3f}")
    print("vote probe (fresh problems; all 3 models must agree):")
    for k, p in sorted(si["probe"].items(), key=lambda kv: int(kv[0])):
        print(f"  {k} events: per-model accuracy {[round(x, 3) for x in p['per_model_accuracy']]}, kept {p['kept']}/{p['n']}"
              f" ({p['kept'] / p['n']:.0%}), kept-label accuracy {p['kept_label_accuracy']}")
    ga = acc("plain", 0, "train_vocab/4")
    gb = acc("oracle", 0) - acc("plain", 0)
    p6 = si["probe"]["6"]
    two = p6["kept"] / p6["n"] >= 0.20 and (p6["kept_label_accuracy"] or 0) >= 0.90
    print(f"\ngate A, learnability: plain at 4 events {ga:.3f} (needs 0.90) -> {'PASS' if ga >= 0.90 else 'FAIL'}")
    print(f"gate B, headroom: oracle - plain at 12 events {gb:+.3f} (needs +0.30) -> {'PASS' if gb >= 0.30 else 'FAIL'}")
    print(f"self-improvement schedule: vote at 6 events keeps {p6['kept'] / p6['n']:.0%} (needs 20%) with label accuracy"
          f" {p6['kept_label_accuracy']} (needs 0.90) -> {'+2' if two else '+1'} events per round")
    print("training seconds:", {f"{c} s{s}": m["train_seconds"] for (c, s), m in sorted(runs.items())})
else:
    seeds = (9, 10, 11)
    conds = ("plain", "cancel_spread", "self_improve", "oracle")
    missing = [(c, s) for c in conds for s in seeds if (c, s) not in runs]
    if missing:
        raise SystemExit(f"missing runs: {missing}")
    v = [acc("cancel_spread", s) - acc("plain", s) for s in seeds]
    d = [acc("cancel_spread", s) - acc("self_improve", s) for s in seeds]
    md = float(np.mean(d))
    print(f"\nvalidity: cancel_spread - plain at 12 events, per seed {[round(x, 3) for x in v]} (needs >= 0.15 on every seed)"
          f" -> {'holds' if min(v) >= 0.15 else 'FAILS'}")
    print(f"decision: cancel_spread - self_improve (model 0) at 12 events, per seed {[round(x, 3) for x in d]}, mean {md:+.3f}")
    print("  rule: METHOD BETTER if mean >= +5 points and > 0 on every seed; SELF-IMPROVEMENT BETTER if mean <= -5"
          " points and < 0 on every seed; otherwise NO CLEAR DIFFERENCE")
    if min(v) < 0.15:
        verdict = "METHOD FAILS HERE (it does not beat plain by 15 points on every seed)"
    elif md >= 0.05 and min(d) > 0:
        verdict = "METHOD BETTER"
    elif md <= -0.05 and max(d) < 0:
        verdict = "SELF-IMPROVEMENT BETTER"
    else:
        lead = (" (the method ahead on every seed, below the margin)" if min(d) > 0 else
                " (self-improvement ahead on every seed, below the margin)" if max(d) < 0 else "")
        verdict = "NO CLEAR DIFFERENCE" + lead
    means = {c: {k: float(np.mean([acc(c, s, k) for s in seeds])) for k in SETS if k in runs[(c, seeds[0])]["metrics"]}
             for c in conds}
    print("secondary, means:", {c: {k: round(x, 3) for k, x in m.items()} for c, m in means.items()})
    print(f"secondary, oracle - cancel_spread at 12: {means['oracle']['train_vocab/12'] - means['cancel_spread']['train_vocab/12']:+.3f}")
    other = [runs[("self_improve", s)]["self_improve"]["final_other_models"][m]["train_vocab/12"]["accuracy"]
             for s in seeds for m in ("1", "2")]
    print(f"secondary, self_improve models 1-2 at 12 (mean of 6): {np.mean(other):.3f}")
    for c in ("cancel_spread", "self_improve"):
        print(f"secondary, {c} cost at 4 events (plain - {c}): "
              f"{means['plain']['train_vocab/4'] - means[c]['train_vocab/4']:+.3f}")
    print("secondary, training seconds (mean per run):",
          {c: round(float(np.mean([runs[(c, s)]["train_seconds"] for s in seeds]))) for c in conds})
    print(f"verdict: {verdict}")
