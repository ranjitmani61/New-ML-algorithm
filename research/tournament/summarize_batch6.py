"""Apply batch 6's registered rules (LEDGER.md, "Batch 6", written before running) to results_batch6/.

    python3 -I summarize_batch6.py [results_batch6]

Written after 9 of the 12 jobs had finished; it only implements the rules as registered:
  proxy check (batch 4's, per seed): at 12 events, pairs fall by at least 30 points from winners
    with 0-2 events to winners with 5-6 events, and the oracle falls less. Cannot pass if pairs
    score below about 30% at 12 events.
  keep rule (split, split_pairs): mean at 12 events at least 10 points above pairs, both seeds above
    pairs; in-distribution (mean of 1-4) at least 95%, or no more than 3 points below the baseline.
"""
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_batch6")
CONDS = ["baseline", "pairs", "oracle", "split", "split_pairs", "mined"]
SEEDS = [0, 1]


def load(c, s):
    f = os.path.join(D, f"{c}_s{s}.json")
    if not os.path.exists(f):
        return None
    r = json.load(open(f))["seeds"][str(s)]
    return None if "stopped" in r else r


R = {(c, s): load(c, s) for c in CONDS for s in SEEDS}


def acc(c, s, n):
    r = R[(c, s)]
    return None if r is None else 100 * r["eval"][str(n)]


def ind(c, s):
    r = R[(c, s)]
    return None if r is None else 100 * sum(r["eval"][str(n)] for n in (1, 2, 3, 4)) / 4


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if len(xs) == len(SEEDS) else None


def f(x, w=5):
    return f"{x:{w}.1f}" if x is not None else " " * (w - 1) + "-"


print("== accuracy, % (1,000 fresh problems per length and seed)")
print("condition     1 ev  4 ev  1-4 mean (s0, s1)    8 ev  12 ev (s0, s1)      16 ev")
for c in CONDS:
    i0, i1 = ind(c, 0), ind(c, 1)
    t0, t1 = acc(c, 0, 12), acc(c, 1, 12)
    print(f"{c:12s} {f(mean([acc(c, s, 1) for s in SEEDS]))} {f(mean([acc(c, s, 4) for s in SEEDS]))} "
          f"{f(mean([i0, i1]))} ({f(i0, 4)}, {f(i1, 4)})  {f(mean([acc(c, s, 8) for s in SEEDS]))} "
          f"{f(mean([t0, t1]))} ({f(t0, 4)}, {f(t1, 4)})  {f(mean([acc(c, s, 16) for s in SEEDS]))}")

print("\n== proxy check (per seed): fall at 12 events from winners with 0-2 events to 5-6 events")
proxy = []
for s in SEEDS:
    p, o = R[("pairs", s)], R[("oracle", s)]
    if p is None or o is None:
        print(f"seed {s}: pending")
        proxy.append(None)
        continue
    fp = 100 * (p["by_winner_events_12"]["0-2"][0] - p["by_winner_events_12"]["5-6"][0])
    fo = 100 * (o["by_winner_events_12"]["0-2"][0] - o["by_winner_events_12"]["5-6"][0])
    ok = acc("pairs", s, 12) >= 30 and fp >= 30 and fo < fp
    proxy.append(ok)
    print(f"seed {s}: pairs fall {fp:.1f} points (needs >= 30), oracle fall {fo:.1f} (needs < pairs); "
          f"pairs at 12 events {acc('pairs', s, 12):.1f}% -> {'holds' if ok else 'fails'}")
print("proxy check:", "pending" if None in proxy else ("HOLDS" if all(proxy) else "FAILS"))

print("\n== keep rule (split, split_pairs vs pairs at 12 events; in-distribution vs baseline)")
for c in ("split", "split_pairs"):
    ds = [None if acc(c, s, 12) is None else acc(c, s, 12) - acc("pairs", s, 12) for s in SEEDS]
    if None in ds:
        known = [f"seed {s}: {d:+.1f}" for s, d in zip(SEEDS, ds) if d is not None]
        dead = any(d is not None and d <= 0 for d in ds)
        print(f"{c}: pending ({', '.join(known) or 'no seed yet'})" + ("; cannot be kept: a seed is not above pairs" if dead else ""))
        continue
    md = sum(ds) / len(ds)
    inds = [ind(c, s) for s in SEEDS]
    base = [ind("baseline", s) for s in SEEDS]
    ind_ok = all(i >= 95 or b - i <= 3 for i, b in zip(inds, base))
    keep = md >= 10 and min(ds) > 0 and ind_ok
    print(f"{c}: d = {', '.join(f'{d:+.1f}' for d in ds)} (mean {md:+.1f}; needs >= +10 and every seed > 0); "
          f"in-distribution {', '.join(f'{i:.1f}' for i in inds)} (baseline {', '.join(f'{b:.1f}' for b in base)}) "
          f"-> {'KEPT' if keep else 'not kept'}")

print("\n== mined (reported against pairs; not part of the decision)")
for s in SEEDS:
    r = R[("mined", s)]
    if r is None:
        print(f"seed {s}: pending")
        continue
    c = r["certification"]
    print(f"seed {s}: M0 gate {100 * r['m0_in_dist']:.1f}%; certified {c['certified']} of {c['edits']} "
          f"(exact {c['certified_and_exact']} of {c['exact']}): precision {c['precision']:.3f}, recall {c['recall']:.3f}; "
          f"closest wrong score {c['closest_wrong'][0]:.3f} vs threshold {r['ref']['2'] - 0.05:.3f}; "
          f"12 events {acc('mined', s, 12):.1f} vs pairs {acc('pairs', s, 12):.1f} (d {acc('mined', s, 12) - acc('pairs', s, 12):+.1f}); "
          f"train labels true {r['train_probe']['label_true']:.3f}, valid {r['train_probe']['valid']:.3f}")

print("\n== by winner's events at 12 events, mean of seeds (0-2 / 3-4 / 5-6 / 7+)")
for c in CONDS:
    if any(R[(c, s)] is None for s in SEEDS):
        continue
    row = [100 * sum(R[(c, s)]["by_winner_events_12"][b][0] for s in SEEDS) / len(SEEDS) for b in ("0-2", "3-4", "5-6", "7+")]
    print(f"{c:12s} " + " / ".join(f"{x:.1f}" for x in row))

print("\n== seconds per job:", {f"{c}_s{s}": R[(c, s)]["seconds"] for c in CONDS for s in SEEDS if R[(c, s)] is not None})
