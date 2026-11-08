"""Post-hoc diagnosis of the three-person final (written after its results; descriptive only).

    python3 -I diagnose_multi.py results/multi_final

Rebuilds the 12-event test problems of seeds 9-11 with the registered generator, checks that they
are exactly the logged ones, and asks where each condition fails:
  1. accuracy by the structure of the problem (margin between the top two, lead changes, how many
     events touched the winner, the winner's net change);
  2. what the wrong answers are (another person's final amount, the winner's amount before some
     later event, off by one of the winner's events, other).
"""
import collections
import glob
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_multi as RM  # noqa: E402

SEEDS, K, SET = (9, 10, 11), 12, "train_vocab/12"
CONDS = ("plain", "cancel_spread", "self_improve", "oracle")

rows = {}
for root in sys.argv[1:]:
    for mp in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        m = json.load(open(mp))
        if m["args"].get("family") != "multi":
            continue
        g = os.path.join(os.path.dirname(mp), "generations.jsonl")
        rs = [json.loads(line) for line in (open(g) if os.path.exists(g) else gzip.open(g + ".gz", "rt"))]
        rows[(m["args"]["condition"], m["args"]["seed"])] = [r for r in rs if r["set"] == SET]

feats = []  # one dict per (seed, problem)
for s in SEEDS:
    probs = RM.build_eval(s, [K], 500)[K]
    for c in CONDS:
        assert [r["text"] for r in rows[(c, s)]] == [p.text() for p in probs], (c, s)
    for i, p in enumerate(probs):
        traj = [list(p.starts)] + p.amounts()
        fin = traj[-1]
        top = sorted(fin, reverse=True)
        winners = [j for j in range(3) if fin[j] == top[0]]
        w = winners[0]
        leaders = [max(range(3), key=lambda j: (x[j], -j)) for x in traj]
        ev_w = [e for e in p.events if e.who == w]
        f = {"seed": s, "i": i, "answer": p.answer(), "margin": top[0] - top[1],
             "lead_changes": sum(a != b for a, b in zip(leaders, leaders[1:])),
             "winner_events": len(ev_w), "winner_net": fin[w] - p.starts[w],
             "start_leader_wins": leaders[0] in winners, "tie": len(winners) > 1}
        for c in CONDS:
            r = rows[(c, s)][i]
            f[c] = r["correct"]
            pred = r["parsed"]
            if r["correct"]:
                kind = "right"
            elif pred is None:
                kind = "no number"
            elif pred in [fin[j] for j in range(3) if j not in winners]:
                kind = "another person's final"
            elif pred in [x[w] for x in traj[:-1]]:
                kind = "winner's amount before a later event"
            elif any(abs(pred - fin[w]) == e.n for e in ev_w):
                kind = "off by one of the winner's events"
            else:
                kind = "other"
            f[c + "_err"] = kind
        feats.append(f)

print(f"== {len(feats)} problems at {K} events (seeds {SEEDS}); test problems rebuilt and matched exactly")


def table(title, key, buckets):
    print(f"\n== accuracy by {title}")
    print(f"  {'bucket':22s} {'n':>5s} " + " ".join(f"{c:>14s}" for c in CONDS))
    for name, test in buckets:
        b = [f for f in feats if test(f[key])]
        if b:
            print(f"  {name:22s} {len(b):5d} " + " ".join(f"{sum(f[c] for f in b) / len(b):14.3f}" for c in CONDS))


table("margin between the top two final amounts", "margin",
      [("0 (tie)", lambda v: v == 0), ("1-2", lambda v: 1 <= v <= 2), ("3-5", lambda v: 3 <= v <= 5),
       ("6-10", lambda v: 6 <= v <= 10), ("11+", lambda v: v >= 11)])
table("lead changes during the story", "lead_changes",
      [("0", lambda v: v == 0), ("1", lambda v: v == 1), ("2", lambda v: v == 2), ("3+", lambda v: v >= 3)])
table("events that touched the winner", "winner_events",
      [("0-2", lambda v: v <= 2), ("3-4", lambda v: 3 <= v <= 4), ("5-6", lambda v: 5 <= v <= 6),
       ("7+", lambda v: v >= 7)])
table("the winner's net change", "winner_net",
      [("-10 or less", lambda v: v <= -10), ("-9 to -1", lambda v: -9 <= v <= -1), ("0", lambda v: v == 0),
       ("1-9", lambda v: 1 <= v <= 9), ("10-19", lambda v: 10 <= v <= 19), ("20+", lambda v: v >= 20)])

print("\n== what the wrong answers are (share of each condition's wrong answers)")
for c in CONDS:
    errs = collections.Counter(f[c + "_err"] for f in feats if not f[c])
    n = sum(errs.values())
    print(f"  {c:14s} wrong {n:4d}: " + "; ".join(f"{k} {v / n:.2f}" for k, v in errs.most_common()))
