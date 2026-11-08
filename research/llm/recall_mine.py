"""Post-hoc (not registered): why did certification miss exact edits?  python3 -I recall_mine.py results/mine_multi"""
import glob, gzip, json, os, sys
from collections import defaultdict
import statistics as st

for d in sorted(glob.glob(os.path.join(sys.argv[1], "*_mined_multi_s*"))):
    m = json.load(open(os.path.join(d, "metrics.json")))
    ref2 = m["ref"]["2"]; thr = ref2 - 0.05
    rows = [json.loads(l) for l in gzip.open(os.path.join(d, "certification.jsonl.gz"), "rt")]
    ex = [r for r in rows if r["exact"]]
    s1_rej = [r for r in ex if r["n"] == 8]
    s2 = [r for r in ex if r["n"] > 8]
    s2_rej = [r for r in s2 if not r["certified"]]
    print(f"\n{os.path.basename(d)}: ref2 {ref2:.3f}, threshold {thr:.3f}; exact {len(ex)}: certified {sum(r['certified'] for r in ex)}, "
          f"rejected at stage 1 {len(s1_rej)}, rejected at stage 2 {len(s2_rej)}")
    print(f"  exact edits with 64 problems: mean score {st.mean(r['score'] for r in s2):.3f} (ref2 {ref2:.3f}); "
          f"pooled over all exact edits: {st.mean(r['score'] for r in ex):.3f} (biased by stage-1 stopping)")
    # breakdowns
    def key_kind(r): return r["edit"][0]
    def key_order(r): return "gain first" if r["edit"][1][0][1] == "gain" else "loss first"
    def key_amt(r): return "amount 1-3" if r["edit"][1][0][2] <= 3 else ("amount 4-6" if r["edit"][1][0][2] <= 6 else "amount 7-9")
    for name, kf in (("kind", key_kind), ("order in edit", key_order), ("amount", key_amt)):
        g = defaultdict(list)
        for r in ex: g[kf(r)].append(r)
        print("  by", name + ":", "; ".join(f"{k}: {sum(r['certified'] for r in v)}/{len(v)} certified, valid {st.mean(r['valid'] for r in v):.2f}"
                                          for k, v in sorted(g.items())))
    # non-exact edits: how close did any come?
    nx = [r for r in rows if not r["exact"] and r["n"] > 8]
    best = sorted(nx, key=lambda r: -r["score"])[:3]
    print(f"  non-exact edits reaching stage 2: {len(nx)}; top scores: " +
          ", ".join(f"{r['edit'][0]} {r['edit'][1]} score {r['score']:.3f} true keep-rate {r['inv']:.3f}" for r in best))
