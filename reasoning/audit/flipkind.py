import sys, pickle, collections, re
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
import selection
from types import SimpleNamespace as NS
dump = pickle.load(open(sys.argv[1], "rb"))["out"]
def kind(k):
    if re.fullmatch(r"-?\d+", k): return "int>=10" if abs(int(k)) >= 10 else "int<10"
    if re.fullmatch(r"-?\d+/\d+", k): return "fraction"
    return "other"
flips = collections.Counter()
for qid, (keys, gold, mentions, pv) in dump.items():
    tr = [NS(answer=k, mentions=m) for k, m in zip(keys, mentions)]
    cnt = collections.Counter(k for k in keys if k)
    if not cnt: continue
    first = {}
    for i, k in enumerate(keys):
        if k: first.setdefault(k, i)
    sc = max(cnt, key=lambda a: (cnt[a], -first[a]))
    new = selection.terminal_ratio_vote(tr, pv, 2.0)[0]
    if sc != new:
        flips[(kind(sc), ("R" if sc == gold else "W") + "->" + ("R" if new == gold else "W"))] += 1
for t in ("int<10", "int>=10", "fraction", "other"):
    print(f"   majority {t:9s}", {f: flips[(t, f)] for f in ("W->R", "R->W", "W->W")})
