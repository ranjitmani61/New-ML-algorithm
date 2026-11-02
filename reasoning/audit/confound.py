"""Is the clean waypoint signal more than vote share? Compare per-attempt pass-through rates at equal vote share."""
import sys, pickle, collections, re, gzip, json, glob, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_detector_lib import results
dump = pickle.load(open(sys.argv[1], "rb"))["out"]
texts = {}
for p in sorted(glob.glob(os.path.join(sys.argv[2], "part*.jsonl.gz"))):
    for line in gzip.open(p, "rt"):
        it = json.loads(line); texts[it["id"]] = [s["text"] for s in it["samples"]]
bins = collections.defaultdict(lambda: {"R": [0, 0, 0], "W": [0, 0, 0]})
for qid, (keys, gold, _, pv) in dump.items():
    c = collections.Counter(k for k in keys if k)
    if not c: continue
    a, v = c.most_common(1)[0]
    if not re.fullmatch(r"-?\d+(/\d+)?", a): continue  # detector only sees numbers
    res = [results(t) for t in texts[qid]]
    others = [r for b, r in zip(keys, res) if b != a]
    L = sum(a in r for r in others)
    share = v / len(keys)
    b = "<35%" if share < .35 else "35-50%" if share < .5 else "50-70%" if share < .7 else ">=70%"
    g = "R" if a == gold else "W"
    bins[b][g][0] += 1; bins[b][g][1] += L; bins[b][g][2] += len(others)
print("vote share of leader | right: n, pass-through per other attempt | wrong: n, rate")
for b in ("<35%", "35-50%", "50-70%", ">=70%"):
    R, W = bins[b]["R"], bins[b]["W"]
    rate = lambda x: x[1] / x[2] if x[2] else float("nan")
    print(f"   {b:7s} right n={R[0]:4d} rate={rate(R):.3f} | wrong n={W[0]:4d} rate={rate(W):.3f}")
