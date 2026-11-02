"""Core-idea attack: does the waypoint signal survive a detector that only counts computed results ('= A')?"""
import sys, pickle, collections, re, gzip, json, glob, os, statistics
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
import benchmarks as bm
dump = pickle.load(open(sys.argv[1], "rb"))["out"]
texts = {}
for p in sorted(glob.glob(os.path.join(sys.argv[2], "part*.jsonl.gz"))):
    for line in gzip.open(p, "rt"):
        it = json.loads(line); texts[it["id"]] = [s["text"] for s in it["samples"]]
# value right after '=' and ending the expression: not followed by digit, operator, letter, '\', '^', '_', '('
RESULT = re.compile(r"=\s*\$?\s*(-?(?:\\[dt]?frac\{-?\d+\}\{\d+\}|\d+(?:\.\d+)?(?:\s*/\s*\d+)?))(?!\s*[\d.,]*\d)(?!\s*[-+*/^_(\\a-zA-Z{])")
def results(t):
    out = set()
    for m in RESULT.finditer(t):
        v = bm.normalize(m.group(1))
        if v: out.add(v)
    return out
def kind(k):
    if re.fullmatch(r"-?\d+", k): return "int<10" if abs(int(k)) < 10 else "int>=10"
    if re.fullmatch(r"-?\d+/\d+", k): return "fraction"
    return "other"
rows = collections.defaultdict(list); placebo = collections.defaultdict(list)
for qid, (keys, gold, _, pv) in dump.items():
    res = [results(t) for t in texts[qid]]
    c = collections.Counter(k for k in keys if k)
    if not c: continue
    a, v = c.most_common(1)[0]
    L = sum(1 for b, r in zip(keys, res) if b != a and a in r)
    rows[kind(a)].append((a == gold, v / (v + L)))
    for d in [str(i) for i in range(10)] + ["12", "24", "100"]:
        if d not in c and d not in pv:
            placebo[d].append(sum(d in r for r in res) / len(res))
print("placebo rate, clean detector:", {d: f"{statistics.mean(r):.0%}" for d, r in placebo.items()})
print("majority answer: tau if right vs tau if wrong (clean detector)")
for t in ("int<10", "int>=10", "fraction", "other"):
    r = rows[t]; right = [x[1] for x in r if x[0]]; wrong = [x[1] for x in r if not x[0]]
    if r: print(f"   {t:8s} n={len(r):4d}  tau_right={statistics.mean(right):.3f}  tau_wrong={statistics.mean(wrong) if wrong else float('nan'):.3f}"
                f"  share with any pass-through: right {sum(x<1 for x in right)/max(len(right),1):.0%} wrong {sum(x<1 for x in wrong)/max(len(wrong),1):.0%}")
