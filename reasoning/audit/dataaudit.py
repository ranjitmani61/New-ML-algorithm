import gzip, json, glob, os, sys, collections, re
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
import benchmarks as bm
def items(d):
    for p in sorted(glob.glob(os.path.join(d, "part*.jsonl.gz"))):
        for line in gzip.open(p, "rt"):
            yield json.loads(line)
norm = lambda s: " ".join(s.split())
stats = {}
probs = {}
for name in ("hard_dev", "hard_test"):
    d = f"{sys.argv[1]}/{name}/pools"
    ids, golds, nboxed, unfinished, noans, toks, types = [], [], [], 0, 0, [], collections.Counter()
    n = 0
    probs[name] = set()
    for it in items(d):
        ids.append(it["id"]); probs[name].add(norm(it["question"]))
        g = bm.gold_answer(it); golds.append(g)
        nboxed.append(it["solution"].count("\\boxed"))
        types[it["id"].split("/")[1]] += 1
        for s in it["samples"]:
            n += 1
            unfinished += not s["finished"]
            noans += s["finished"] and bm.model_answer(s["text"]) is None
            toks.append(s["tokens"])
    dup = [k for k, v in collections.Counter(ids).items() if v > 1]
    print(name, "questions", len(ids), "dup ids", len(dup),
          "gold None", sum(g is None for g in golds), "gold empty", sum(g is not None and not g.strip() for g in golds),
          "solutions with >1 boxed", sum(b > 1 for b in nboxed), "no boxed", sum(b == 0 for b in nboxed))
    print("   attempts", n, "truncated(%)", round(100*unfinished/n, 2), "finished-but-no-answer(%)", round(100*noans/n, 2),
          "mean tokens", round(sum(toks)/n), "at cap 2048", sum(t >= 2048 for t in toks))
    print("   subjects", dict(types))
print("problem-text overlap dev/test", len(probs["hard_dev"] & probs["hard_test"]))
