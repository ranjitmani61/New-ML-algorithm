import sys, gzip, json, glob, os, pickle, collections, re
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
import benchmarks as bm, grading
d = sys.argv[1]
dump = pickle.load(open(sys.argv[2], "rb"))["out"]
multi_gold, dagger, chain, multibox = [], [], [], []
for p in sorted(glob.glob(os.path.join(d, "part*.jsonl.gz"))):
    for line in gzip.open(p, "rt"):
        it = json.loads(line)
        raws = [bm.model_answer(s["text"]) if s["finished"] else None for s in it["samples"]]
        keys, members = grading.cluster(raws, it["question"])
        gold = grading.map_choice(bm.gold_answer(it), grading.mc_options(it["question"]))
        cnt = collections.Counter(k for k in keys if k)
        match = [k for k, rs in members.items() if any(grading.matches_gold(gold, r) for r in rs)]
        if len(match) > 1:
            multi_gold.append((it["id"], gold, [(k, cnt[k]) for k in match]))
        if dump[it["id"]][1].endswith("†"):
            dagger.append((it["id"], gold))
        for k, rs in members.items():
            if len(rs) > 2:
                bad = [(a, b) for i, a in enumerate(rs) for b in rs[i+1:] if not grading.same_answer(a, b)]
                if bad:
                    chain.append((it["id"], k, bad[:2]))
        if it["solution"].count("\\boxed") > 1:
            sol = it["solution"]
            starts = [m.start() for m in re.finditer(r"\\boxed", sol)] + [len(sol)]
            boxes = [bm.last_boxed(sol[a:b]) for a, b in zip(starts, starts[1:])]
            multibox.append((it["id"], boxes, gold))
print("questions where >1 cluster matches gold:", len(multi_gold))
for x in multi_gold[:12]: print("  ", x)
print("gold collides with a non-matching cluster (dagger):", len(dagger), dagger[:5])
print("clusters merged by chaining (members not pairwise equal):", len(chain))
for x in chain[:8]: print("  ", x)
print("multi-boxed solutions:", len(multibox))
for x in multibox: print("  ", x)
