"""Placebo mention rates and pool difficulty from a gradedump.py pickle."""
import collections
import pickle
import statistics
import sys

dump = pickle.load(open(sys.argv[1], "rb"))["out"]
rates = collections.defaultdict(list)
maj = orac = head = 0
for keys, gold, mentions, pv in dump.values():
    cands = set(k for k in keys if k)
    for d in [str(i) for i in range(10)] + ["12", "24", "100"]:
        if d not in cands and d not in pv:
            rates[d].append(sum(d in m for m in mentions) / len(mentions))
    c = collections.Counter(k for k in keys if k)
    maj += bool(c) and c.most_common(1)[0][0] == gold
    orac += gold in c
    head += bool(c) and c.most_common(1)[0][0] != gold and gold in c
n = len(dump)
p1 = statistics.mean(sum(k == g for k in keys) / len(keys) for keys, g, _, _ in dump.values())
print("placebo mention rate:", {d: f"{statistics.mean(r):.0%}" for d, r in rates.items()})
print(f"pass@1 {p1:.1%}  majority@32 {maj/n:.1%}  any-correct {orac/n:.1%}  majority-wrong-gold-present {head}")
