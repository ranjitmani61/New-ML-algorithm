import sys, pickle, collections, re
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
import selection
dump = pickle.load(open(sys.argv[1], "rb"))["out"]
def kind(k):
    if re.fullmatch(r"-?\d+", k): return "integer" if abs(int(k)) >= 10 else "integer<10"
    if re.fullmatch(r"-?\d+/\d+", k): return "fraction"
    return "other(sqrt,pi,tuple,expr...)"
# 1) can the extractor see the key at all, if the key is written verbatim?
cover = collections.Counter(); total = collections.Counter()
gold_kind = collections.Counter()
for qid, (keys, gold, mentions, pv) in dump.items():
    gold_kind[kind(gold)] += 1
    for k in set(x for x in keys if x):
        total[kind(k)] += 1
        cover[kind(k)] += k in selection.mentioned_values("$" + k + "$")
print("gold answers by type:", dict(gold_kind))
print("candidate answers the extractor can detect when written verbatim:")
for t in total: print(f"   {t:30s} {cover[t]}/{total[t]} = {cover[t]/total[t]:.0%}")
# 2) majority answer over all 32: tau and correctness by type
rows = collections.defaultdict(list)
for qid, (keys, gold, mentions, pv) in dump.items():
    cnt = collections.Counter(k for k in keys if k)
    if not cnt: continue
    a, v = cnt.most_common(1)[0]
    L = 0 if a in pv else sum(1 for b, m in zip(keys, mentions) if b != a and a in m)
    rows[kind(a)].append((a == gold, v / (v + L), a in pv))
print("majority answer: n, accuracy, mean tau if right, mean tau if wrong, exempt as problem value")
for t, r in rows.items():
    right = [x[1] for x in r if x[0]]; wrong = [x[1] for x in r if not x[0]]
    print(f"   {t:30s} n={len(r):3d} acc={sum(x[0] for x in r)/len(r):.2f} tau_right={sum(right)/max(len(right),1):.2f} tau_wrong={sum(wrong)/max(len(wrong),1):.2f} exempt={sum(x[2] for x in r)}")
# 3) incidental matches: how often does a wrong-answer attempt 'mention' a small integer gold answer
