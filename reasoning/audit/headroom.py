"""Questions a perfect tie-breaker could win vs questions a noisy one could lose (gradedump.py pickle)."""
import collections
import pickle
import sys

dump = pickle.load(open(sys.argv[1], "rb"))["out"]
win = lose = 0
for keys, gold, _, _ in dump.values():
    c = collections.Counter(k for k in keys if k)
    if not c:
        continue
    (lead, lv), *rest = c.most_common()
    if lead != gold and gold in c and c[gold] >= lv / 2:
        win += 1
    if lead == gold and rest and rest[0][1] >= lv / 2:
        lose += 1
n = len(dump)
print(f"winnable (gold >= half the leader's votes) {win/n:.1%}; losable (right leader, runner-up >= half) {lose/n:.1%}")
