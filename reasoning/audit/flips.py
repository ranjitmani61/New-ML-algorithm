"""Post-hoc (exploratory, after the sealed test): answer changes trv vs base on the full 32-attempt pool."""
import json, os, sys
from collections import Counter
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
import grading, lab, selection
pools, setname = sys.argv[1], sys.argv[2]
spec = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "hard_spec.json")))
qs = grading.load_pools(pools, {setname})
c = spec["conf"]["32"]
flips, subj = Counter(), Counter()
majwrong_goldpresent = 0
for q in qs:
    t = q.traces
    w = lab.conf_weights(t, c["measure"], c["eta"])
    mass = {}
    for tr, wi in zip(t, w):
        if tr.answer is not None: mass[tr.answer] = mass.get(tr.answer, 0) + float(wi)
    base = max(mass, key=mass.get) if mass else None
    new = selection.terminal_ratio_vote(t, q.problem_values, 2.0, w)[0]
    if base != new:
        key = ("R" if base == q.gold else "W") + "->" + ("R" if new == q.gold else "W")
        flips[key] += 1
        subj[(getattr(q, "subject", None) or q.qid.split("/")[0], key)] += 1
print(setname, len(qs), dict(flips))
