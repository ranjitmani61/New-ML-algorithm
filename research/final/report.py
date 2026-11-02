"""Apply SPEC.md's pass criteria and registered predictions to the final's results.

    python3 report.py RESULTS.json [RESULTS.json ...] > report.txt

Merges the per-lane result files of all kernels. Every condition and seed in SPEC.md must be
present; a missing run is reported, not skipped silently.
"""

import json
import sys

import numpy as np

CONDITIONS = ("baseline", "sigmoid", "fill", "zeros", "pairs", "oracle")
SEEDS = (100, 101, 102, 103, 104)
LENGTHS = ("8", "16", "24", "32", "48")

runs = {}
for path in sys.argv[1:]:
    runs.update(json.load(open(path))["runs"])

missing = [f"{c}/{s}" for c in CONDITIONS for s in SEEDS if f"{c}/{s}" not in runs]
print("missing runs:", missing or "none")


def acc(c, n):
    return {s: runs[f"{c}/{s}"]["eval"][n] for s in SEEDS if f"{c}/{s}" in runs}


print("\naccuracy by length (mean over seeds; per seed in brackets)")
print("condition".ljust(10), *(f"{n} terms".ljust(30) for n in LENGTHS))
for c in CONDITIONS:
    cells = []
    for n in LENGTHS:
        a = acc(c, n)
        cells.append((f"{np.mean(list(a.values())):.3f} [" + " ".join(f"{v:.2f}" for v in a.values()) + "]").ljust(30)
                     if a else "-".ljust(30))
    print(c.ljust(10), *cells)

print("\naccuracy at 24 terms by |sum| (mean over seeds; examples per seed in the bin)")
bins = list(next(iter(runs.values()))["by_sum_24"])
print("condition".ljust(10), *(b.ljust(14) for b in bins))
for c in CONDITIONS:
    rows = [runs[f"{c}/{s}"]["by_sum_24"] for s in SEEDS if f"{c}/{s}" in runs]
    if rows:
        cells = []
        for b in bins:
            vals = [r[b][0] for r in rows if r[b][0] is not None]
            cells.append(f"{np.mean(vals):.3f} (n~{int(np.mean([r[b][1] for r in rows]))})".ljust(14) if vals else "-".ljust(14))
        print(c.ljust(10), *cells)

if missing:
    print("\nverdict: INCOMPLETE (missing runs), criteria not evaluated")
    sys.exit(0)

b24, p24, f24, o24 = (acc(c, "24") for c in ("baseline", "pairs", "fill", "oracle"))
gain = [p24[s] - b24[s] for s in SEEDS]
c1 = all(g >= 0.30 for g in gain)
c2 = np.mean([p24[s] - f24[s] for s in SEEDS]) >= 0.20
share = (np.mean(list(p24.values())) - np.mean(list(b24.values()))) / (np.mean(list(o24.values())) - np.mean(list(b24.values())))
c3 = share >= 0.70
print("\npass criteria (24 terms)")
print(f"  1. pairs - baseline >= +30 points on all seeds: {[round(g, 3) for g in gain]} -> {'PASS' if c1 else 'FAIL'}")
print(f"  2. mean(pairs - fill) >= +20 points: {np.mean([p24[s] - f24[s] for s in SEEDS]):+.3f} -> {'PASS' if c2 else 'FAIL'}")
print(f"  3. share of the oracle's gain >= 70%: {share:.1%} -> {'PASS' if c3 else 'FAIL'}")
print(f"  verdict: {'PASS' if (c1 and c2 and c3) else 'FAIL'}")
p48 = np.mean(list(acc("pairs", "48").values()))
print("\nregistered predictions (not criteria)")
print(f"  pairs at 48 terms below 50%: {p48:.3f} -> {'as predicted' if p48 < 0.5 else 'NOT as predicted'}")
large = lambda c: np.mean([np.mean([runs[f'{c}/{s}']['by_sum_24'][b][0] for b in bins[2:]
                                    if runs[f'{c}/{s}']['by_sum_24'][b][0] is not None]) for s in SEEDS])
print(f"  pairs weaker than oracle on large sums (|sum| >= 20 bins): pairs {large('pairs'):.3f} vs oracle "
      f"{large('oracle'):.3f} -> {'as predicted' if large('pairs') < large('oracle') else 'NOT as predicted'}")
print("\ntraining seconds per run (mean):", {c: int(np.mean([runs[f'{c}/{s}']['train_seconds'] for s in SEEDS])) for c in CONDITIONS})
