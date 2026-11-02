"""Table of final accuracy per condition across seeds, plus pseudo-label quality per round."""

import json
import sys

import numpy as np

runs = [json.load(open(p)) for p in sys.argv[1:]]
conds = [c for c in runs[0]["conditions"]]
ns = sorted(int(n) for n in runs[0]["conditions"][conds[0]][0]["eval"])
print("final-round greedy accuracy, mean over seeds [per seed]")
print("condition".ljust(12), *(f"n={n}".ljust(24) for n in ns))
for c in conds:
    cells = []
    for n in ns:
        v = [r["conditions"][c][-1]["eval"][str(n)] for r in runs if c in r["conditions"]]
        cells.append(f"{np.mean(v):.3f} [{' '.join(f'{x:.2f}' for x in v)}]".ljust(24))
    print(c.ljust(12), *cells)
print("\npseudo-labels per round (mean over seeds): coverage / precision / wrong labels per expression")
for c in conds:
    rows = []
    for k in range(1, len(runs[0]["conditions"][c])):
        q = [r["conditions"][c][k]["labels"] for r in runs if c in r["conditions"]]
        prec = [x["precision"] for x in q if x["precision"] is not None]
        rows.append(f"r{k}: {np.mean([x['coverage'] for x in q]):.2f}/{np.mean(prec) if prec else float('nan'):.2f}/"
                    f"{np.mean([x['wrong_labels_per_expr'] for x in q]):.2f}")
    print(c.ljust(12), "  ".join(rows))
