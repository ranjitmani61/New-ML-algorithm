#!/bin/bash
# Batch 6 (LEDGER.md): split on the calibrated toy. Two CPU lanes, two cores each.
cd "$(dirname "$0")"
OUT=results_batch6; mkdir -p $OUT
S="--seeds 0 1 --steps 16000"
run() { local cores=$1 cond=$2; shift 2
  taskset -c "$cores" python3 run3.py "$cond" "$OUT/$cond.json" $S "$@" > "$OUT/$cond.log" 2>&1 || echo "FAILED $cond" >> "$OUT/failed.txt"; }
lane1() { run 0,1 baseline; run 0,1 mined --m0 "$OUT/baseline_s{seed}.pkl"; run 0,1 split; }
lane2() { run 2,3 pairs; run 2,3 oracle; run 2,3 split_pairs; }
lane1 & lane2 & wait
echo done
