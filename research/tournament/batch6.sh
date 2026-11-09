#!/bin/bash
# Batch 6 (LEDGER.md): split on the calibrated toy. Two CPU lanes, two cores each.
# Resumable: one job per condition and seed; a job whose JSON exists is skipped (run3.py writes
# the JSON only when the seed has finished). Lane order as registered.
# ONE_PER_LANE=1: run at most one job per lane, then exit. A round then fits inside one
# session-tracked task, which keeps the machine from being reclaimed while the jobs run.
cd "$(dirname "$0")"
OUT=results_batch6; mkdir -p $OUT
ran=0  # per lane: each lane runs in its own subshell
run() { local cores=$1 cond=$2 seed=$3; shift 3
  local out="$OUT/${cond}_s${seed}.json"
  [ -f "$out" ] && return 0
  [ "${ONE_PER_LANE:-0}" = 1 ] && [ "$ran" = 1 ] && return 0
  ran=1
  echo "$(date -u +%FT%TZ) start $cond $seed (cores $cores)" | tee -a "$OUT/rounds.log"
  taskset -c "$cores" python3 run3.py "$cond" "$out" --seeds "$seed" --steps 16000 "$@" > "$OUT/${cond}_s${seed}.log" 2>&1 \
    || echo "FAILED $cond $seed" >> "$OUT/failed.txt"
  echo "$(date -u +%FT%TZ) end $cond $seed" | tee -a "$OUT/rounds.log"; }
lane1() { for s in 0 1; do run 0,1 baseline $s; done
          for s in 0 1; do run 0,1 mined $s --m0 "$OUT/baseline_s{seed}_s{seed}.pkl"; done
          for s in 0 1; do run 0,1 split $s; done; }
lane2() { for c in pairs oracle split_pairs; do for s in 0 1; do run 2,3 $c $s; done; done; }
lane1 & lane2 & wait
left=0
for c in baseline mined split pairs oracle split_pairs; do for s in 0 1; do
  [ -f "$OUT/${c}_s${s}.json" ] || left=$((left + 1)); done; done
echo "jobs left: $left"
[ "$left" = 0 ] && touch "$OUT/DONE"
exit 0
