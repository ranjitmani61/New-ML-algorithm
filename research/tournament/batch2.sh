#!/bin/bash
# Two-lane runner for run2.py: batch2.sh TASK OUTDIR "ideaA ideaB" "ideaC ideaD"
TASK=$1; OUT=$2; LANE1=$3; LANE2=$4; STEPS=${STEPS:-4000}; SEEDS=${SEEDS:-"0 1"}
cd "$(dirname "$0")"
mkdir -p $OUT
lane() { for i in $1; do python3 run2.py $TASK $i $OUT/$i.json --seeds $SEEDS --steps $STEPS > $OUT/$i.log 2>&1 || echo "FAILED $i" >> $OUT/failed.txt; done; }
lane "$LANE1" & lane "$LANE2" & wait
echo done
