#!/bin/bash
# Run ideas in two parallel lanes: batch.sh OUTDIR "ideaA ideaB ..." "ideaC ideaD ..."
OUT=$1; LANE1=$2; LANE2=$3; STEPS=${STEPS:-4000}
cd "$(dirname "$0")"
lane() { for i in $1; do python3 run.py $i $OUT/$i.json --seeds 0 1 --steps $STEPS > $OUT/$i.log 2>&1 || echo "FAILED $i" >> $OUT/failed.txt; done; }
lane "$LANE1" & lane "$LANE2" & wait
echo done
