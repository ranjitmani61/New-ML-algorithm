#!/bin/bash
# Preregistered confirmation: kept ideas + baseline + best reference on untouched seeds 2 3 4.
OUT=$1; shift
cd "$(dirname "$0")"
lane() { for i in $1; do python3 run.py $i $OUT/$i.json --seeds 2 3 4 --steps 4000 > $OUT/$i.log 2>&1 || echo "FAILED $i" >> $OUT/failed.txt; done; }
lane "baseline npia" & lane "sigmoid neia" & wait
echo done
