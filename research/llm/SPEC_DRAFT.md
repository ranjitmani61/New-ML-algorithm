# Language-model test: pass criteria, written before any trained real-model result

Committed while the second Kaggle smoke run (which includes a 150-step training job) was still
running, so no trained-model result had been seen. The budget may still be calibrated by a
one-seed pilot; the criteria below may not change. Any change to anything else is logged with a
reason.

## Stages (same discipline as the tournament)

1. **Pilot (seed 0; calibration only, not a result).** Conditions `plain` and `oracle`, additive
   family. It checks that the budget learns the labeled range (plain at 4 events at least 90%)
   and that headroom exists (oracle at 12 events at least 20 points above plain). If either check
   fails, adjust the budget once and log it, or stop.
2. **Discovery (seeds 1-2):** all six conditions, additive family. The mixed family runs on
   seed 1 only.
3. **Registered final (fresh seeds 3, 4, 5):** all six conditions, additive family, with the
   specification and code hashes committed before the run.

## Primary metric

Exact-match accuracy (strict parse) on 500 fresh additive problems with 12 events (3x the
labeled maximum of 4, within the expansion range of 16), per seed.

## Pass criteria (registered final; all must hold)

1. `cancel_spread` beats `plain` by at least 15 points on every seed.
2. `cancel_spread` beats the better of the two length-matched controls (`irrelevant`,
   `inactive`) by at least 10 points on average.
3. `cancel_spread` recovers at least 50% of the oracle's gain over `plain` (mean over seeds).
4. `cancel_spread` loses at most 3 points against `plain` at 4 events (the in-distribution
   cost).

## Predictions (recorded, not criteria)

* Gains shrink at 16 events and on large answers, because expanded problems keep the short
  problems' answer range (as in the toy).
* `cancel_adjacent` is close to `cancel_spread` (toy audit: 86% vs 90%).
* Mixed family: little or no gain at long lengths, and a possible in-distribution cost (as S3).
* Held-out names and items at 12 events: within 5 points of the in-vocabulary result.
