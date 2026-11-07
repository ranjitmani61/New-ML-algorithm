# Language-model test: pass criteria, written before any trained real-model result

Committed while the second Kaggle smoke run (which includes a 150-step training job) was still
running, so no trained-model result had been seen. The budget may still be calibrated by a
one-seed pilot; the criteria below may not change. Any change to anything else is logged with a
reason.

## Stages (same discipline as the tournament)

1. **Pilot (seed 0; calibration only, not a result).** Conditions `plain` and `oracle`, additive
   family. It checks that the budget learns the labeled range (plain at 4 events at least 90%)
   and that headroom exists (oracle at 12 events at least 20 points above plain).
   * Budget: 1,500 steps, batch 16, learning rate 2e-4, LoRA rank 16, 24,000 training problems,
     500 test problems per length.
   * Adjustment, fixed in advance (added 2026-11-07, before the pilot ran): if plain at 4 events
     is below 90%, double the steps to 3,000 once and rerun the pilot. If it is still below 90%,
     or the oracle headroom is missing, stop and report it.
2. **Discovery (seeds 1-2):** all six conditions, additive family. The mixed family runs on
   seed 1 only.
   * Go/no-go to the final (added 2026-11-07, before the pilot ran): continue only if
     `cancel_spread` beats `plain` at 12 events by at least 15 points on both discovery seeds,
     and beats the better length-matched control by at least 10 points on average. Otherwise
     stop and report the negative result without spending the final's budget.
3. **Registered final (fresh seeds 3, 4, 5):** all six conditions, additive family, with the
   specification and code hashes committed before the run. Discovery results never change the
   pass criteria below.

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

## Decision log

* **2026-11-07, pilot 1** (seed 0, 1,500 steps):
  * plain: 83.8% at 4 events, 9.2% at 12.
  * oracle: 83.8% at 4 events, 57.4% at 12.
  * Gate 1 failed (83.8% < 90%); gate 2 passed (+48.2 points).
  * Errors at 4 events are small arithmetic slips (80 of 81 within 2 of the answer); there were
    no parse failures.
  * Per the rule fixed in advance: steps doubled to 3,000, once, and the pilot reruns
    (`jobs/pilot2.json`). If plain stays below 90% at 4 events, stop.
* **2026-11-07, pilot 2** (seed 0, 3,000 steps):
  * plain: 93.0% at 4 events, 8.6% at 12.
  * oracle: 93.6% at 4 events, 78.2% at 12.
  * Both gates pass (93.0% >= 90%; +69.6 points of headroom).
  * The budget is fixed at 3,000 steps for discovery and the final.
  * Discovery jobs were rebalanced by measured run times (plain 928 s, oracle 1,816 s), with the
    go/no-go runs first in each lane.
