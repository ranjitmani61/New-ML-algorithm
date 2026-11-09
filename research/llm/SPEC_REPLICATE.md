# Registered replication: rewrite mining vs the hand rule on fresh seeds 12-14

Registered on 2026-11-09, before any run on seeds 12, 13 or 14. Results are reported against this
file unchanged, whatever the outcome. Any deviation goes into an erratum here.

## Why

`SPEC_MINE.md` gave **MATCHES** on seeds 9-11. On those seeds, the model's self-certified edits
trained it about as well as my hand-designed cancelling-pair rule: d = +0.8, +2.0 and +1.0
points at 12 events. Those seeds were not fresh. Their comparison results were known, and their
12-event test problems had been analysed. This test repeats the same method on seeds no
three-person run has used (so far seeds 0 and 9-11). All three conditions are trained in one
kernel.

## What is the same

* **Code, unchanged** (sha256, as recorded in `SPEC_MINE.md` and the kernel's own metrics):
  * `mine_multi.py` ea2bd63f…
  * `run_multi.py` fbbc6d3e…
  * `multi.py` 43c22ac0…
  * `run_llm.py` 2fe0b2af…
  * `problems.py` d8cb98f5…
  * `kaggle_llm.py` 9692fb98…
* **Settings, unchanged.**
  * Certification: stage 1 of 8 problems with at most 1 miss; stage 2 up to 64; margin 5 points;
    reference 1,000 problems per edit size; gate 90% at 3-4 events.
  * Training: 6,000 steps, batch 16, learning rate 2e-4, LoRA rank 16, 24,000 problems with 1-4
    events, half lengthened up to 16 events.
  * Evaluation: 500 problems at each of 4, 8, 12 and 16 events, plus 500 at 12 events with
    held-out vocabulary, all greedy.
* **M0** is the plain model of the same seed, trained earlier in the same lane of this kernel.
  `mine_multi.py` finds it under `/kaggle/working/out` and records its sha256. On seeds 9-11, M0
  was the final's plain model, trained with the same recipe.

## What is new

* **Seeds 12, 13 and 14.**
* **Conditions:** plain, cancel_spread and mined. self_improve and oracle are left out to save
  compute; they are not part of the decision rule. On seeds 9-11, self_improve was within 3.2
  points of plain at 12 events.
* **Job:** `jobs/mine_replicate.json`, kernel `llm-mine-replicate`, T4 x2.
  * Lane 0: plain 12, mined 12, plain 13, mined 13.
  * Lane 1: plain 14, mined 14, then cancel_spread 12, 13 and 14.

## Decision rule (SPEC_MINE.md's, unchanged; `summarize_replicate.py`)

The primary metric is accuracy at 12 events (training vocabulary), per seed. Let d = mined -
cancel_spread.

1. **Gate.** If fewer than two seeds pass M0's gate: NO TEST.
2. **Validity.** If mined minus plain is below 15 points on a seed that passed: MINING FAILS HERE.
   * A seed whose M0 passes the gate but certifies no edit also counts as MINING FAILS HERE.
     Finding nothing is a failure of mining, not a reason to drop the seed.
   * If a run is missing for an infrastructure reason (for example, M0 not found), the verdict
     waits for the one allowed rerun.
3. **MINING BETTER:** mean d at least +5 points, and d > 0 on every seed.
4. **HAND RULE BETTER:** mean d at most -5 points, and d < 0 on every seed.
5. **Otherwise: MATCHES.**

The result is **REPLICATED** if the verdict is MATCHES or MINING BETTER. It is **NOT REPLICATED**
if the verdict is HAND RULE BETTER or MINING FAILS HERE.

`summarize_replicate.py` was committed with this file. Run on seeds 9-11, it reproduces
`SPEC_MINE.md`'s numbers and verdict. A local run of the whole job file passed before
registration: tiny model, 30 steps, perfect-predictor certification on 150 random edits. All nine
jobs ran in order, and each mined run loaded that kernel's plain model of its seed (checked by
sha256). The run tested only the plumbing.

## Secondary (reported, not decisive)

* Certification precision and recall. The closest wrong edit's score against the threshold.
  Edits certified wrongly, with their true keep-rate. A wrongly certified edit is not excluded
  afterwards: its effect on training is part of the result.
* The share of true training labels and of valid training problems.
* Accuracy at 4, 8 and 16 events and with held-out vocabulary. On seeds 9-11, mined was below
  cancel_spread at 8 events and with held-out vocabulary on every seed, and above it at 16
  events on every seed.
* GPU seconds.
* Mean d over all six seeds 9-14, labelled as including the three seeds that were not fresh.

## Predictions (recorded before the run)

1. The gate passes on all three seeds (about 80%).
2. Verdict: MATCHES about 70%, MINING FAILS HERE about 10%, MINING BETTER about 8%, HAND RULE
   BETTER about 8%, NO TEST about 4%.
3. No wrong edit is certified on any seed (about 60%). On seed 11 the closest wrong edit was
   only 1 point below the threshold.
4. Recall is between 50% and 90% on every seed (about 75%).
5. Mean at 12 events (about 75% each):
   * mined 37-48%;
   * cancel_spread 36-47%;
   * plain 19-28%.
6. As on seeds 9-11, mined's mean is below cancel_spread's at 8 events (about 70%) and above
   it at 16 events (about 70%).

## Known weaknesses, stated before the run

* **This tests reproducibility, not generality.** It uses the same task, the same edit space
  (chosen by me), the same model size and the same recipe. A different task is a separate test.
* **The fixed 5-point margin is a known weak point.** On seed 11 a wrong edit came within 1 point
  of certification. This run does not change the margin, because a replication has to test the
  method that was tested.
* **No self_improve or oracle on these seeds.**

## Budget and stop rule

* Lane 0: about 2 x (2,415 + 1,040 + 4,980) s, about 4.7 hours.
* Lane 1: about 8,450 + 3 x 4,090 s, about 5.8 hours. This is about 5.9 hours of GPU quota, the
  wall time of the longer lane, after the quota refresh on 2026-11-14.
* A job that fails for an infrastructure reason may be rerun once with the same seed and
  settings, and the rerun is reported.
* If the kernel fails as a whole, I fix the cause and rerun once if enough quota is left.
  Otherwise it waits for the next refresh.
