# Registered comparison: cancelling pairs vs chaining short problems (h1-style)

Registered on 2026-11-07, before any chaining run on a real model. Results are reported against
this file unchanged, whatever the outcome. Any deviation goes into an erratum here.

## Question

Does cancelling-pair insertion (the method, which passed `SPEC.md`) beat chaining short labeled
problems? Chaining is the data construction of the closest prior art. Both get the same pipeline
and the same labeled data.

## Chaining as implemented (`problems.chain`)

* **What h1 does** (arXiv 2510.07312, read for this spec):
  * It composes short problems serially. Its "substitution chaining" feeds each answer into the
    next problem's input, recomputes the label, and applies well-posedness checks.
  * It trains with reinforcement learning and a curriculum over chain length. **Only the data
    construction is compared here**, under one supervised pipeline.
* **How it is built here:**
  1. A short labeled problem is extended by appending the events of other short labeled
     problems. They come from the same 24,000-problem training pool, re-rendered on the same
     person and item.
  2. The running amount replaces each piece's start (an identity adapter).
  3. The label is composed from labels alone: label + (piece answer - piece start).
  4. A piece that would make the amount negative is skipped.
* **Format:** one continuous story, the same format as the test problems. h1 lists the
  sub-problems separately; the single story is the in-format, and so stronger, choice for this
  test.
* **Length:** the target distribution is the method's. Measured on seed 6's training set,
  chaining vs the method:
  * 5.77 vs 5.82 changes per problem;
  * 184 vs 185 characters on average;
  * 359 vs 362 characters at the 90th percentile.
* **Answers:** unlike the method's, chained labels go beyond the short problems' range. The
  largest is 105 on seed 6, against 61 for the method.
* **Tests:** `test_problems.py` checks three things:
  * the composed label equals an independent parse of the text, over 3,000 chains;
  * the problem stays valid;
  * every event is about the same person and item.

## Code (sha256)

* `problems.py`: d8cb98f513f14c3ce41ab088d3ff3023f2827b75b69a83b99d56bb8c364e325a
* `run_llm.py`: 2fe0b2afe62ef404ccb163d087842dc8892b2cebb5ae34204fbf19b24e4db6b7
* Every earlier condition's training and test data is bit-identical to the code registered in
  `SPEC.md`. This was checked for all six earlier conditions on seeds 3 and 6.
* Jobs: `jobs/chain_a.json` and `jobs/chain_b.json` (12 runs).

## Setup (as in `SPEC.md`)

* **Model and training:**
  * Qwen2.5-0.5B-Instruct with LoRA rank 16.
  * 3,000 steps, batch 16, learning rate 2e-4, 24,000 training problems.
  * Each training problem is expanded with probability 0.5.
* **Evaluation:** greedy decoding with a strict parse. 500 test problems at each of 4, 8, 12 and
  16 events, plus 500 at 12 events with held-out names and items.
* **Conditions:** plain, cancel_spread (the method), chain, oracle.
* **Seeds:** 6, 7, 8 (never used before).

## Primary metric and decision rule

The primary metric is accuracy at 12 events (train vocabulary), per seed. With d = chain -
cancel_spread:

* **CHAINING BETTER:** mean d is at least +5 points, and d > 0 on every seed.
* **METHOD BETTER:** mean d is at most -5 points, and d < 0 on every seed.
* **Otherwise:** NO CLEAR DIFFERENCE.

The rule is applied mechanically by `summarize.py --gate chain`.

## Secondary (reported, not decisive)

* Replication: cancel_spread beats plain by at least 15 points at 12 events on every seed
  (`SPEC.md` criterion 1).
* chain against the oracle at 12 events.
* Accuracy on test answers of 60 or more (`audit_runs.py`).
* chain's in-distribution cost at 4 events.
* Held-out vocabulary at 12 events.

## Predictions (recorded before the run)

1. The verdict is CHAINING BETTER.
2. chain reaches at least 65% at 12 events (mean). The method lands about where it did in the
   final (58%).
3. chain scores above 10% on answers of 60 or more; the method scores 0%.
4. The replication holds.
5. chain costs at most 3 points at 4 events.

## Budget and stop rule

* 12 runs on 2 kernels (T4 x2), estimated at 2.9 GPU hours. 20.8 hours remain this week.
* The registered jobs run once.
* A job that fails for an infrastructure reason (not the method) may be rerun once with the same
  seed. Any rerun is logged here.

## What each outcome means

* **CHAINING BETTER:** in this composable additive task, the prior art's construction wins. The
  method's value would then be limited to domains where labels cannot be composed, which is
  untested.
* **METHOD BETTER or NO CLEAR DIFFERENCE:** the method is at least competitive with chaining here,
  at equal labeled data and length.

## Result (appended after the run; the text above is unchanged)

* **Verdict: NO CLEAR DIFFERENCE** (`summarize.py --gate chain`). chain - cancel_spread at 12
  events was +4.8, +4.4 and +3.4 points on seeds 6, 7 and 8, a mean of +4.2. The rule needed +5.
* No deviations: the 12 registered jobs ran once and all exited 0, with no reruns. Both kernels
  recorded the code hashes above. GPU use was 2.85 hours (from the kernel logs).
* Chaining was ahead on every seed. The paired 95% bootstrap interval over test problems is +1.7
  to +6.7 at 12 events, and +6.9 to +12.6 at 16 events, where chaining led by +9.8.
* So the outcome meaning registered above for NO CLEAR DIFFERENCE ("at least competitive")
  overstates the method's position. The report gives the weaker reading the data support.
* Predictions: 4 and 5 held; 1, 2 and 3 missed (3 by the narrowest margin, 3 of 30 = 10%).
* See `REPORT.md` and `results/chain/`.
