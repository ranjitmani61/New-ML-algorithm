# Plan: cancelling pairs where answers cannot be composed (three-person test)

Written on 2026-11-07, before any run of this test. The pilot rules below are fixed now. The
final will be registered in `SPEC_MULTI.md` after the pilot and before any run of the method on
this task.

## Why this test

Chaining short labeled problems (h1, arXiv 2510.07312) beat the method in the additive task
(`SPEC_CHAIN.md`), because there the labels compose exactly. This test asks whether the method
wins where they cannot.

The methods need different things:

| method | what it needs besides short labeled problems |
|---|---|
| chaining (h1) | a way to compute the composed problem's answer |
| compositional self-improvement (Shin et al. 2026, arXiv 2611.00452) | a decomposition into parts and a valid rule for combining their answers |
| direct self-improvement (Lee et al. 2025, arXiv 2502.01612) | unlabeled longer problems, several models trained with different seeds, rounds of retraining |
| the method (cancelling pairs) | one pair of events known to cancel (gain a, later lose a) |

Where answers cannot be composed, only the last two apply. So the test is the method against
direct self-improvement, with plain training and the oracle for reference.

## The task (`multi.py`)

* Three people each start with 5-30 of one item. Each event changes one person's amount by 1-9.
* The question asks for the largest final amount: "How many apples does the person with the most
  apples have now?"
* A label (the maximum) does not say who has it or what the others have. So appending events to a
  labeled problem gives an answer the label cannot determine (`test_multi.py` shows two problems
  with the same label that differ after the same event).
* The method inserts "P gains a ... P loses a", gain first. It never reads an amount (a test makes
  amount lookups raise during the construction). No amount can go negative, and every final amount
  is unchanged.
* The training labels cover 1-4 events. Tests: 500 problems at each of 4, 8, 12 and 16 events,
  plus 500 at 12 with held-out names and items. The primary metric is accuracy at 12 events.
* Setup as before: Qwen2.5-0.5B-Instruct, LoRA rank 16, 3,000 steps, batch 16, learning rate
  2e-4, 24,000 labeled problems, direct numeric answers, greedy decoding, strict parse.

## Conditions (`run_multi.py`)

* **plain:** the short labeled problems only.
* **cancel_spread (the method):** as registered for the additive task, with no tuning on this task
  (expansion probability 0.5, up to 16 events).
* **self_improve:** direct self-improvement, adapted from Lee et al.:
  * 3 models trained on the same labeled problems with different seeds; model 0 is trained exactly
    like plain (bit-identical in a local check);
  * each round, 4,000 unlabeled problems at the next lengths, labeled by all 3 models, keeping a
    label only when all 3 agree (as in Lee et al.'s maze runs);
  * all models then train 500 more steps on all data so far, with the newest round's data
    sampled at 50% (as in Lee et al.);
  * rounds go up to 16 events; the step per round is set by the pilot (below);
  * true labels of the unlabeled problems are logged for diagnosis and never used to train or
    filter;
  * model 0 is evaluated; models 1 and 2 are reported as secondary.
  * It uses about 6 times the method's training steps, plus unlabeled problems. That is part of
    the method being compared, and it will be reported.
* **oracle:** true labels for 1-16 events. This needs a solver and is the ceiling.
* Not included: the inactive and irrelevant-sentence controls. They were weak in the additive
  task (15.9% and 16.1% against the method's 58.2%), and the budget goes to self-improvement.

## Pilot (seed 0, calibration only; fixed now)

* Runs:
  * self_improve with no rounds, which trains and evaluates its 3 plain models, plus a vote
    probe at 5, 6, 7 and 8 events (500 fresh problems each);
  * plain;
  * oracle.
* Evaluation at 4, 6, 8, 12 and 16 events, plus held-out vocabulary at 12.
* **Gate A, learnability:** plain reaches at least 90% at 4 events.
  * If not, the steps double to 6,000 once and the pilot reruns.
  * If it still fails, stop: the task is too hard for this budget.
* **Gate B, headroom:** the oracle beats plain by at least 30 points at 12 events. If not, stop:
  there is nothing to recover.
* **Self-improvement schedule** (chosen to give it its best start, not to favor the method):
  * +2 events per round if the 3-model vote at 6 events keeps at least 20% of the problems with
    at least 90% label accuracy;
  * otherwise +1 event per round, which doubles its cost.
* The method is not run in the pilot.

## After the pilot

* `SPEC_MULTI.md` will register the final before any of its runs:
  * fresh seeds 9, 10 and 11;
  * the code hashes;
  * the decision rule: the method against self-improvement at 12 events, with a validity
    condition that the method beats plain by at least 15 points;
  * the predictions;
  * the budget and the stop rule.
* Estimated GPU use: about 1 hour for the pilot and 7-9 hours for the final, of about 18 left this
  week.

## Pilot 1 result (2026-11-07; `results/multi_pilot1/`)

* Checks: all 3 jobs exited 0; the code hashes match; no leakage; every set recomputes. Model 0 of
  self_improve equals the separate plain run on every set (also on the GPU).
* Plain: 88.8% at 4 events, 24.2% at 12. Oracle: 95.8% at 4, 74.2% at 12.
* **Gate A, learnability: FAIL** (88.8%, needs 90%).
* Gate B, headroom: PASS (+50.0 points).
* Vote probe: all 3 models agree on 79%, 66%, 55% and 48% of fresh problems at 5, 6, 7 and 8
  events. The agreed labels are right 92%, 91%, 84% and 79% of the time. So the schedule
  rule gives +2 events per round.
* **Per the rule above, the steps double to 6,000 once and the pilot reruns** (seed 0,
  otherwise unchanged; `jobs/multi_pilot2.json`). If gate A fails again, the test stops.
* GPU: 1.07 hours.

## Pilot 2 result (6,000 steps; `results/multi_pilot2/`)

* Checks: all 3 jobs exited 0; the code hashes match; no leakage; every set recomputes. Model 0
  of self_improve equals the separate plain run.
* Plain: 92.8% at 4 events, 27.6% at 12. Oracle: 97.4% at 4, 88.8% at 12.
* **Gate A: PASS** (92.8%). **Gate B: PASS** (+61.2 points).
* Vote probe: all 3 models agree on 79%, 69%, 57% and 53% of problems at 5-8 events. The agreed
  labels are right 95%, 94%, 91% and 83% of the time. **Schedule: +2 events per round.**
* GPU: 1.97 hours.
* The final is registered in `SPEC_MULTI.md` (6,000 steps, seeds 9-11) before any of its runs.

## Final result (seeds 9-11; `results/multi_final/`)

* **Verdict by the registered rule: METHOD BETTER.**
  * At 12 events the method beat self-improvement by +15.6, +18.4 and +18.6 points, and beat plain
    by +18.0 to +18.8 (validity holds).
  * Means at 12 events: the method 41.7%, self-improvement 24.1%, plain 23.3%, oracle 88.5%.
* Self-improvement gave no gain over plain (+0.9 points). Its kept labels decayed from 94% right
  in round 1 to 15% in round 6.
* The method itself is far below the oracle: it recovers 28% of the oracle's gain.
* Details and the prediction checks are in `SPEC_MULTI.md` (Result) and `REPORT.md`.
* GPU: 10.33 hours. Total for this test: 13.37 hours (pilots 3.04).
