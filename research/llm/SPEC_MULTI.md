# Registered test: cancelling pairs where answers cannot be composed (three-person task)

Registered on 2026-11-07, after the pilot (`PLAN_MULTI.md`, seed 0) and before any run of the
method on this task. Results are reported against this file unchanged, whatever the outcome. Any
deviation goes into an erratum here.

## Question

In a task where a short problem's answer cannot be combined into a longer problem's answer, does
the method (cancelling pairs, unchanged from `SPEC.md`) beat direct self-improvement (Lee et al.
2025, arXiv 2502.01612)? Self-improvement is the strongest existing method found that still
applies here. Chaining (h1) and compositional self-improvement (arXiv 2611.00452) both need a rule
for combining answers (`PLAN_MULTI.md`).

## Task and code

* Task: `multi.py`. Three people, events of 1-9 on one person each, and the question asks for the
  largest final amount. Labeled training problems have 1-4 events.
* Code (sha256), the same as in both pilots:
  * `multi.py`: 43c22ac06782e270facbb8fd6463ea44bd62100d480114776dee310787a9e79d
  * `run_multi.py`: fbbc6d3eba216dc7393af1b4171d4f299897545eae120869b57ff9d10f1ab3e4
  * `run_llm.py`: 2fe0b2afe62ef404ccb163d087842dc8892b2cebb5ae34204fbf19b24e4db6b7 (helpers
    only)
  * `problems.py`: d8cb98f513f14c3ce41ab088d3ff3023f2827b75b69a83b99d56bb8c364e325a (vocabulary
    and sentence templates only)
* Jobs: `jobs/multi_final_a.json` and `jobs/multi_final_b.json` (12 runs).

## Setup

* **Model and training:**
  * Qwen2.5-0.5B-Instruct with LoRA rank 16.
  * **6,000 steps** (doubled by pilot gate A), batch 16, learning rate 2e-4.
  * 24,000 labeled problems.
* **Evaluation:** greedy decoding with a strict parse. 500 test problems at each of 4, 8, 12 and
  16 events, plus 500 at 12 events with held-out names and items. Within a seed, every condition
  gets the same test problems.
* **Conditions:**
  * plain;
  * cancel_spread, the method: expansion probability 0.5, up to 16 events. Not tuned on this
    task.
  * self_improve:
    * 3 models; model 0 is trained exactly like plain;
    * 6 rounds of +2 events (5-6, 7-8, ..., 15-16), the step set by the pilot rule;
    * each round, 4,000 unlabeled problems; a label is kept only if all 3 models give the same
      number;
    * 500 more steps per model per round, with the newest round sampled at 50%;
    * model 0 is the result.
  * oracle: true labels for 1-16 events.
* **Seeds:** 9, 10, 11 (never used before).

## Decision rule (applied by `summarize_multi.py --gate final`)

The primary metric is accuracy at 12 events (training vocabulary), per seed.

1. **Validity:** cancel_spread - plain must be at least 15 points on every seed. If not, the
   verdict is METHOD FAILS HERE.
2. Otherwise, with d = cancel_spread - self_improve (model 0):
   * **METHOD BETTER:** mean d is at least +5 points, and d > 0 on every seed.
   * **SELF-IMPROVEMENT BETTER:** mean d is at most -5 points, and d < 0 on every seed.
   * **Otherwise: NO CLEAR DIFFERENCE.**
     * If every seed favors one side, it is reported as that side ahead on every seed, below the
       margin.
     * Paired bootstrap intervals over test problems will be reported alongside. They are
       descriptive, and they do not change the verdict.

## Secondary (reported, not decisive)

* accuracy at 4, 8 and 16 events, and on held-out vocabulary;
* the method's and self-improvement's cost at 4 events (plain minus each);
* self-improvement models 1 and 2;
* self-improvement per round: problems kept and their label accuracy (a diagnostic from logged
  true answers, never used in training);
* accuracy by answer size, and the shortcut rules (`audit_multi.py`);
* training seconds per condition, the compute cost.

## Predictions (recorded before the run)

1. **Verdict: METHOD BETTER.** My confidence is about 60%.
2. At 12 events (means):
   * the method at 60-75%;
   * self-improvement at 40-60%;
   * plain about 28% and oracle about 89%, as in pilot 2.
3. Validity holds.
4. Self-improvement's kept labels fall below 85% accuracy by round 4 (11-12 events). In pilot 2
   the vote kept 91% accurate labels at 7 events and 83% at 8.
5. The method costs at most 3 points at 4 events.

## Budget and stop rule

* **Estimated GPU use: about 10.2 hours on 2 kernels,** of about 14.9 left this week:
  * self-improvement about 3.9 hours per seed;
  * the method and the oracle about 1.1 hours each;
  * plain about 0.6 hours.
* The registered jobs run once.
* A job that fails for an infrastructure reason may be rerun once with the same seed. Any rerun
  is logged here.

## What each outcome means

* **METHOD BETTER:** where answers cannot be composed, the pure rule beats the strongest existing
  method that applies. It does so with no unlabeled problems and about a fifth of the training
  compute.
* **SELF-IMPROVEMENT BETTER:** even here the existing method wins. The method's remaining
  advantage would be cost only.
* **NO CLEAR DIFFERENCE:** reported as it is, including the direction.
* **METHOD FAILS HERE:** the rule did not carry over to this task.
* Scope in every case:
  * one 0.5B model;
  * one synthetic task;
  * self-improvement's unlabeled problems come from the test distribution, which favors it;
  * its settings are an adaptation of Lee et al. (stated above), not their exact recipe.

## Addendum (2026-11-07, after registration and before any final result; not part of the decision rule)

Answer sizes, from the generator only (`answer_support_multi.py`, output in
`results/multi_final/answer_support.txt`; no model output is read):

* The method never changes a label, so its training answers are those of 1-4 event problems:
  median 26; only 0.2-0.3% are 45 or more; the largest is 55-58.
* At 12 events, 5.2-8.0% of test answers are 45 or more (the oracle trains on about 4% such
  answers). On each seed, 9-13% of test answers exceed the 99th percentile of the method's training
  answers.
* So answer size alone can cost the method at most about 13 points at 12 events. In the sum toy it
  was the main failure (`research/tournament/LEDGER.md`, idea #8). Here, most of any gap to the
  oracle would have to come from tracking the amounts.

**Prediction 6 (secondary, about 70% confidence):** at 12 events (training vocabulary), pooled
over seeds 9-11, the method's accuracy on answers of 45 or more is at least 20 points below its
accuracy on answers of 29 or less. The oracle's drop between the same answer sizes is smaller than
the method's. Checked from `audit_multi.py` section 2, with the bins weighted by problem count.

## Addendum 2 (2026-11-07, before any final result): an applicable method missed at registration

* The Question section says self-improvement is "the strongest existing method found that still
  applies here". A search after registration found another one.
* PAR (arXiv 2610.27628, 2026-10-20) retrofits Qwen2.5-0.5B as a looped model and uses prefix
  self-distillation, which needs only final answers on short problems. A prefix of a three-person
  problem is a valid shorter problem, so PAR applies to this task.
* It reports 85% at 16x the training length on S5 state tracking without step labels. This project
  has shown about 3-4x.
* PAR is not compared here. The registered test can show at most whether the method beats direct
  self-improvement, not every method that applies. The decision rule is unchanged.
