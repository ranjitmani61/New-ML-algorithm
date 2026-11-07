# Cancelling-event insertion in a language model: registered final result

**Verdict: PASS.** All four pass criteria in `SPEC.md` hold on fresh seeds 3, 4 and 5.

* `SPEC.md` was committed in b6dc946, before any discovery result existed.
* The code, seeds, budget, metric and criteria ran unchanged.
* Two of the three recorded predictions held. The held-out-vocabulary prediction missed.

## The method

The method trains a model for problems with many events using only short labeled problems.

1. Take a short labeled word problem (1-4 events).
2. Insert pairs of events that cancel, on the problem's own quantity. For example, "Sara finds 3
   beads. ... Sara loses 3 beads."
3. The input gets longer, with more real state changes, but the answer stays the same. So no
   new label is computed.

Details:

* Each training example is expanded with probability 0.5.
* An expansion inserts between 0 and the maximum number of pairs (uniformly), so the problem has
  at most 16 changes.
* In the registered variant, each loss lands at a random position after its gain ("spread").

This transfers the inverse-pair insertion result (`research/final/REPORT.md`) from small
transformers on signed sums to a pretrained language model on word problems.

## The test

* **Model and training:**
  * Qwen2.5-0.5B-Instruct with LoRA rank 16 on all attention and MLP projections.
  * 3,000 steps, batch 16, learning rate 2e-4, 24,000 training problems.
  * The model answers with the number only (no chain of thought).
  * Greedy decoding with a strict parse.
* **Test sets:** 500 fresh problems at each of 4, 8, 12 and 16 events. Another 500 at 12 events
  use held-out names and items.
  * The primary metric is accuracy at 12 events.
  * Within a seed, every condition gets the same test problems.
* **Conditions:**
  * plain;
  * irrelevant sentences with numbers ("The room has 4 chairs.", in the style of PROBLEMATHIC or
    NoOp);
  * inactive sentences ("Sara counts the beads.");
  * the method;
  * the same pairs placed adjacent;
  * an oracle trained on true labels for 1-16 events.
* **Length matching:**
  * The four inserted conditions have matched training prompts: means of 80.0-80.5 tokens and a
    90th percentile of 131-133.
  * The irrelevant control's longest prompt is shorter (157-162 tokens against 180-191).
  * For reference: plain 58.7 tokens, oracle 99.1.

## Result

Means over seeds 3-5:

| condition | 4 events | 8 | **12 (primary)** | 16 | 12, held-out names and items |
|---|---|---|---|---|---|
| plain | 93.1% | 12.9% | **10.0%** | 6.9% | 8.4% |
| irrelevant sentences (control) | 92.7% | 28.1% | **16.1%** | 9.9% | 15.1% |
| inactive sentences (control) | 93.4% | 27.3% | **15.9%** | 9.2% | 13.5% |
| **cancelling pairs, spread (the method)** | 93.5% | 68.3% | **58.2%** | 47.0% | 52.3% |
| cancelling pairs, adjacent | 93.7% | 69.0% | **59.0%** | 50.3% | 54.7% |
| oracle (true labels, 1-16 events) | 93.3% | 82.1% | **76.1%** | 73.9% | 65.9% |

At 12 events, per seed:

| condition | seed 3 | seed 4 | seed 5 |
|---|---|---|---|
| plain | 10.0% | 9.2% | 10.8% |
| irrelevant sentences (control) | 13.8% | 20.0% | 14.6% |
| inactive sentences (control) | 12.6% | 15.6% | 19.6% |
| **cancelling pairs, spread (the method)** | 53.8% | 60.2% | 60.6% |
| cancelling pairs, adjacent | 59.0% | 60.2% | 57.8% |
| oracle (true labels, 1-16 events) | 75.4% | 76.4% | 76.6% |

**Pass criteria** (from `summarize.py --gate final`):

1. The method beats plain by at least 15 points on every seed: +43.8, +51.0 and +49.8. **Pass.**
2. It beats the better control (irrelevant, 16.1%) by at least 10 points: +42.1. **Pass.**
3. It recovers at least 50% of the oracle's gain: 72.9%. **Pass.**
4. It loses at most 3 points at 4 events: it loses none (-0.4 points, so it is slightly ahead).
   **Pass.**

**Recorded predictions** (from `check_predictions.py` and `audit_runs.py`):

* **Gains shrink at 16 events and on large answers: held.**
  * The share of the oracle's gain falls with length: 80% at 8 events, 73% at 12, 60% at 16.
  * The method scores 0% on the 43 test problems at 12 events whose answer is 60 or more. The
    oracle scores 20-29% on them.
* **Adjacent pairs land close to spread pairs: held.** At 12 events, adjacent minus spread is
  +0.8 points (95% interval -1.5 to +3.1).
* **Held-out names and items within 5 points of the training vocabulary: missed.** The method
  drops 5.9 points on average (0.4, 8.4 and 9.0 per seed).
  * The oracle drops more (10.3) and plain drops 1.6.
  * The method's gain over plain stays at +43.9 points on held-out vocabulary.

**Uncertainty.** The paired 95% bootstrap intervals resample test problems within each seed. They
do not include seed-to-seed variation; the per-seed gains above show that instead.

* The method minus plain at 12 events: +48.2 points (+45.5 to +50.9).
* The method minus the irrelevant control: +42.1 points (+39.3 to +44.9).

## Checks

* All 18 jobs exited 0.
* Both kernels recorded code hashes equal to `SPEC.md`.
* No test problem appears in any training set (exact text hash).
* None of the 18 runs truncated a training example.
* Every metric (90 of 90 sets) recomputes exactly from the raw generations, using code
  independent of the runner.
* No parse failures, and the strict and lenient parses agree.
* Reading raw outputs by eye (20 per condition) found bare numbers only.
* **Shortcut rules** score at most 7.6% on any test set:
  * the start amount;
  * the last number in the text;
  * the most common answer.
* **Error shape:** 86-89% of the method's wrong answers at 12 events are within 2 of the label.

## What this shows and what it does not

* **It shows** that the toy result transfers to a pretrained language model, with the same
  ordering.
  * Inserted material helps only when it is a real state change: cancelling pairs scored 58.2%,
    while irrelevant sentences (16.1%) and inactive sentences (15.9%) barely beat plain (10.0%).
  * This held under preregistration, on fresh seeds, at no in-distribution cost.
* **It is not compared with the closest prior art.**
  * h1 (arXiv 2510.07312) chains short problems into longer ones with composed labels.
  * In this additive task, chaining can build exact labels for long problems, so it may come
    close to the oracle, which beats the method by 17.9 points (+15.4 to +20.5). That is
    untested.
  * So the claim is "beats plain fine-tuning and clutter controls", not "beats the best known
    way to train long problems from short labels".
  * `PRIOR_ART.md` listed chaining as a needed comparison, and it was not run.
* **Large answers:** the method cannot reach answers beyond the short problems' range.
  * Its largest training answer is 57-61 per run (the oracle's is 98-100), and it scores 0% on
    test answers of 60 or more.
  * This is the same small-sum bias as in the toy.
* **Length:** the gain shrinks as problems get longer (above). Nothing was tested beyond 16
  events, the expansion limit. In the toy, neither the method nor the oracle extrapolated past
  the covered length.
* **Spreading the pairs is not the active ingredient.** Adjacent pairs do as well. The active
  ingredient is real cancelling state changes.
* **Order-dependent problems** (discovery only; one seed; doubling and halving):
  * The budget did not learn this family even in range: plain 65.0% and the oracle 66.8% at 4
    events.
  * Adjacent pairs raised accuracy at 12 events from 4.8% to 12.8%, but cost 4.0 points at 4
    events. The oracle reached 38.8%.
* **Scope:**
  * one 0.5B model;
  * one templated synthetic task;
  * direct answers, with no chain of thought;
  * LoRA fine-tuning.
  * The irrelevant control is templated sentences in the style of PROBLEMATHIC and NoOp, not
    those papers' own data.
  * Nothing here tests real benchmarks such as GSM8K, larger models, chain-of-thought outputs or
    naturally written problems.

**In one line:** a simple data rule, confirmed under registration on a real language model,
within a narrow synthetic setting. Whether it is better than chaining is open.

## Erratum

* `SPEC.md` says "50-step warmup, then linear decay". The code multiplies a 50-step linear
  warmup by a linear decay that starts at step 0. At step 49, for example, the factor is 0.984
  instead of 1.
  * This is a wording difference only; the code ran exactly as registered.
  * `SPEC.md` keeps its registered text, and the erratum is also appended there.

## How it was reached

1. **Smoke tests** (before 2026-11-07) found and fixed several bugs before any result. They
   include a torchao and PEFT crash on Kaggle; the list is in `PLAN.md`.
2. **Pilot gate** (seed 0, calibration only):
   * At 1,500 steps, plain reached 83.8% at 4 events, below the 90% gate.
   * Per the rule fixed in advance, the steps doubled once. At 3,000 steps plain reached 93.0%,
     with 69.6 points of headroom.
3. **Discovery** (seeds 1-2, 17 runs): at 12 events the method beat plain by +50.0 and +47.8, and
   the better control by +43.2. Go/no-go: GO.
4. **Registered final:** seeds 3-5, 18 runs, this report.

The decision log is in `SPEC_DRAFT.md`. GPU use this week was 9.22 of 30 hours, covering the
pilots, discovery and final, on T4 x2 kernels.

## Files and how to reproduce

* `results/{pilot1,pilot2,discovery,final}/`: per-run `metrics.json`, raw generations
  (`generations.jsonl.gz`), kernel logs, and the outputs of the scripts below. Adapters are not
  committed; they stay in the Kaggle kernel outputs.
* Commands, run in `research/llm/`:
  * `python3 summarize.py --gate final results/final` gives the verdict.
  * `python3 inspect_run.py results/final` recomputes the metrics and checks hashes and leakage.
  * `python3 -I audit_runs.py results/final` runs the shortcut, answer-size and raw-output audit.
  * `python3 -I check_predictions.py results/final` checks the predictions and gives the
    intervals.
* To rerun: `python3 kaggle_llm.py push --slug NAME --jobs jobs/final_a.json`, and the same for
  `final_b.json`.
