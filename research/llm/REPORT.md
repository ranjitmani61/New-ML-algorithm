# Cancelling-event insertion in a language model: registered results

Three registered tests on Qwen2.5-0.5B-Instruct and templated word problems:

1. **Against plain fine-tuning and clutter controls** (`SPEC.md`, seeds 3-5): **PASS** on all
   four criteria. At 12 events the method scored 58.2%, plain 10.0%.
2. **Against chaining short labeled problems**, the data construction of the closest prior art
   (`SPEC_CHAIN.md`, seeds 6-8): **NO CLEAR DIFFERENCE** by the registered rule.
   * But chaining was ahead on every seed: +4.2 points at 12 events (95% interval +1.7 to +6.7),
     short of the registered 5-point bar, and +9.8 at 16 events.
   * **The method does not beat the prior art here.**
3. **Where answers cannot be composed, against direct self-improvement** (Lee et al. 2025;
   `SPEC_MULTI.md`, seeds 9-11): **METHOD BETTER.** At 12 events the method was ahead by +17.5
   points, on every seed.
   * But self-improvement gave no gain over plain there (+0.9 points).
   * The method itself recovered only 28% of the oracle's gain (41.7% against 88.5%).
   * A stronger applicable method, PAR (arXiv 2610.27628), was found after registration and not
     compared.

* Each spec was committed before any of its results existed: `SPEC.md` in b6dc946,
  `SPEC_CHAIN.md` in 855a918, `SPEC_MULTI.md` in 4cdafb5.
* The code, seeds, budget, metric and decision rule ran unchanged in all three.
* Recorded predictions: two of three held in the first test, two of five in the second, and five
  of six in the third.

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

## Result against plain and controls (seeds 3-5)

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

## Checks (seeds 3-5)

* All 18 jobs exited 0.
* Both kernels recorded code hashes equal to `SPEC.md`.
* No test problem appears in any training set (exact text hash).
* None of the 18 runs truncated a training example.
* Every metric (90 of 90 sets) recomputes exactly from the raw generations, using code
  independent of the runner.
* No parse failures, and the strict and lenient parses agree.
* Reading raw outputs by eye (20 per condition) found bare numbers only.
* **Shortcut rules** score at most 7.6% on the 4-, 12- and 16-event test sets:
  * the start amount;
  * the last number in the text;
  * the most common answer.
* **Error shape:** 86-89% of the method's wrong answers at 12 events are within 2 of the label.

## Against chaining (seeds 6-8)

**Question** (`SPEC_CHAIN.md`): does the method beat chaining short labeled problems? h1
(arXiv 2510.07312) builds long problems this way.

* **Chaining as built here:**
  * A short labeled problem is extended with the events of other short labeled problems from the
    same 24,000-problem pool. They are rewritten on the same person and item, as one story.
  * The label is composed from the pieces' labels. A test checks it against an independent parse
    of the text.
* **Fairness:**
  * the same labeled data, pipeline and test problems as the method;
  * matched length: 5.74-5.77 changes per training problem against 5.71-5.82 for the method, and
    80.5-80.7 prompt tokens against 80.3-81.0.
* **Not compared:** h1's reinforcement learning and curriculum. Only its data construction is
  compared, under the same supervised training.

Means over seeds 6-8:

| condition | 4 events | 8 | **12 (primary)** | 16 | 12, held-out names and items |
|---|---|---|---|---|---|
| plain | 93.8% | 13.4% | **8.4%** | 4.5% | 8.7% |
| cancelling pairs (the method) | 93.7% | 70.6% | **55.4%** | 47.3% | 53.9% |
| chaining (prior-art construction) | 94.9% | 71.0% | **59.6%** | 57.1% | 56.4% |
| oracle (true labels, 1-16 events) | 93.3% | 81.2% | **77.1%** | 74.7% | 66.9% |

At 12 events, per seed:

| condition | seed 6 | seed 7 | seed 8 |
|---|---|---|---|
| plain | 9.4% | 7.4% | 8.4% |
| cancelling pairs (the method) | 56.8% | 54.6% | 54.8% |
| chaining | 61.6% | 59.0% | 58.2% |
| oracle | 76.0% | 76.2% | 79.0% |

**Decision** (`summarize.py --gate chain`): chaining minus the method at 12 events is +4.8, +4.4
and +3.4 points, a mean of +4.2. "Chaining better" needed a mean of at least +5 with every seed
above zero, so the verdict is **NO CLEAR DIFFERENCE**.

**That verdict does not mean the two are equal.** Chaining minus the method, paired, with 95%
bootstrap intervals over test problems (not over seeds):

| test set | difference (points) | 95% interval |
|---|---|---|
| 4 events | +1.3 | +0.1 to +2.5 |
| 8 events | +0.4 | -1.9 to +2.6 |
| **12 events** | **+4.2** | **+1.7 to +6.7** |
| 16 events | +9.8 | +6.9 to +12.6 |
| 12, held-out names and items | +2.5 | -0.0 to +5.0 |

Chaining leads at 12 and 16 events on every seed, and it is not behind on average on any test
set.

**Where the lead comes from** (descriptive; `audit_runs.py`, `check_chain.py`):

* By answer size, at 12 events:
  * answers below 30: equal (chaining 61-65% per seed, the method 62-65%);
  * answers 30-59: chaining 52-57%, the method 34-39%;
  * answers of 60 or more: chaining 3 of 30, the method 0 of 30.
* The training data differ the same way. The sets below were regenerated with the registered code
  (seed 6) and match the statistics each run recorded.

  | changes per training problem | the method: mean answer, share ≥ 30 | chaining | oracle |
  |---|---|---|---|
  | 1-4 | 18.0, 16% | 18.1, 16% | 18.0, 16% |
  | 5-8 | 18.1, 16% | 20.7, 24% | 19.4, 23% |
  | 9-12 | 18.0, 17% | 23.4, 32% | 20.9, 27% |
  | 13-16 | 18.3, 17% | 25.7, 37% | 22.9, 32% |

  The 12-event test problems have a mean answer of 21.2, and 28% of their answers are 30 or more.
* So the method's long examples keep the short problems' answers. Chaining's answers grow with
  length, as real long problems' do (a little more). That fits the gap on answers of 30 or more.
  This is a reading of the data, not a separate experiment.
* Chaining can do this here because labels compose exactly by addition.
* The oracle is still 17.5 points ahead of chaining at 12 events (+15.1 to +19.9). It trains on
  longer problems: 8.5 changes on average, against 5.8.

**Predictions** recorded in `SPEC_CHAIN.md` (`check_chain.py`); two of five held:

1. Verdict CHAINING BETTER: **missed.** The direction held on every seed, but the margin was short
   (+4.2 against +5).
2. Chaining at least 65% at 12 events: **missed** (59.6%). The method scored 55.4% (58.2% in the
   final).
3. Chaining above 10% on answers of 60 or more at 12 events: **missed** by the narrowest margin,
   with 3 of 30, exactly 10%. There are too few such problems to read much into it.
   * The method at 0%: held.
   * At 16 events (descriptive): chaining 9 of 65, the method 2 of 65, the oracle 17 of 65.
4. The method beats plain by at least 15 points on every seed: **held** (+47.4, +47.2, +46.4).
5. Chaining costs at most 3 points at 4 events: **held.** It scored 1.1 points above plain.

**Checks:**

* All 12 jobs exited 0, with no reruns.
* Both kernels recorded the code hashes registered in `SPEC_CHAIN.md`.
* 60 of 60 sets recompute exactly from the raw generations.
* No test problem appears in any training set, and no training example was truncated.
* No parse failures.
* Within each seed, every condition got the same test problems.
* Shortcut rules score at most 8.0% on the 4-, 12- and 16-event test sets.
* GPU: 2.85 hours from the kernel logs (2.9 estimated).

**What it means.** `SPEC_CHAIN.md` registered that NO CLEAR DIFFERENCE would mean "the method is at
least competitive with chaining". The data support a weaker reading:

* the method is close to chaining at 4 and 8 events (chaining +1.3 and +0.4 points);
* it trails at 12 events (by less than the registered margin) and clearly at 16;
* so where labels can be composed, as here, chaining is the better construction.

The method's remaining difference is that it computes no new label. It lengthens a labeled problem
without knowing how to combine answers. Whether that helps where labels cannot be composed is
untested.

## Where answers cannot be composed: against self-improvement (seeds 9-11)

**Question** (`SPEC_MULTI.md`): chaining needs labels that compose. Where they cannot, does the
method beat direct self-improvement (Lee et al. 2025, arXiv 2502.01612)?

* **Task** (`multi.py`):
  * Three people start with 5-30 of one item, and each event changes one person's amount by 1-9.
  * The question asks for the largest final amount.
  * A short problem's label (the maximum) does not say who has what, so labels cannot be chained.
* **Self-improvement as adapted here:**
  * 3 models are trained like plain.
  * It runs 6 rounds of +2 events (5-6 up to 15-16). Each round, 4,000 unlabeled problems are kept
    only when all 3 models agree, and each model trains 500 more steps.
  * Model 0 is evaluated.
  * Its unlabeled problems come from the test distribution, which favours it.
* **Training:** 6,000 steps (doubled by the pilot gate), otherwise as in the earlier tests.
* Registered in `SPEC_MULTI.md` (4cdafb5) before any run of the method on this task.

Means over seeds 9-11:

| condition | 4 events | 8 | **12 (primary)** | 16 | 12, held-out names and items |
|---|---|---|---|---|---|
| plain | 91.9% | 56.4% | **23.3%** | 11.7% | 23.2% |
| cancelling pairs (the method) | 92.2% | 69.3% | **41.7%** | 25.8% | 42.1% |
| self-improvement (model 0) | 85.1% | 52.9% | **24.1%** | 10.7% | 23.4% |
| oracle (true labels, 1-16 events) | 97.7% | 93.9% | **88.5%** | 79.9% | 86.5% |

At 12 events, per seed:

| condition | seed 9 | seed 10 | seed 11 |
|---|---|---|---|
| plain | 21.6% | 24.8% | 23.4% |
| the method | 40.4% | 43.2% | 41.4% |
| self-improvement | 24.8% | 24.8% | 22.8% |
| oracle | 86.8% | 87.4% | 91.4% |

**Decision** (`summarize_multi.py --gate final`):

* Validity holds: the method minus plain was +18.8, +18.4 and +18.0 points, and 15 were needed
  on every seed.
* The method minus self-improvement was +15.6, +18.4 and +18.6, a mean of +17.5. The 95% interval
  over test problems is +15.0 to +20.1.
* Verdict: **METHOD BETTER.**

**Why, and what it does not show** (descriptive; `check_multi.py`, `audit_multi.py`,
`check_pred6.py`):

* **Self-improvement failed here, so the win mostly reflects that failure.**
  * It gained +0.9 points over plain at 12 events (interval -1.3 to +3.0) and lost 6.9 at 4 events.
  * The vote kept labels that were right 94% of the time in round 1 (5-6 events) and 85% in
    round 2. Then they fell to 68%, 43%, 26% and 15%.
  * The models agreed more often as they got worse (44-62% kept in rounds 3-6): the consensus trap
    seen in the toy.
  * Models 1 and 2 were no better (24.3% at 12 events, mean of 6).
* **The method's gain is real but modest.**
  * It was +18.4 points over plain at 12 events (interval +15.7 to +21.1), on every seed.
  * It held with held-out names and items (+18.9) and at 16 events (+14.1).
  * It recovered only 28% of the oracle's gain at 12 events; the oracle was 46.9 points ahead
    (+44.1 to +49.7).
  * In the additive test (`SPEC.md`), the gain over plain was +47.
* **Answer size is not the main limit here.**
  * On answers of 29 or less, the method had 44.3% and the oracle 89.8% (pooled).
  * The method's training answers stay those of 1-4-event problems (largest 55-58). It scored 17%
    on answers of 45 or more, against the oracle's 75%. But such answers are only 5-8% of the
    12-event test set.
  * So most of the gap is in tracking the three amounts.
* **Cost:**
  * no loss at 4 events (0.3 points above plain);
  * training time 1.7 times plain's, and a third of self-improvement's;
  * no unlabeled problems.

**Predictions** (`SPEC_MULTI.md`): five of six held.

* Held: the verdict, validity, the decay of the kept labels, the cost at 4 events, and the
  answer-size prediction.
* The levels missed badly: the method scored 41.7% (60-75% predicted) and self-improvement 24.1%
  (40-60% predicted).

**Checks:**

* All 12 jobs exited 0, once each.
* All runs recorded the registered code hashes.
* 60 of 60 sets recompute exactly from the raw generations.
* No leakage, truncation or parse failures.
* Within each seed, every condition got the same test problems.
* Shortcut rules score at most 6.8% at 12 and 16 events (17.2% at 4).
* GPU: 10.33 hours (10.2 estimated).

**Not compared:** PAR (arXiv 2610.27628).
* It retrofits the same base model as a looped model and trains it with prefix
  self-distillation, which also needs only short labels.
* It was found after registration (`SPEC_MULTI.md`, Addendum 2). It reports far longer
  extrapolation on state tracking: 85% at 16 times the training length.
* So this test shows that the method beats direct self-improvement here, not every method that
  applies.

## What this shows and what it does not

* **It shows** that the toy result transfers to a pretrained language model, with the same
  ordering.
  * Inserted material helps only when it is a real state change: cancelling pairs scored 58.2%,
    while irrelevant sentences (16.1%) and inactive sentences (15.9%) barely beat plain (10.0%).
  * This held under preregistration, on fresh seeds, at no in-distribution cost. It replicated on
    seeds 6-8 (+46.4 to +47.4 points over plain).
* **It does not beat the closest prior art's construction.**
  * Chaining short labeled problems (h1) was close to the method at 4 and 8 events (+1.3 and
    +0.4 points). It led at 12 events (+4.2) and 16 events (+9.8), on every seed (section above).
  * So the claim is "beats plain fine-tuning and clutter controls", not "beats the best known
    way to train long problems from short labels".
* **Where labels cannot be composed, it beats direct self-improvement** (`SPEC_MULTI.md`): +17.5
  points at 12 events, on every seed.
  * But self-improvement gave nothing over plain there.
  * The method recovered only 28% of the oracle's gain, and its gain over plain (+18.4) was much
    smaller than in the additive task (+47).
  * PAR, a looped method that also needs only short labels, was not compared.
* **Large answers:** the method cannot reach answers beyond the short problems' range.
  * Its largest training answer is 57-61 per run in the final (the oracle's is 98-100), and it
    scores 0% on test answers of 60 or more.
  * This is the same small-sum bias as in the toy.
  * Chaining reaches a few of them: 3 of 30 at 12 events and 9 of 65 at 16, against 0 and 2 for
    the method.
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
  * Chaining was compared as data under supervised training, not with h1's reinforcement
    learning.
  * Nothing here tests real benchmarks such as GSM8K, larger models, chain-of-thought outputs or
    naturally written problems.

**In one line:** a simple data rule that beats plain fine-tuning under registration, in narrow
synthetic settings: by about 47 points in an additive task and 18 in a three-person task.
* Where labels compose, chaining short problems (the prior art's construction) does at least as
  well.
* Where they do not, the rule beats direct self-improvement, which failed there. It stays far
  below true long labels.
* A stronger looped method (PAR) was not compared.

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
4. **Registered final:** seeds 3-5, 18 runs. PASS.
5. **Chaining comparison:** seeds 6-8, 12 runs, registered in `SPEC_CHAIN.md` (855a918). NO CLEAR
   DIFFERENCE by the rule; chaining was ahead on every seed.
6. **Three-person test** (`PLAN_MULTI.md`):
   * Pilot 1 failed the learnability gate (88.8% at 4 events). Per the rule, the steps doubled,
     and pilot 2 passed (92.8%, with 61.2 points of headroom).
   * The final was registered in `SPEC_MULTI.md` (4cdafb5) and ran on seeds 9-11, 12 runs:
     METHOD BETTER.

The decision log is in `SPEC_DRAFT.md`. GPU use in the week to 2026-11-14 was about 25.4 of 30
hours on T4 x2 kernels, from the kernel log times:
* 9.22 for the pilots, discovery and final;
* 2.85 for the chaining test;
* 13.37 for the three-person test (pilots 3.04, final 10.33).

## Files and how to reproduce

* `results/{pilot1,pilot2,discovery,final,chain,multi_pilot1,multi_pilot2,multi_final}/`: per-run
  `metrics.json`, raw generations
  (`generations.jsonl.gz`), kernel logs, and the outputs of the scripts below. Adapters are not
  committed; they stay in the Kaggle kernel outputs.
* Commands, run in `research/llm/`:
  * `python3 summarize.py --gate final results/final` gives the first verdict, and
    `--gate chain results/chain` the second.
  * `python3 inspect_run.py results/final` (or `results/chain`) recomputes the metrics and checks
    hashes and leakage.
  * `python3 -I audit_runs.py results/final` (or `results/chain`) runs the shortcut, answer-size
    and raw-output audit.
  * `python3 -I check_predictions.py results/final` and `python3 -I check_chain.py results/chain`
    check the predictions and give the intervals. `check_chain.py` also regenerates and describes
    the training data.
  * Three-person test:
    * `python3 summarize_multi.py --gate final results/multi_final` gives the verdict;
    * `python3 inspect_run.py results/multi_final` recomputes and checks;
    * `python3 -I audit_multi.py`, `python3 -I check_multi.py` and `python3 -I check_pred6.py`, each
      on `results/multi_final`, give the audit, the predictions with intervals, and prediction 6;
    * `python3 answer_support_multi.py` gives the answer-size check (generator only).
* To rerun: `python3 kaggle_llm.py push --slug NAME --jobs jobs/final_a.json`, and the same for
  `final_b.json`, `chain_a.json`, `chain_b.json`, `multi_final_a.json` and `multi_final_b.json`.
