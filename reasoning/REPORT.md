# Test-time reasoning: results (honest status)

Part A: easy setting (GSM8K / MATH-500), two candidates, both rejected on development.
Part B: hard setting (MATH level 5), terminal-ratio voting, preregistered and tested once.

## Part A: GSM8K and MATH-500

Goal: a new test-time algorithm for LLM reasoning that beats existing methods at equal
generated-token cost. Discipline: everything below is on **development data only**
(gsm8k_dev = 400 random GSM8K train questions, math_dev = 150 random MATH-500 questions).
The test pools (all 1,319 GSM8K test + the other 350 MATH-500) stay sealed until a candidate
clears development and is preregistered. No candidate has cleared development yet.

## Data

Qwen2.5-Math-1.5B-Instruct, 32 attempts per question, temperature 0.8, top-p 0.95, generated on
Kaggle T4s (`kaggle_runner.py`). 17,600 attempts, 6.3M tokens, with per-token confidence.
Methods are compared by replaying random orderings of each question's pool (`lab.py`);
cost = generated tokens per question.

| | GSM8K dev | MATH dev | all |
|---|---|---|---|
| single attempt accuracy | 94.5% | 71.0% | 88.0% |
| majority of 32 | 95.5% | 77.3% | 90.55% |
| any of 32 correct (ceiling) | 97.3% | 90.0% | 95.3% |
| attempts with a falsified arithmetic step | 0.2% | 2.1% | 0.7% |
| accuracy of falsified attempts | 54% | 32% | 36% |

The whole gain from repeated sampling is 2.5 points (88.0% to 90.55%).

## Baselines (accuracy at tokens/question)

| method | best points |
|---|---|
| SC (majority of k) | k=1: 88.04% @ 358; k=4: 89.65% @ 1,433; k=32: 90.55% @ 11,474 |
| confidence-weighted vote (DeepConf-style, offline) | k=2: 89.18% @ 716; k=16 top-50%: 90.80% @ 5,739 |
| ASC (Beta stopping) | c=0.76: 90.34% @ 1,138; c=0.9: 90.55% @ 2,078 |
| ESC (window stopping) | w=2: 90.34% @ 1,188 |
| UAB (cross-question allocation) | avg 3: 90.25% @ 1,332 |

DeepConf-online is not run: its 16-attempt warm-up alone (~5,700 tokens) exceeds the budget at
which ASC already reaches majority-of-32 accuracy.

## Candidate 1: verified-step self-consistency (VSC) — no effect

Exact, zero-token checking of inline arithmetic claims, down-weighting attempts with false
steps, plus final-step repair. The check is precise (1 false alarm on 159 MATH reference
solutions; catches 483/496 planted errors) and its signal is real (36% vs 88% accuracy), but
only 0.7% of attempts contain a detectable slip. Effect on accuracy at every budget: within
±0.02 points. Final-step repair was harmful (7 right answers made wrong, 2 wrong made right).
Reason: this model's errors are semantic (wrong setup), not arithmetic.

## Candidate 2: learned optimal stopping (least-squares Monte Carlo) — no meaningful gain

`stopping.py`: stop or draw another attempt to maximise expected correctness minus a token
cost; continuation values fitted by Longstaff–Schwartz backward induction on replayed pools.
Question-level 5-fold cross-validation on development.

* Comparing two separately fitted values (stop vs continue) stopped on noise: 89.2% @ 1,007.
* Regressing the same-path advantage of continuing fixed that: 90.0% @ 920–940.
* Against a fair baseline grid, the gain is +0.3 points at ~920 tokens, +0.1 at ~1,010 and
  none at ≥1,140 tokens. Confidence features add about nothing over answer counts; the
  correctness target is not clearly better than the majority target. Learned stopping from
  answer statistics already exists (RL-Guided Sampling, arXiv 2606.03102).

## Conclusion so far

For this model and these benchmarks, adaptive sampling (ASC/ESC) already captures nearly all
of the repeated-sampling gain at about 3 attempts, leaving little for any stopping or weighting
rule. The remaining headroom is in **selection** on harder problems: on MATH dev the
majority is right on 77.3% but some attempt is right on 90.0%. Neither candidate goes to the
sealed test set.

Files: `results/dev_results.json` (all numbers above), `lab.py` (baselines, replay),
`stopping.py` (candidate 2), `benchmarks.py` (grading, step checker), `test_reasoning.py`.

Note: Part A used the string-normalisation grader. Part B found that grader misses equivalent
answers on hard problems (below); Part A's conclusions are null results and are not affected
in direction, but its exact numbers would shift slightly under the Part B grader.

## Part B: MATH level 5 (hard setting)

### Data

Same model and sampling (Qwen2.5-Math-1.5B-Instruct, 32 attempts, T=0.8, top-p 0.95), generated
with vLLM on Kaggle T4s, max 2,048 new tokens. **Development:** 500 random MATH *train* level-5
problems (seed 20261101). **Sealed test:** all 1,190 MATH *test* level-5 problems that are not
in MATH-500. 730 tokens per attempt on average.

### Grading fix (applies to every method)

String normalisation mis-graded and mis-clustered many hard answers (`\frac{\sqrt{6}}2` vs
`\frac{\sqrt{6}}{2}`, `(-3/2,3)` vs `(-1.5,3)`, units text, AMC letter answers vs option
values). `grading.py` clusters answers by normalisation or two-way Math-Verify equivalence, maps
multiple-choice letters to the option text in the problem, and grades with Math-Verify. This
moved majority-of-32 accuracy on dev from 65.0% to 69.0%; all 20 changed verdicts were checked
by hand and are real matches. Clustering is label-free, so all methods see identical clusters.

| dev (500) | value |
|---|---|
| single attempt | 62.2% |
| majority of 32 | 69.0% |
| any of 32 correct | 85.6% |
| majority wrong, correct answer present | 83 questions (51 of them: correct answer ranked 2nd) |

### What does not separate a wrong majority from the right minority

On the 83 questions where the majority is wrong but the right answer is in the pool, the
right cluster is no better than chance on: mean confidence (tail, mean, lowest window, logprob),
chain diversity among supporters (Jaccard of written values; a design-effect vote), "verify" /
"wait" wording, or the first hypothesis, **latent votes** (other attempts mentioning the answer
as support): the wrong majority had the higher mention rate in 39 of 50 decided cases. A
conditional-logit scorer over all these features gave +1.2 to +2.4 points (CV), with the
mention count entering with a *negative* weight.

### Terminal-ratio voting (new)

For candidate answer A with v(A) attempts answering it, let L(A) be the number of attempts that
answer something else but write the value A in their reasoning (values from the problem text
excluded). tau(A) = v(A) / (v(A) + L(A)) is the fraction of attempts that reached the value A
and stopped there. Score(A) = W(A) * tau(A)^gamma, where W is the vote mass of any base rule
(count or confidence-weighted); gamma = 0 is the base rule. Zero extra tokens, one parameter
(gamma = 2, chosen on dev; the effect is flat for gamma in [1.5, 3]).

Rationale from the dev failures: wrong majorities are often waypoints that other attempts pass
through and then transform (72 vs 1152 = 72 * 16; 100 vs 199; one root -3/5 vs both roots
{3/5, -12/5}; 2/3 vs 2*sqrt(3)/3). Prior art: RISC (arXiv 2606.05054) notes the same failure
(majority stops at an intermediate step) and addresses it with a learned ranker over embedding
features; DDRG (2610.18606) aligns claim graphs with an LLM verifier (extra calls); ConsensusBench
(2609.04648) *rewards* shared intermediate values for RL training. I found no method that scores
answers by exact pass-through counts. The novelty is a simple rule, not a new paradigm.

### Development result (paired over the same replayed attempts, equal tokens)

Baseline "conf" = confidence-weighted vote with (measure, top-eta filter) tuned on dev at each k
(12 settings per k; this favours the baseline). trv_conf = the same weights times tau^2.

| k (tokens/question) | SC | tuned conf | trv_conf | trv_conf - conf [95% CI] |
|---|---|---|---|---|
| 4 (2.9k) | 66.2% | 67.2% | 67.4% | +0.28 [-0.11, +0.66] |
| 8 (5.8k) | 67.9% | 68.9% | 69.4% | +0.59 [+0.10, +1.11] |
| 16 (11.7k) | 68.6% | 69.7% | 70.6% | +0.81 [+0.17, +1.49] |
| 32 (23.3k) | 68.8% | 69.6% | 70.6% | +1.00 [+0.00, +2.20] |

Primary dev statistic (mean over k = 8, 16, 32): **+0.80 points [+0.19, +1.49]**. Over plain SC:
+1.6 to +2.0 points. The gain concentrates in intermediate algebra and counting & probability;
at k = 16, 22 questions are helped and 11 hurt. Effects of this size are small: on dev this is
about 4 questions out of 500.

### Preregistration

`results/hard_spec.json` (sha256 in journal event 22 of `results/journal.sqlite`) fixes gamma,
the per-k baseline settings, k values, 40 orderings, the seed and the primary statistic: mean
over k in {8, 16, 32} of accuracy(trv_conf) - accuracy(conf), paired bootstrap over test
questions, pass iff the 95% interval is above 0. Committed before the test pools were
downloaded. `hard_eval.py test` refuses an unregistered spec and records a single holdout access.

### Sealed test result

TEST_RESULT_PLACEHOLDER
