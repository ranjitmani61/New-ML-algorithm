# Test-time reasoning: development results (honest status)

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
