# Pipeline audit after the failed sealed test (2026-11-02)

Goal: find what is actually broken before spending more GPU time. Diagnostics after the test are
post-hoc. They use the dev pools and the **hard_test pools, which are now spent**: no future
method may be confirmed on them. Scripts are in `audit/`; each number below comes from them.

## Ranked findings

### 1. Critical: the pass-through detector of terminal-ratio voting is mostly noise

`selection.mentioned_values` treats every number token in an attempt as "this attempt reached the
value". That includes exponents, coefficients, step numbers, ranges and `= 0` (`x^2 - 5`,
`5 <= k`, `digits 0-9`). Placebo check: the share of attempts that "mention" a value that is
neither an answer of any attempt nor in the problem (`audit/rates.py`):

| value | 0 | 1 | 2 | 3 | 4 | 5 | 12 | 24 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| dev | 41% | 79% | 90% | 67% | 60% | 39% | 15% | 10% | 8% |
| test | 40% | 77% | 89% | 65% | 61% | 39% | 17% | 9% | 5% |

So for small answers L(A) is mostly chance. The test losses sit there. These are answer changes
of trv_vote vs majority on the full 32-attempt pool (`audit/flipkind.py`):

| majority answer type | dev W->R / R->W | test W->R / R->W |
|---|---|---|
| integer < 10 | 4 / 0 | 7 / 13 |
| integer >= 10 | 4 / 1 | 8 / 9 |
| simple fraction | 3 / 0 | 0 / 5 |
| other (sqrt, pi, tuples, expressions) | 0 / 0 | 1 / 1 |

The detector is also blind to non-numeric answers. Written verbatim, only 23% (dev) and 13%
(test) of them are found (nested braces, tuples, pi, letters), so they are never penalised. That
biases the rule toward complex-looking answers. They are 21% of gold answers.

### 2. Major: the development set does not represent the test set

Dev is MATH *train* level 5; test is MATH *test* level 5 (`audit/rates.py`):

| | dev (500) | test (1,190) |
|---|---|---|
| single attempt correct | 62.1% | 54.4% |
| majority of 32 | 69.0% | 65.6% |
| any of 32 correct | 85.6% | 85.8% |
| majority wrong, gold present | 16.6% | 20.2% |

The subject mix also differs (geometry 18% of dev, 10% of test; prealgebra 11% vs 15%). The
7.7-point gap in single-attempt accuracy fits the model having seen the MATH train split in
training. That is a hypothesis, not verified. Either way, dev behaviour (which answers are right,
how attempts fail) differs from test.

### 3. Major: the dev evidence was weaker than its confidence interval said

About 8 families of ideas were tried on the same 500 dev questions: latent votes, confidence
measures, diversity, wording, first hypothesis, a learned scorer, subtractive and multiplicative
pass-through, and a gamma sweep. The winning rule then rested on about 22 changed answers
(7 W->R vs 2 R->W on the full pool). The bootstrap CI covers resampling of questions, not that
search, so it overstated the evidence.

### 4. Power: only effects of about 1 point or more are detectable

On 1,190 test questions the primary statistic had standard error about 0.34 points. A true gain
of about 1 point over the tuned baseline is needed for 80% power. The dev effect (+0.8) was at
that floor even before the selection bias.

### 5. Minor: grading

* Math-Verify uses `signal.alarm` timeouts and treats a timeout as "not equal". There were 9 on
  test and 0 on dev. Those comparisons can depend on machine speed. Dev grading was repeated
  twice with identical clusters and gold.
* Gold is the last `\boxed{}` of the solution. In the 10 multi-box dev solutions the boxes agree,
  except a base-9 answer (`28` then `28_9`); base-n answers are a weak spot of the grader.
* Checked and fine on dev: no reference answer matches two clusters, no cluster merges by
  chaining, no gold collisions.

### 6. Minor: code

* `trv` with the top-eta confidence filter: filtered attempts still count in v(A) and L(A).
* The 16 unit tests pass but use synthetic texts only. None measures chance mentions in real
  attempts, which is how finding 1 went unnoticed.

## Checked and sound

* Generation: dev and test used identical settings (vLLM 0.32.0, T=0.8, top-p 0.95, 32 attempts,
  2,048 max tokens), both workers exited 0, every question complete.
* Data: no duplicate ids, no dev/test problem overlap, every question has a gold answer, under 1%
  of attempts truncated, none finished without an answer.
* Replay, paired bootstrap and the tuned baseline: the confidence-weighted vote tuned on dev
  transferred to test (+1.0 to +1.3 over SC).
* Protocol: preregistration before test access, a single recorded holdout access, journal
  verifies.

## What the next attempt must change

1. **New data with a representative dev split.** Use one source the model was not trained on,
   split three ways: discovery, confirmation and sealed test. The MATH test set is spent.
2. **Discovery and confirmation separated.** A rule found on the discovery half must pass the
   confirmation half unchanged before any sealed test. Count every idea tried.
3. **Placebo controls for any text signal.** Measure the signal's chance rate on values that are
   not candidates. A signal must be clearly above chance on real attempts, not only in unit tests.
4. **Effect bar of about +1 point** over the tuned baseline on confirmation before using GPU on a
   sealed set.
5. Optional: run Math-Verify without signal timeouts (process-level limit) for determinism.
