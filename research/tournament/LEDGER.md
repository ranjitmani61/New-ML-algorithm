# Idea tournament ledger

Every idea tried is listed here, kept or not, so the number of attempts is visible and the
multiple-comparison risk is honest. Code for discarded ideas is deleted from the active bench (it
remains in git history).

## Protocol (fixed before batch 1)

* **Testbed:** signed sums of single-digit terms. Train only on 1-4 terms; test greedy accuracy on
  fresh 5-8-term expressions (1,000 per length). In-distribution = mean accuracy on 1-4 terms.
* **Model and budget (identical for every idea):** 3-layer, width-64 rotary transformer;
  4,000 steps, batch 128, AdamW, peak learning rate 2e-3, warmup 200, cosine decay.
* **Discovery seeds:** 0, 1. **Confirmation seeds:** 2, 3, 4 (untouched until an idea is kept).
  **Final:** fresh seeds at larger scale on Kaggle, metric registered before the run.
* **Keep rule:** mean 8-term accuracy at least +10 points over the baseline, both seeds above the
  baseline, and in-distribution accuracy at least 95%. Otherwise discard.
* **Comparison bar:** a candidate must also beat the best reference (known method) to count as a
  new contribution.
* **Confirmation (fixed now, before any winner):** a kept idea is rerun unchanged on seeds 2, 3, 4
  with the baseline and the best reference on the same seeds. It passes if the mean long-length
  gain over the baseline is at least +10 points, all three seeds beat the baseline, and it beats
  the best reference by at least +5 points on average. Then a novelty check, then the Kaggle final
  with a specification registered before the run.
* **Process note:** batch 1 first lost about 30 minutes because a wait loop matched its own
  command line (`pgrep -f`). Waits now use explicit process ids or result files.

## Ideas

| # | idea | kind | hypothesis | prediction (before run) | result (discovery) | decision |
|---|---|---|---|---|---|---|
| 1 | metamorphic self-training (earlier pilot) | candidate | equivariant relations give label-free signal that escapes the consensus trap | +15 points over majority at 8 terms | one seed: 8 terms 6% (majority 8%, none 7%, oracle 84%); 5 terms 56% vs majority 33% | discarded (fails 8-term target) |
| 2 | baseline | baseline | vanilla softmax transformer | 8 terms 5-15% | 8 terms 6.6% (8.4, 4.8); 6 terms 37% | reference point |
| 3 | ssmax | reference (2501.19399) | length-scaled logits keep attention sharp | small to moderate | 6.0% (-0.6) | discarded |
| 4 | sigmoid attention | reference (2409.04431) | unnormalised attention can output sums | large if training finds the sum solution | 9.1% (+2.5) | discarded; best reference, used in confirmation |
| 5 | random position offsets | reference (randomised positions) | exposes larger position values | small | 8.2% (+1.5) | discarded |
| 6 | filler tokens | reference (random spaces) | lengthens inputs without new terms | 0 to +20 | 7.9% (+1.3); 6 terms 51% | discarded |
| 7 | identity-element insertion (NEIA) | candidate | errors come from structural length; zero terms cover lengths 5-8 with exact labels | +30 or more | 16.9% (+10.3: +7.6, +13.0); gains only on sums below 10 | kept (barely), to confirmation |
| 8 | inverse-pair insertion (NPIA) | candidate | pairs (+a, -a) cover length and non-zero count, labels unchanged | +40 or more | 90.3% (+83.7: 87.7, 92.8); seed 0 by sum: 99.7% below 10, 41% at 20-29, about 0 at 30 or more; 10/12/16 terms 38/14/7% | kept, to confirmation |
| 9 | weight extrapolation along difficulty (WED) | candidate | theta4 + a (theta4 - theta3) continues the difficulty direction | none or harm (wild card) | destroys the model: 1-4 terms 16% (a=0.5), 0.3% (a=1) | discarded |
| 10 | risk extrapolation across lengths (REx) | candidate (known method, new use) | low risk variance across lengths favours a length-invariant mechanism | 0 to +10 | 4.9% (-1.8); lowers 4-term accuracy | discarded |
| 17 | oracle (true labels on 1-8 terms) | reference (upper bound) | violates the short-only constraint; calibrates the ceiling | high | 94.3% (93.0, 95.5); seed 0 by sum: 100% below 10, 78% at 20-29, 25% at 30-36; 10/12/16 terms 35/12/5% | ceiling; npia recovers about 95% of its gain |
| 18 | shuffled self-duplication (label doubled) | candidate | covers long inputs with large sums, which zero-sum insertion never shows | better on large sums, weaker on small | 29.3% (+22.7) | discarded: dominated by npia |
| 19 | inverse pairs + duplication | candidate | structure from pairs, value range from duplication | at least npia at 8 terms; above 50% on sums of 20 or more | 85.4% (npia 90.3%); seed 0 by sum: 54% at 20-29 (npia 41%), 12% at 30-36 (npia 1%), but 92% at 10-19 (npia 97%); about 43% on sums of 20 or more | discarded: prediction failed, worse than npia overall |

Ideas 17-19 were added after seeing npia's seed-0 result and its mechanism check (large sums
fail). They are follow-ups found on discovery data, and they are counted here as such.

## Batch 2 (testbed 2: S3 state tracking), written before running

Train words of length 1-6, test 7-16; chance is 16.7%. Same model, budget and keep rule, with
16-element accuracy in place of 8-term accuracy. Here a longer word means genuinely more
sequential composition, so a gain cannot come from covering length alone.

| # | idea | kind | hypothesis | prediction (before run) |
|---|---|---|---|---|
| 11 | baseline | baseline | vanilla transformer | 1-6 learned; 12-16 near chance |
| 12 | sigmoid attention | reference | unnormalised sums help the counting parts (parity, rotation count) | small to moderate |
| 13 | random position offsets | reference | larger position values | small |
| 14 | identity insertion | candidate | covers length only | 0 to +10 |
| 15 | adjacent inverse pairs | candidate | long words with exact labels; risk: the model learns to skip pairs | small to moderate |
| 16 | short identity blocks (length 2-3) | candidate | richer local identities, harder to shortcut | small to moderate |

## Confirmation (seeds 2, 3, 4; protocol fixed before any winner)

| idea | 8 terms, mean | gain over baseline per seed | vs best reference (sigmoid, 9.4%) | min in-distribution | verdict |
|---|---|---|---|---|---|
| baseline | 7.6% (2.2, 9.6, 11.1) | | | 99.8% | |
| inverse-pair insertion (#8) | **88.3%** (89.6, 88.3, 86.9) | +87.4, +78.7, +75.8 | +78.8 | 99.5% | **PASS** |
| identity-element insertion (#7) | 37.0% (28.6, 46.3, 36.0) | +26.4, +36.7, +24.9 | +27.5 | 99.1% | PASS (large seed variance) |

What the confirmed result means and does not mean, from the mechanism checks:

* It shows that inserting cancelling non-zero pairs into short labeled problems gives most of the
  benefit of long labeled data (oracle 94.3% on discovery seeds) without any long label.
* It does not extrapolate beyond the covered length (10/12/16 terms: 38/14/7%, the same as the
  oracle trained to 8 terms), and it is biased toward small sums because the labels never change.
* It is shown only for commutative sums so far. The S3 batch tests whether it carries over to
  order-dependent composition.

## Novelty check for inverse-pair insertion (after confirmation)

Five alphaXiv searches (identity or inverse insertion, zero-sum or label-preserving construction,
cancelling elements, relator or Tietze moves, state-tracking augmentation).

* **Closest prior art: h1 (arXiv 2510.07312, 2025).** It builds long-horizon training problems
  from short labeled ones (GSM8K, MATH) by chaining them. One variant, "transformation chaining",
  rewrites a parameter in terms of an earlier answer so the final answer is unchanged. The
  high-level principle (long problems with known answers from short labeled ones, including
  answer-preserving rewrites) is therefore not new.
* **What differs here:**
  * the mechanism: cancelling elements inserted inside one input, rather than serial chaining of
    sub-problems;
  * the setting: supervised length generalization in small transformers;
  * the controlled finding: the inserted material must be computationally active. Inverse pairs
    gave 88%, zeros 37%, fillers 8%, and true long labels 94%.
* Related: random spaces and fillers (our control), zero-padding of digits, self-improvement
  (2502.01612), theory that a few long strings help (2610.35715).

Assessment: a solid controlled result and a simple, general data rule, in an area where the core
idea exists. Not a new paradigm.

## Batch 2 interim: mechanism check on S3 (seed 0, before the batch finished)

| model | 6 elements exact / parity right | 12 elements | 16 elements |
|---|---|---|---|
| baseline | 56% / 100% | 18% / 53% (chance) | 17% / 48% (chance) |
| adjacent inverse pairs (#15) | 51% / 100% | 33% / 98% | 28% / 85% |

Parity (even or odd permutation) is the commutative part of S3 composition: a sum mod 2.
Inverse pairs make the model track it over long words, while the order-dependent rotation part
stays at chance (exact accuracy is about parity accuracy times 1/3). So cancelling-pair insertion
transfers length generalization to the commutative part of a computation, consistent with the
sum result, and not to the non-commutative part. Caveat: at 4,000 steps even in-distribution
rotation tracking is incomplete (6 elements: 51-56% exact), so this testbed is under-trained, and
the keep rule's 95% in-distribution check cannot be met by any S3 idea at this budget.
