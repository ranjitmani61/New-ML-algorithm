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

## Ideas

| # | idea | kind | hypothesis | prediction (before run) | result (discovery) | decision |
|---|---|---|---|---|---|---|
| 1 | metamorphic self-training (earlier pilot) | candidate | equivariant relations give label-free signal that escapes the consensus trap | +15 points over majority at 8 terms | one seed: 8 terms 6% (majority 8%, none 7%, oracle 84%); 5 terms 56% vs majority 33% | discarded (fails 8-term target) |
| 2 | baseline | baseline | vanilla softmax transformer | 8 terms 5-15% | pending | reference point |
| 3 | ssmax | reference (2501.19399) | length-scaled logits keep attention sharp | small to moderate | pending | |
| 4 | sigmoid attention | reference (2409.04431) | unnormalised attention can output sums | large if training finds the sum solution | pending | |
| 5 | random position offsets | reference (randomised positions) | exposes larger position values | small | pending | |
| 6 | filler tokens | reference (random spaces) | lengthens inputs without new terms | 0 to +20 | pending | |
| 7 | identity-element insertion (NEIA) | candidate | errors come from structural length; zero terms cover lengths 5-8 with exact labels | +30 or more | pending | |
| 8 | inverse-pair insertion (NPIA) | candidate | pairs (+a, -a) cover length and non-zero count, labels unchanged | +40 or more | pending | |
| 9 | weight extrapolation along difficulty (WED) | candidate | theta4 + a (theta4 - theta3) continues the difficulty direction | none or harm (wild card) | pending | |
| 10 | risk extrapolation across lengths (REx) | candidate (known method, new use) | low risk variance across lengths favours a length-invariant mechanism | 0 to +10 | pending | |

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
