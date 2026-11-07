# Inverse-pair insertion: final result

**Verdict: PASS.** All three pass criteria registered in `SPEC.md` (commit 0b59f7b, before any
run) hold. Both registered predictions about the method's limits also held.

## What the method is

To train a model on longer problems without long labels, insert cancelling pairs (+a, -a) at
random positions into short labeled problems. The input becomes longer and contains more real
terms, but the label is unchanged, so no long label is ever computed. It is a data rule, so it
needs no change to the model or the loss.

## Final test

Run on Kaggle (2 kernels, T4 x2, 3.27 GPU hours). The registered code ran unchanged, with sha256
`2aa08eb5...` verified in both kernels; all 30 runs (6 conditions x 5 fresh seeds) finished with
exit code 0. Setup: transformer of width 128 with 4 layers; 20,000 steps; labels only for 1-8
terms; expansions up to 32 terms.

| condition | 8 terms | 16 | **24 (primary)** | 32 | 48 |
|---|---|---|---|---|---|
| baseline | 99.9% | 4.1% | **1.9%** | 1.7% | 1.1% |
| sigmoid attention (known method) | 99.8% | 5.3% | 0.9% | 0.3% | 0.2% |
| filler tokens (meaningless lengthening) | 99.7% | 43.0% | 4.8% | 1.9% | 0.5% |
| zero terms (inactive lengthening) | 99.7% | 63.6% | 12.7% | 4.5% | 0.8% |
| **inverse pairs (the method)** | 98.9% | **92.3%** | **84.7%** | **75.5%** | 7.6% |
| oracle (true labels to 32 terms) | 99.6% | 97.9% | 92.2% | 88.3% | 5.5% |

Each figure is the mean over seeds 100-104 (per-seed values are in `results/report.txt`).

**Pass criteria (24 terms):**

1. Inverse pairs beat the baseline by at least 30 points on every seed: +81, +84, +81, +83, +85.
   **Pass.**
2. Inverse pairs beat fillers by at least 20 points: +79.8. **Pass.**
3. Inverse pairs recover at least 70% of the oracle's gain: 91.6%. **Pass.**

**Registered predictions (both held):**

* Beyond the expansion range (48 terms), inverse pairs stay below 50%: 7.6%. The oracle fails
  there too (5.5%), so neither extrapolates beyond the structure it was trained on.
* Inverse pairs are weaker than the oracle on large sums, because their labels never exceed
  the short problems' range. At 24 terms the accuracy by |sum| is:

  | |sum| | 0-9 | 10-19 | 20-29 | 30-39 | 40 or more |
  |---|---|---|---|---|---|
  | inverse pairs | 99.4% | 99.4% | 96.9% | 81.6% | 12.2% |
  | oracle | 100% | 99.1% | 92.7% | 89.7% | 64.9% |

## What this shows and what it does not

* It shows that length alone does not help: the inserted material has to be **real non-zero
  terms**. Cancelling pairs scored 84.7%, inactive zeros 12.7% and meaningless fillers 4.8%. With
  only short labels, inverse pairs come within 7.5 points of training on true long labels.
* **Control caveat (found in the post-final audit):**
  * The filler control is length-mismatched: its training inputs reach at most 41 tokens, with
    only 8% longer than 33. Inverse pairs reach 65 tokens, with 28% longer than 33. So
    criterion 2 (against fillers) compared with a weaker control than intended.
  * Zero terms are the length-matched control, with the same token lengths as inverse pairs. They
    also lose, by 72 points, so the conclusion stands.
  * Adjacent pairs, which a model could skip locally, work almost as well (86% vs 90% at 8
    terms; ledger #21). So "the model is forced to sum scattered terms" is not the right
    explanation. What matters is that non-zero cancelling terms are present.
* It costs **1 point in-distribution** (8 terms: 98.9% vs 99.9%).
* It does **not** make models extrapolate beyond the lengths the expansions cover.
* It does **not** extend to order-dependent composition. On S3 state tracking it teaches only
  the commutative part: parity is tracked at 97-100% on 16-element words while the rotation part
  stays at chance. It also lowers in-distribution accuracy on the order-dependent part
  (6 elements: 74% vs 87%). This was confirmed at an adequate training budget in the audit (ledger
  #22).
* It is shown on one synthetic task family (signed sums) with small transformers; nothing here
  shows it on language models.
  * Follow-up (2026-11-07): a registered test on a language model (Qwen2.5-0.5B-Instruct, templated
    word problems) passed. See `research/llm/REPORT.md`, including what it does not show.

## Prior art and novelty

The broad principle of building long training problems with known answers from short labeled ones
is not new. h1 (arXiv 2510.07312, 2025) chains short GSM8K and MATH problems, including an
answer-preserving variant, for reinforcement learning of language models. Two things are
specific to this work:

* the mechanism: cancelling elements inserted inside one input, rather than sub-problems chained
  in series;
* the controlled comparison showing that inserted material must be computationally active.

This is a solid controlled result and a simple data rule, not a new paradigm.

## How it was reached

* 17 ideas and references were tested on two testbeds, plus two baselines, and logged in
  `research/tournament/LEDGER.md`. Predictions were written before each run; three were declared
  follow-ups added after a discovery result.
  * 2 passed confirmation: inverse pairs and zero terms.
  * 1 is the ceiling reference: the oracle.
  * 14 were discarded.
* Code was deleted for the discarded ideas, except sigmoid attention and fillers, which are kept
  on purpose as the reference and the control. (An earlier version of this report said "20
  logged, 17 discarded"; that count was wrong.)
* Inverse pairs went through three stages:
  1. discovery seeds 0-1: 90.3% vs 6.6% at 8 terms;
  2. preregistered confirmation on seeds 2-4: 88.3% vs 7.6%;
  3. this registered final at 4x the length scale.
* Earlier directions that failed are documented too: latent and terminal-ratio voting
  (`reasoning/REPORT.md`, `reasoning/AUDIT.md`) and metamorphic self-training
  (`research/DIRECTION.md`).

## Erratum and pipeline limits (post-final audit)

* **Registration wording:** `SPEC.md` says "half of each batch is augmented". The registered code
  augments each example with probability 0.5, so half on average, not exactly half. `SPEC.md` is
  left unchanged as registered.
* **Final models not saved:** `final.py` records metrics only. Any further mechanism check on the
  final models needs retraining.
* **References not tested:**
  * a scratchpad or chain-of-thought format (running partial sums as output);
  * a looped transformer trained with a loop count tied to length;
  * NoPE was tested in the audit (ledger #20): 5.0% at 8 terms, below the baseline.
  * "Beats known methods" holds only for the methods actually tested: sigmoid attention, SSMax,
    random position offsets and fillers.
* **Wording to avoid:** "3x the training length". The labeled length was 8. Inputs up to 32 terms
  did appear in training, through the expansions, so the primary length (24) is beyond the
  labeled length but within the trained input length.

## Reproduce

* Final: `python3 research/final/final.py --out results.json` (or `kaggle_final.py push ...`).
* Report: `python3 research/final/report.py research/final/results/kernel_*.json`.
