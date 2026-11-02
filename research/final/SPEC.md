# Final test (Kaggle): inverse-pair insertion at larger scale

Registered before any run of `final.py` on Kaggle. The commit that adds this file is the
registration; results are reported against it unchanged, pass or fail.

## Claim under test

Inserting cancelling pairs (+a, -a) into short labeled problems lets a transformer that only has
short labels learn much longer inputs. It recovers most of the benefit of having long labels, and
inserting meaningless or inactive material does not.

## Setup (all fixed)

* **Task:** signed sums of single-digit terms. Answer = sign + three digits, least significant
  first. Labeled training problems have 1-8 terms.
* **Model:** pre-norm causal transformer with rotary positions; width 128, 4 layers, 4 heads,
  MLP 512.
* **Training:** 20,000 steps, batch 256, AdamW (weight decay 0.01), peak learning rate 1e-3,
  warmup 500, cosine decay to 10%, gradient clipping 1.0. Half of each batch is augmented when
  the condition uses an expansion.
* **Seeds:** 100, 101, 102, 103, 104 (never used before). Evaluation uses fresh expressions from
  a separate generator per seed: 1,000 per length; greedy decoding.
* **Test lengths:** 8 (in-distribution), 16, 24, 32 (beyond the labeled length, within the
  expansion range) and 48 (beyond both).

## Conditions

| name | training data |
|---|---|
| baseline | labeled 1-8 terms |
| sigmoid | labeled 1-8 terms, sigmoid attention (best known method in the pilot) |
| fill | 1-8 terms plus neutral filler tokens up to 32 items (meaningless lengthening) |
| zeros | 1-8 terms plus zero terms up to 32 terms (inactive lengthening) |
| pairs | 1-8 terms plus cancelling pairs (+a, -a) up to 32 terms (**the method**) |
| oracle | labeled 1-32 terms (upper bound; uses long labels) |

## Primary metric and pass criteria

Primary metric: accuracy at 24 terms. The claim passes only if all of these hold:

1. **pairs** beats **baseline** by at least 30 points, on all five seeds.
2. **pairs** beats **fill** by at least 20 points on average.
3. **pairs** recovers at least 70% of the oracle's gain over the baseline (mean over seeds).

## Predictions also registered (not pass criteria)

* At 48 terms (beyond the expansion range) **pairs** stays below 50%: no claim of extrapolation
  beyond the covered length.
* **pairs** is weaker than **oracle** on large sums, because its labels never exceed the short
  problems' range.

## Reporting

All six conditions times five seeds, per-length accuracy, accuracy by |sum| at 24 terms, run
times, and any failed run. No condition, seed or metric is dropped or added after the run.
