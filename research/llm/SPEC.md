# Registered final: cancelling-event insertion in a language model

Registered on 2026-11-07, before any discovery result existed: both discovery kernels were still
running. The final runs only if discovery's go/no-go (in `SPEC_DRAFT.md`) passes. Results are
reported against this file unchanged, pass or fail. Any deviation goes into an erratum here.

## Code (sha256; verified in every run's metrics.json)

* `problems.py`: d4d3128bd64043ac3449b5bc04a0bc995c3cdb6aa9525236f9140697e63b61a9
* `run_llm.py`: 5a684fec81f3bcad5b85db2e56acae8dd63440dcc2f02f2457e6b21f3c1be08a
* Runner: `kaggle_llm.py`. Jobs: `jobs/final_a.json` and `jobs/final_b.json` (18 runs).

## Setup

* **Model:** Qwen2.5-0.5B-Instruct (Kaggle Models `qwen-lm/qwen2.5/transformers/0.5b-instruct/1`).
* **Fine-tuning:** LoRA rank 16 (alpha 32) on all attention and MLP projections; fp16 with fp32
  LoRA weights on T4 GPUs.
* **Training:** 3,000 steps, batch 16, learning rate 2e-4 (50-step warmup, then linear decay),
  24,000 training problems.
* **Task:** additive word problems. Labeled training problems have 1-4 events; expansions extend
  them to at most 16 changes. The model must answer with the number only.
* **Evaluation:** greedy decoding, strict parse. 500 fresh problems at each of 4, 8, 12 and 16
  events, plus 500 problems at 12 events with held-out names and items.
* **Conditions:** plain, irrelevant, inactive, cancel_spread (the method), cancel_adjacent,
  oracle (true labels at 1-16 events).
* **Seeds:** 3, 4, 5 (never used before).

## Primary metric

Exact-match accuracy at 12 events (train vocabulary), per seed.

## Pass criteria (unchanged from SPEC_DRAFT.md; all must hold)

1. `cancel_spread` beats `plain` by at least 15 points on every seed.
2. `cancel_spread` beats the better of `irrelevant` and `inactive` (higher mean over seeds) by at
   least 10 points on average.
3. `cancel_spread` recovers at least 50% of the oracle's gain over `plain` (means over seeds).
4. `cancel_spread` loses at most 3 points against `plain` at 4 events (mean over seeds).

Evaluated mechanically by `summarize.py --gate final`.

## Predictions (recorded, not criteria)

* Gains shrink at 16 events and on large answers.
* `cancel_adjacent` lands close to `cancel_spread`.
* The held-out vocabulary result at 12 events is within 5 points of the in-vocabulary result.

## Erratum (added 2026-11-07, after the final; the registered text above is unchanged)

* "50-step warmup, then linear decay" is loose wording. The code multiplies a 50-step linear
  warmup by a linear decay that starts at step 0 (the factor at step 49 is 0.984, not 1). The
  code ran exactly as registered, so this is a wording difference only.

## Result

PASS on all four criteria. See `REPORT.md`.
