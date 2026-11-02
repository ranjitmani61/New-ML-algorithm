# Step 1: prior art for cancelling events in language-model word problems (2026-11-02)

Question: would training a language model on short word problems expanded with *relevant,
net-zero* events (a quantity gained and later lost) be new, and how should it be tested?

## What exists (alphaXiv searches and a full read of the closest paper)

* **Irrelevant-information robustness, studied as benchmarks:** GSM-IC (2302.00093), GSM-Symbolic
  and its NoOp clauses (2410.05229), GSM-DC (2505.18761), and a mechanistic study (2609.17804).
* **Training with added irrelevant information:**
  * PROBLEMATHIC (2406.15444, ICLR 2025 workshop) adds variables that by construction are
    unrelated to the problem's quantities, in different units. Fine-tuning on them improves
    robustness by about 8%. The test is robustness to distractors, not longer problems.
  * DAGGER (2601.06853) and iterative distractor generation (2510.08615) are similar in spirit.
* **Longer-horizon training from short labeled problems:** h1 (2510.07312), which chains problems
  for reinforcement learning and includes an answer-preserving variant, and Reasoning Cache
  (2602.03773).
* **Longer word problems:** an evaluation study (2405.14804).

## The gap

No work found trains with **relevant, net-zero events on the problem's own quantities** (for
example, "she bought 5 more apples and later gave 5 away") to reach **problems with more
state-changing steps than any labeled one**. This is the direct analogue of inverse-pair
insertion. Not finding it is not proof that it is new.

## Design implications from our own results

1. The toy results predict gains only where the problem's state changes combine additively
   (inventories, balances). For order-dependent steps (doubling, halving) S3 predicts little or no
   gain and a possible in-distribution cost. So test the two families separately.
2. Comparisons needed for any claim:
   * plain fine-tuning;
   * irrelevant-variable augmentation (PROBLEMATHIC or NoOp style), which separates "robustness to
     clutter" from "more steps";
   * h1-style chaining;
   * true long labels (the ceiling).
3. Controls carried over from the audit:
   * length-matched inactive insertions;
   * adjacent versus spread events;
   * saved models, so mechanism checks need no retraining.

## Compute reality

GPU: 1.01 h left until the weekly refresh on 2026-11-07. TPU: 20 h, but language-model
fine-tuning there needs a JAX language-model stack that is not built. A credible test (a small
instruct model, LoRA fine-tuning, generation-based evaluation on longer problems, several seeds)
needs several GPU hours, so a full run is realistic after 2026-11-07 or after a TPU port.
