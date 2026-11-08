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

## Batch 2 results (S3, discovery seeds 0-1): everything discarded

| idea | in-distribution (1-6) | 8 | 12 | 16 | gain at 16 | decision |
|---|---|---|---|---|---|---|
| baseline | 89.3% | 27% | 18% | 17.2% | | (in-distribution below 95% even here) |
| identity insertion | 90.3% | 34% | 15% | 16.3% | -0.9 | discarded |
| adjacent inverse pairs | 85.2% | 35% | 33% | 28.7% | +11.5 | discarded (in-distribution drops; parity plateau) |
| short identity blocks | 81.4% | 35% | 32% | 31.5% | +14.2 | discarded (in-distribution drops; parity plateau) |
| sigmoid attention | 91.5% | 40% | 18% | 16.7% | -0.5 | discarded |
| random position offsets | 89.5% | 31% | 16% | 16.8% | -0.5 | discarded |

Conclusion: answer-preserving insertion does not carry over to order-dependent composition. It
teaches the commutative part (parity, about 1/3 exact), at a cost in in-distribution accuracy.
The final test is therefore limited to the commutative setting where the method was confirmed.

## Cleanup

Code for every discarded idea (SSMax, random offsets, REx, weight extrapolation, duplication, the
S3 expansions and references) was deleted from the active bench after this batch. It remains in
git history at commit 2fff9a8. Kept: baseline, sigmoid (best reference), fill (control), zeros
and inverse pairs (passed confirmation), oracle (ceiling).

## Final (registered, Kaggle): PASS

Inverse pairs at 24 terms (3x the labeled length): 84.7% vs baseline 1.9%, fillers 4.8%, zeros
12.7%, sigmoid 0.9%, oracle 92.2% (5 fresh seeds). All three registered criteria passed
(+81 to +85 on every seed; +79.8 over fillers; 91.6% of the oracle's gain). Both registered
limits held (48 terms: 7.6%; weaker than the oracle on large sums). Full write-up:
`research/final/REPORT.md`.

## Audit after the final (predictions written before running)

| # | check | why | prediction |
|---|---|---|---|
| 20 | NoPE (no positional encoding), sum testbed | known length-generalization method not tested before | unknown; npia's claim to beat known methods requires NoPE to stay well below npia |
| 21 | adjacent inverse pairs, sum testbed | if locally skippable pairs work as well, the gain is not about having to sum the inserted terms | weaker than spread pairs |
| 22 | S3 baseline and inverse pairs at 16,000 steps (4x) | the S3 conclusion came from an under-trained setup (baseline in-distribution 89%) | pairs still plateau near the parity coset (about 1/3) at 16 elements |

### Audit results so far (sum testbed, discovery seeds 0-1)

* **#20 NoPE:** 8 terms 5.0% (6.1, 3.8), below the baseline's 6.6%. It also lowers 4-term
  accuracy (92-95%). A known method that does not threaten inverse pairs (90.3%).
* **#21 adjacent inverse pairs:** 8 terms 86.2% (85.9, 86.5) vs spread pairs 90.3%. Only about 4
  points weaker. The prediction "weaker" holds only in that small sense: most of the effect
  survives when each pair could be skipped locally. So the explanation "the model is forced to
  sum the scattered inserted terms" is not supported. What the evidence supports is that
  non-zero cancelling terms are present (zeros and fillers, at the same or shorter lengths, fail).
* **#22 S3 at 4x budget (16,000 steps).** The first launch failed: in `cd dir && (A) & (B) &` the
  `cd` applies only to lane A. Rerun with absolute paths.
  * The baseline is now adequately trained (in-distribution 1-6: 98%), but long words stay at
    chance (16 elements: 15.0%).
  * Inverse pairs: 16 elements 34.2%. Parity is tracked at 97-100% on 12-16 elements, while exact
    accuracy given the right parity is 0.32-0.34, so rotation is still at chance.
  * In-distribution, pairs hurt the order-dependent part: 6 elements 73-74% vs 86-89% for the
    baseline.
  * The S3 conclusion holds under adequate training: the trick teaches the commutative part and
    costs accuracy on the order-dependent part.

### Audit verdict

* No bug in the final pipeline: the numbers recompute exactly, the code hash is verified, and all
  runs exited 0.
* Errors were in my reporting and are corrected: the idea count, the phrase "3x the training
  length", the SPEC wording (erratum), and the unmentioned 1-point in-distribution cost.
* Claims narrowed:
  * The filler control was length-mismatched; zeros is the matched control.
  * The "forced to sum scattered terms" explanation is dropped (adjacent pairs work almost as
    well).
  * The S3 limit is confirmed at an adequate budget, with an in-distribution cost.
* Still untested: scratchpad references and a faithful looped transformer. The final models were
  not saved.

## Batch 3 (2026-11-07): written before running

Motivation: in the language-model test, chaining beat inverse pairs because the pairs' long
examples keep short answers (`research/llm/REPORT.md`). Self-improvement (Lee et al. 2025, arXiv
2502.01612) labels real longer problems, which fixes that, but it only extends to lengths a model
already gets mostly right. Inverse pairs give a large free jump in length. The two might combine.
A pseudo-label check by agreement across answer-preserving rewrites was not re-tried: it is
close to idea #1 (discarded), and prior work reports that training on such agreed labels can
lower accuracy (arXiv 2608.05670).

| # | idea | kind | hypothesis | prediction (before run) |
|---|---|---|---|---|
| 23 | npia, then self-improvement | candidate (combination) | starting from npia, self-generated labels at 9-16 terms are accurate enough to keep extending; from the baseline they are not | after 6 rounds (+2 terms each, 3 models, full-agreement vote, same compute for both starts): npia start at least 60% at 16 terms; baseline start at most 20%; npia alone is 7% at 16 (#8) |

Protocol for #23: discovery seed 0 only (model seeds 0-2), script `si_combo.py`
(committed with this entry, before the run); 4,000 initial steps, then 6 rounds of 2,000 unlabeled problems and 500
steps per model. If kept, it goes to the language-model test only through a registered spec.
| 24 | npia with pairs up to 16 terms (no self-improvement) | control for #25 | covering lengths 9-16 with pairs alone; sums stay those of 1-4 terms | at 16 terms between 25% and 60%; weak on large sums |
| 25 | self-improvement where each round's agreed problems are also lengthened with pairs (up to 16 terms) | candidate | #23 breaks at the edge of covered length (round 3: models 40-58% on 9-10 terms, kept labels 76% right). Lengthening the self-labeled problems keeps training coverage ahead of the labeling frontier, while their labels bring real long-problem sums | after 6 rounds: at least 50% at 16 terms and at least 10 points above #24; kept-label accuracy at least 90% in rounds 3-6 |

Notes written before #24 and #25 ran:
* #23 (npia start) round 3 had already degraded when these were added: accuracy at 10 terms fell
  from 35% to 22% after training on 76%-accurate labels. #24 and #25 were designed after seeing
  that, so they are follow-ups on discovery data and are counted as such.
* The baseline-start arm of #23 did not start on the first launch: `cd dir && (A) & (B) &` runs B
  outside `dir`, the same mistake as #22. It was relaunched unchanged.
* `si_combo.py` gained `--cap` and `--expand-pseudo-cap`; their defaults reproduce #23 exactly
  (checked: the cap-8 pair function matches #8's transform on 300 problems).

### Batch 3 results (seed 0, discovery; outputs in `results_batch3/`)

| # | at 8 / 12 / 16 terms | self-labels kept and right (rounds 1 to 6) | prediction | decision |
|---|---|---|---|---|
| 23, baseline start | 9.8 / 3.8 / 2.4% (start: 5.2 / 3.6 / 1.6%) | right 64%, then 17%, 5%, 5%, 4%, 3% | at most 20% at 16: held | the 3 models agree on the same wrong sums (consensus trap) |
| 23, npia start | 85.0 / 9.4 / 4.6% (start: 85.0 / 10.8 / 6.4%) | right 99.7%, 98.9%, 76%, 10%, 6%, 4.5% | at least 60% at 16: **missed** | **discarded**: collapses one step past the covered length |
| 24, npia with pairs up to 16 | 81.2 / 69.4 / 58.2% | (no self-improvement) | 25-60% at 16: held | control: the pure rule's reach follows the length its pairs cover |
| 25, self-improvement with pair-lengthened self-labels | 89.8 / 78.0 / 65.8% | right 99.7%, 99.3%, 95.9%, 88.5%, 83.5%, 77.7% | at least 68.2% at 16 and labels at least 90% in rounds 3-6: **missed** (65.8%; labels below 90% from round 4) | **discarded** under its own bar; +7.6 points over #24 at 16 terms, on one seed, at about 6x the compute |

Reading:
* Direct self-improvement failed from both starts. Full agreement among 3 models did not stop
  shared errors, the trap noted for idea #1 and in prior work.
* The pure pair rule, with pairs covering 16 terms, reached 58.2% at 16 terms from labels on 1-4
  terms. That is the result batch 3 adds, and it needs no self-improvement.
* Ideas tried so far: 25 (counting #23-#25 as three).

## Idea #26: dropped before running, on prior art (2026-11-07)

* **The idea:** align the model's internal representation of a pair-expanded problem with that of
  its short original. Both end in the same full state, and in the three-person task the max label
  does not say who has what. So aligned states would carry more than the label does.
* **Prior art found before any code was written:**
  * PAR (arXiv 2610.27628, 2026-10-20) trains a looped Qwen2.5-0.5B with prefix
    self-distillation: the model's answers on prefixes, which are valid shorter problems, supervise
    the matching iterations on the full input. It uses final answers only.
  * Without step labels, PAR reports 85% at 16x the training length on single-query S5 (chance
    20%), and up to 99.7% on swap-S5 after a positional-stretch fine-tune. This project has shown
    about 3-4x.
  * Du et al. (ICML 2025, "Long-short alignment") regularize the gap between output distributions
    at different lengths.
  * Aligning representations across our rewrites would be a small variation on these.
* **Related, seen only in search results:** "Tracking States or Tracking Cosets?" (arXiv
  2609.29951) is on what models learn when tracking group products. The batch 2 plateau near the
  parity coset in S3 is that kind of outcome.
* **Decision:** not run. Ideas tried so far: still 25.

## Audit #27: S3 oracle (written before running, 2026-11-07)

* **Why:** "Tracking States or Tracking Cosets?" (arXiv 2609.29951, read 2026-11-07) shows that
  Transformers trained on long labeled words still plateau at 1/|[G,G]| exact accuracy beyond
  their exact frontier: they track the abelianization (for S3, parity) and guess within its class
  (1/3). Their Theorem 1: an order-blind model can output exactly the functions of the product that
  are constant on [G,G]-cosets. So the S3 plateau of inverse pairs (#15, #22: about 1/3 exact,
  parity right) is a known Transformer behaviour, not something specific to pairs. This project
  never ran an S3 oracle, so "pairs do not carry over to order-dependent composition" was never
  compared with what true long labels give on this bench.
* **Run:** `run2.py s3 oracle`: true labels on words of length 1-16, otherwise as #22
  (16,000 steps, seeds 0 and 1). Compared with #22 (same steps, seeds): baseline 15.0% and pairs
  34.2% at 16 elements.
* **Prediction (about 60% confidence):** the oracle at 16 elements is between 30% and 50%, so
  within about 15 points of pairs: at this scale the limit at 16 is the architecture, not the
  data. At 8 elements the oracle is at least 15 points above pairs.
* **What each outcome means:**
  * oracle at most 50% at 16: the S3 claim is restated. At the longest lengths, true labels also
    give about the abelianization, so pairs lose little to real long data there.
  * oracle at least 70% at 16: pairs leave a large gap on S3, and the claim stands as written.
* This is an audit check, not an idea. Ideas tried so far: still 25.

### Audit #27 result, part 1: the oracle (seeds 0-1, 16,000 steps; `results_audit27/`)

| elements | 6 | 8 | 12 | 16 |
|---|---|---|---|---|
| oracle exact (seed 0, seed 1) | 62.2, 65.1% | 38.1, 39.1% | 35.4, 37.1% | 30.5, 31.9% |
| oracle parity right | 99.9, 99.9% | 100.0, 99.9% | 99.9, 100.0% | 99.0, 99.7% |

* **At 16 elements** the oracle has 31.2% (mean); in #22, pairs had 34.2% and the baseline 15.0%.
  * The first part of the prediction held: 31.2% is within 30-50%, and 3 points below pairs.
  * True long labels do not beat pairs here.
* **From 8 elements on**, exact accuracy given the right parity is 0.31-0.39, about 1/3. The oracle
  tracks parity and guesses within it: the same plateau as pairs, and as the Transformers of
  2609.29951.
* **In range**, training on lengths up to 16 costs short-length accuracy. At 6 elements the oracle
  has 62-65%, against the baseline's 86-89% (trained on 1-6) and pairs' 73-74%.
* **Pending:** the second part (the oracle at least 15 points above pairs at 8 elements). It waits
  for a deterministic rerun of #22's pairs runs, whose per-length numbers were not saved.
* The parity numbers come from the saved parameters (`oracle_s*_s*.pkl`), evaluated on the same
  test words as `engine.evaluate`. The exact accuracies recompute to the saved values.

### Audit #27 result, part 2: pairs rerun, and verdict

| elements | 6 | 8 | 12 | 16 |
|---|---|---|---|---|
| pairs (#22 rerun; seed 0, seed 1) | 72.7, 72.3% | 45.2, 44.9% | 37.1, 32.1% | 35.2, 33.2% |
| oracle (mean of seeds 0-1) | 63.7% | 38.6% | 36.2% | 31.2% |
| oracle minus pairs | -8.8 | -6.5 | +1.7 | -3.0 |

* **The rerun reproduces #22 at 16 elements** (34.2%). At 6 elements it gives 72.3-72.7%, where
  #22's entry says 73-74%. The original per-length files were not saved, so that small
  difference cannot be traced.
* **Prediction, part 2: missed.** At 8 elements the oracle is 6.5 points below pairs, not 15
  above.
* **In-distribution (1-6):** pairs 93.8-94.1%, oracle 89.7-92.0%; the baseline had 98% in #22.
* **Verdict:**
  * At this scale on S3, true labels on long words give no more than pairs at any tested length.
  * Both track parity and stay at about 1/3 within it from about 10 elements on.
  * So the S3 failure is not a shortfall of the pair rule against long data. No data tested here
    teaches the order-dependent part.
  * The claim "pairs do not carry over to order-dependent composition" is replaced: on S3 at this
    scale, neither pairs nor true long labels teach it, and pairs match true long labels.
  * The in-distribution cost at 6 elements is shared by the oracle. It comes from training on
    longer words, not from pairs specifically.
* **Untested:** whether pairs would fall short in a model that does learn order-dependent tracking
  from long data, such as a recurrent or looped model (PAR; the LSTMs and GRUs of 2609.29951).
* `research/final/REPORT.md` is corrected to match. Ideas tried so far: still 25.

## Batch 4 (2026-11-08): splitting real changes, written before running

**Motivation.** The diagnosis of the language-model three-person final
(`research/llm/results/multi_final/diagnosis.txt`, post-hoc) found where the method fails.

* It fails at adding up many real changes for one person, not at comparing people. Only 4% of its
  wrong answers name another person's amount.
* Its accuracy falls from 80% to 12% as the winner's events go from 0-2 to 7 or more. The
  oracle's falls from 94% to 67%.
* Cancelling pairs add zero. So in the method's long training problems each person still has at
  most 4 real changes.

**Idea #28, split.** Rewrite a real change as several with the same total ("gains 7" becomes
"gains 3" and "gains 4").

* It uses another relation of the same event algebra, composition, where pairs use inverses.
* It needs no solver and reads no amount: a gain's second piece moves earlier, and a loss's later.
* The answer is unchanged, but the model must now add up many real changes per person.

**Prior art.** Answer-preserving rewrites are a known family: h1's chaining, and math-word-problem
augmentation (for example arXiv 2010.01556, reverse operations). Two searches found nothing that
splits changes to teach accumulation over long problems. So this is a new rewrite in a known
family.

**Testbed 3.** `task_multi.py`, the three-person task for the small model.

* Rules as in `multi.py`, with answers capped at 99.
* Training on 1-4 events. The test is 1,000 fresh problems at each of 1-4, 8, 12 and 16 events,
  and the primary metric is 12 events.
* Model and budget as in the protocol.
* Discovery seeds 0 and 1; runner `run3.py`; tests in `test_task_multi.py`.

| # | condition | kind | prediction (before run) |
|---|---|---|---|
| 30 | baseline (1-4 events) | baseline | at most 15% at 12 events |
| 31 | pairs (`multi.cancel_spread`, p = 0.5, up to 16) | reference: the language-model method | 40-80% at 12 events |
| 32 | oracle (true labels, 1-16) | ceiling | at least 80% at 12 events |
| 28 | split (p = 0.5, up to 16 events) | candidate | at least 10 points above pairs at 12 events (about 35% confidence) |
| 29 | split, then pairs (both relations) | candidate | at least 10 points above pairs at 12 events (about 40% confidence) |

**Proxy check, first.** Does the toy show the language model's failure?

* It holds if pairs fall by at least 30 points at 12 events between problems where the winner had
  0-2 events and 5-6 events, while the oracle falls less.
* About 55% confidence.
* If it fails, the toy cannot test what split is for, and the split result says little about the
  language model.

**Keep rule (as in the protocol, against the best reference):**

* the mean at 12 events at least 10 points above pairs, with both seeds above pairs;
* in-distribution (1-4) at least 95%, or no more than 3 points below the baseline if the baseline
  itself is below 95%.

A kept idea goes to the language model only through a registered spec. Ideas tried so far: 27
after this batch (#28-#29).

### Batch 4 result (2026-11-08): the testbed failed, so split is untested

**Proxy check: failed.** The toy did not reproduce the language model's failure, so this batch
cannot judge split.

* Pairs fell by 3.8 points (seed 0) and 6.6 points (seed 1) between winners with 0-2 events and
  winners with 5-6 events. The check needed at least 30.
* The cause is the toy, not any condition. The small model never learned the task, even at training lengths:
  the baseline gets 82.5% at 1 event and 29.9% at 4 events (mean of seeds 0 and 1).
* At 12 events every condition is between 5% and 8%. Always giving the most common answer scores
  4.5% (seed 0) and 5.6% (seed 1) on the same test problems, so these numbers are at the floor.

| # | condition | 1 event | 4 events | mean, 1-4 | 8 | 12 (s0, s1) | 16 | prediction |
|---|---|---|---|---|---|---|---|---|
| 30 | baseline | 82.5 | 29.9 | 53.6 | 8.2 | 5.3 (5.3, 5.4) | 3.3 | at most 15%: held |
| 31 | pairs | 70.7 | 23.0 | 44.1 | 7.1 | 5.7 (5.0, 6.4) | 5.2 | 40-80%: missed |
| 32 | oracle (1-16) | 53.5 | 15.0 | 30.8 | 6.1 | 7.1 (6.1, 8.0) | 7.2 | at least 80%: missed |
| 28 | split | 78.7 | 28.5 | 50.4 | 11.2 | 6.3 (7.4, 5.3) | 5.6 | +10 over pairs: missed (+0.6) |
| 29 | split, then pairs | 69.8 | 23.5 | 44.0 | 7.5 | 5.7 (5.4, 6.0) | 5.8 | +10 over pairs: missed (0.0) |

All numbers are percent correct on 1,000 fresh problems per length and seed. The per-seed results
and the accuracy by the winner's events are in `results_batch4/*.json`.

**Keep rule.** Nothing is kept. No condition reaches the in-distribution bar; the baseline itself
averages 53.6% at 1-4 events. Split (#28) and split-then-pairs (#29) are neither kept nor deleted:
this batch says nothing about them.

**My errors.**

* I launched the whole batch (about 1.4 hours on each of two CPU lanes) without first checking that the
  baseline learns the task at training lengths. One baseline run (8 minutes) would have shown it.
* My predictions for pairs and the oracle assumed the toy could learn the task. Nothing I had
  measured supported that.

From now on, each new testbed starts with a baseline run at training lengths. A batch goes ahead
only if the baseline reaches at least 90% there.

Ideas tried so far: 27 (#28 and #29 were run, but this testbed could not test them).

## Batch 5 (2026-11-08): certified rewrite mining, written before running

**Why a different kind of idea.** Ideas #7-#29 are rewrites that I designed by hand: zeros,
inverse pairs, splits. Each new task needs a person who knows its algebra. Self-improvement
avoids the person, but it lets the model label its own long problems, and that failed every time
here (the consensus trap: #1, #23, and the language-model final). Rewrite mining keeps the
model's judgment and moves it to the place where the model is reliable. It **certifies rewrites,
not labels.**

**Idea #33, certified rewrite mining.**

1. Train the model on labeled short problems, as the baseline does (this model is M0).
2. Take a generic space of edits that never read the answer. Here: insert one segment (a term or
   group element); insert two segments at independent random positions; insert two segments as an
   adjacent block. The segment vocabulary is all terms (sums: +0 to +9 and -0 to -9) or all group
   elements (S3).
3. For each edit, apply it to labeled problems of 1-2 segments, so the edited problems stay
   inside the training lengths where M0 is reliable. Score the edit by how often M0's answer on
   the edited problem equals the original problem's true label.
4. Certify the edit if that score is at least M0's own accuracy on unedited problems of the same
   lengths, minus 5 points. A rewrite that keeps the answer should score like M0's accuracy. An
   edit that changes the answer cannot, unless M0 ignores it.
5. Train a fresh model with the certified edits applied to labeled short problems up to 8 terms
   (p = 0.5, as in npia). The labels are inherited. M0 never labels a long problem.

No solver is used anywhere. The true answers of edited problems are computed only afterwards,
to measure how accurate the certificates were.

**Prior art (checked 2026-11-08, seven searches, five papers read).**

* **Compositional self-improvement** (arXiv 2611.00452, 2026-10-30) builds supervision for
  harder problems from a model's answers on simpler parts. People supply the composition rules.
  Its conclusion leaves "how to discover and validate such structures" open. That is the gap this
  idea targets.
* **Metamorphic-relation inference** in software testing (GenMorph, arXiv 2312.15302; survey
  arXiv 2406.05397) finds input-output relations by running a program. The relations test
  programs; they do not train models.
* **GraphAug** (arXiv 2202.13248) trains a separate model to predict label invariance and
  augments at the training distribution. It does not use the edits to reach harder problems.
* **HARDEN** (arXiv 2609.30571) makes harder evaluation cases and checks each one with a
  language-model judge. It produces evaluation cases, not training data.
* **CEDC** (arXiv 2512.01187) needs an external verifier and oracle labels for long problems.
* **Verdict.** I found no work that certifies edit types with the learner's own short-length
  answers against true short labels, and then uses them to build long training data. So this is a
  new algorithm in known families (learned augmentation, compositional self-improvement). Search
  can miss work, so this is not proof of novelty.

**Gate (the rule from batch 4).** The M0 baseline must reach at least 90% on 1-4 terms (sums).
If it does not, the batch stops and nothing else is judged.

| # | condition | what it tests | prediction (before run) |
|---|---|---|---|
| 34 | baseline (sums, 1-4 terms); also M0 | reference; gate | 1-4 terms at least 95%; 8 terms 5-15% |
| 35 | npia rerun (same runner) | the hand-designed rule | 8 terms within 3 points of batch 1's 90.3% |
| 33 | mined (sums) | does mining recover npia without a person? | certified set: precision and recall at least 95% against the exact invariants; 8 terms within 5 points of npia on both seeds (about 70% confidence) |
| 36 | certification only (S3, M0 trained on 1-6 elements) | does mining find the right algebra when order matters? | adjacent inverse pairs and the identity certified; spread pairs of non-identity elements rejected; precision and recall at least 90% (about 75% confidence) |

**How to read it.** On sums the best possible outcome is a tie with npia. Mining finds rules; it
does not make better rules than an expert who knows the algebra. The claim to test is that the
person can be removed. In S3, inserting "g ... g^-1" at spread positions does not keep the answer.
A rule copied from the abelian case would be wrong there, and the miner should notice.

**Keep rule.** #33 is kept if its 8-term accuracy is within 5 points of npia on both seeds,
in-distribution stays at least 95%, and certification precision is at least 95%. If kept, the
next test is a task whose algebra is hidden, and then the language model through a registered
spec.

Discovery seeds 0 and 1. Runner: `run4.py`; tests: `test_mine.py`. Compute: mining costs M0's
training plus about 0.1M forward passes for certification, then one normal training run. That is
about 2x npia's compute, and I will report it.

Checks written before the batch ran (`test_mine.py`, and a 300-step smoke run):

* With a perfect predictor, certification accepts exactly the edits that keep every answer: 46 of
  820 for sums and 8 of 78 for S3 (the identity, the six adjacent inverse pairs, and two spread
  identities).
* A predictor blind to the last segment certifies wrong edits (precision 13-16%). If M0 had such a
  blind spot, the certificates would be wrong.
* In the smoke run an undertrained S3 model (54% on 3-4 elements) certified with precision 45%.
  The certificates are only as good as M0's accuracy at the edited lengths. That is why sums have
  the gate. For S3 I report M0's accuracy next to the certificates.
