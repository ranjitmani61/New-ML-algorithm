# Next direction: metamorphic self-training (proposal, not yet tested)

## Why change direction

* Zero-token answer selection is a dead end for us (see `reasoning/AUDIT.md`). The room to win
  is smaller than the room to lose, and gains are about 1 point, which is at the noise floor.
* Ideas inside recurrent "deep thinking" networks are saturated as of 2026-11. Attractor
  training, fixed-point halting, energy-guided selection, test-time search, quantized latent
  states and verifier feedback all have recent papers (for example 2605.21488, 2606.18206,
  2607.10128, 2610.18547, 2607.16237, 2605.08605, 2608.24136).
* Label-free reinforcement learning for reasoning (TTRL, 2504.16084, and dozens of follow-ups)
  rewards agreement with the model's own majority vote. Its documented failure is the
  "consensus trap": confident errors get reinforced (2603.17775, 2605.19444, 2610.37553,
  2604.21327). This failure is open and important.

## The idea

Train a reasoner on unlabeled hard inputs using **equivariant metamorphic relations**: known
ways that transforming an input changes the correct output. Two examples: appending "+ c" to an
expression adds c to its value; scaling every price in a word problem by 10 scales a price answer
by 10. The reward or target is consistency between the model's outputs on an input and on its
transforms. It is not agreement with its own majority vote.

**Core claim (falsifiable).** Write the model's error as e(x) = f(x) - f*(x).

* A majority-vote reward is satisfied by any confident error.
* An invariance relation f(Tx) = f(x) is satisfied by any error with e(Tx) = e(x).
* An equivariant relation f(Tx) = S(f(x)) is satisfied only by errors that themselves transform
  exactly like the answer, e(Tx) = S(e(x)).

So a family of equivariant relations, anchored on easy labeled inputs, rules out most consistent
errors. That is the analogue of a PDE plus boundary conditions in physics-informed training. It
predicts that equivariant self-training corrects systematic errors that majority-vote
self-training entrenches.

**Prior art checked (alphaXiv, 7 searches).** Label-free RLVR uses majority vote, verifiers,
multiple rewards or co-evolving verifiers. Invariance consistency exists (rephrasings, other
languages), and 2608.05675 notes its blind spot. Self-improvement through easier-to-harder
self-labelling: 2502.01612. Zero-padding and position-offset augmentation for arithmetic is known
and is a special case of an exact-label relation. Not found: equivariant cross-input relations as
the training signal against the consensus trap. Not finding it is not proof that it is new.

## First experiment (CPU, JAX, no GPU)

* **Task:** evaluate integer arithmetic expressions with a small transformer. Labeled training
  uses up to 4 terms; unlabeled hard inputs use 5-8 terms; the test is fresh 8-term expressions.
* **Label-free phase, same unlabeled inputs and same compute for every condition:**
  1. majority-vote self-training (TTRL-style baseline);
  2. invariance consistency (term reordering);
  3. equivariant metamorphic consistency (append or prepend a constant, scale);
  4. placebo (a wrong relation with the same loss weight);
  5. oracle labels (upper bound).

## Gates and kill criteria (set before any result)

* **G1 headroom:** after labeled training, 8-term accuracy is below 50%, the oracle reaches above
  90%, and majority self-training stalls clearly below the oracle. If majority self-training
  already reaches the oracle, there is no consensus trap here: change the task or stop.
* **G2 placebo:** the wrong relation must not help. If it does, the gain comes from touching long
  inputs, not from the relation.
* **G3 split:** develop on seeds 0-2; confirm on seeds 3-5 unchanged; the sealed set is fresh
  generator seeds evaluated once.
* **G4 effect:** at least +15 points over majority self-training on 8-term inputs in the
  confirmation seeds. Otherwise stop.
* **G5 mechanism:** the gain must come mostly from errors that are consistent across samples, the
  ones majority voting cannot fix. This is measured, not assumed.

## If it passes

Move to language models: word problems where relations come from dimensional scaling (prices,
units). Use the existing vLLM pipeline for signal tests on GPU, and JAX on the Kaggle TPU hours for
training. That step needs its own prior-art check and gates.

## Honest odds

Uncertain, perhaps one in three that G1-G4 pass. Main risks: errors at length may be random rather
than consistent (then majority voting already works); relations may be satisfiable by wrong
solutions the model prefers; the closest prior (self-improvement) may already capture the gain.

## Pilot results (development seed 0, 2026-11-02)

Setup: signed sums of single-digit terms; a 3-layer, width-64 rotary transformer trained on 1-4
terms for 12,000 steps (`research/mst/`).

**G1 headroom: passed.** Greedy accuracy 99-100% at 1-4 terms, 23% at 5 terms, 6-9% at 6-8.

**Consensus trap: present and extreme.** 95-98% of the model's errors are repeated by the majority
of 8 samples, so majority self-training would train on more than 90% wrong labels at 6-8 terms.

**The relations do not break it (pseudo-label quality before any training, 1,000 expressions per
length).** Precision is the share of kept labels that are correct.

| terms | majority | permutation invariance | one-term shift | all-term shift (idea #2) |
|---|---|---|---|---|
| 5 | 23% | 23% | 26% | 31% |
| 6 | 8% | 9% | 7% | 6% |
| 8 | 6-9% | 7% | 8% | 11% |

The relations cut the number of wrong labels (all-term shift at 8 terms: 0.29 vs 0.90 wrong labels
per expression) but do not add correct ones. Error structure: at 6-8 terms the sign is right and
the value is off by a few units, and the error survives reordering and changing every term's
value. It depends on structure (length, sign pattern), not on the term values, and none of these
relations change that.

**Lesson:** a relation can only expose errors whose cause it changes. For difficulty-driven
(length) errors the cause is difficulty, so only difficulty-changing relations can help. Those
reduce to bootstrapping from easier instances, which is existing work (2502.01612).

Ideas counted in this direction so far: 2 (one-term shift, all-term shift).

## Self-training check (seed 0, 3 rounds of 2,000 unlabeled 5-8-term inputs, 600 steps each)

| condition | 5 terms | 6 terms | 8 terms |
|---|---|---|---|
| none | 23% | 6% | 7% |
| majority vote | 33% | 7% | 8% |
| metamorphic (all-term shift) | 56% | 5% | 6% |
| oracle labels | 96% | 93% | 84% |

**Decision: discarded.** It fails the preregistered 8-term target. The one-seed gain at 5 terms
(+23 over majority) is noted but not claimed. The work continues as an idea tournament in
`research/tournament/` (ledger: `research/tournament/LEDGER.md`).
