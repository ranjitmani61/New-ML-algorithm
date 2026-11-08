# Registered test: can the language model find the cancelling-pair rule by itself?

Registered on 2026-11-08, before any language-model run of mining. Results are reported
against this file unchanged, whatever the outcome. Any deviation goes into an erratum here.

**Status of the method going in.** Certified rewrite mining (`research/tournament/LEDGER.md`,
#33) **failed** its fresh-seed confirmation by the letter of criterion 2: on seed 4 it scored
5.4 points above the hand-designed rule, outside the registered band of 5 points either way.
Its certificates were exact on all six fresh runs, and its mean equalled the hand rule's (88.0%
vs 87.8% at 8 terms). This test was planned to follow a passed confirmation. It runs now
because the user decided so, knowing the failure. The method stays recorded as not confirmed
in the tournament.

## Question

In the three-person test (`SPEC_MULTI.md`), I designed the rule by hand: cancelling pairs
(`multi.cancel_spread`). It beat self-improvement, which let the model label its own long
problems. Can the model find such a rule itself? Here it certifies edits with its own answers on
short problems, where it is reliable. Then it trains on short problems lengthened by the
certified edits, and these keep their labels. No person writes a rule, no solver is used, and the
model never labels a long problem.

## Method (`mine_multi.py`)

* **M0** is the final's plain model for each seed. These are the LoRA adapters from kernel
  `llm-multi-final-b`, mounted read-only (sha256 of `adapter_model.safetensors`):
  * seed 9: d044cf8f9ced89087772bf7a794e7b7c4ba25dd96be2b21acf91607fc4225d0f
  * seed 10: d1cce665eabdb6ddc4ae7d2cd3a4686592ad70318d58a56b0435003d6c375e5d
  * seed 11: b4a02a07690e768c7f07c40914937832a9dba05d57ee971948320ef45b487f1a
* **Edits** never read an answer.
  * A segment is an event: a person (0-2), a gain or a loss, and an amount of 1-9. That gives 54
    segments.
  * There are three kinds of edit: insert one segment; insert two at independent random
    positions; insert two as an adjacent block. In all, 5,886 edits.
  * An inserted event is written with `multi.py`'s templates and the problem's own people and
    item.
* **Certification.** Each edit is applied to new problems with 1-2 events, so the edited problems
  have 2-4 events. The score is how often M0's greedy answer equals the original problem's true
  label.
  * Stage 1: 8 problems. An edit with more than 1 mismatch is rejected.
  * Stage 2: up to 64 problems in total.
  * An edit is certified if its score is at least M0's accuracy on unedited problems of the same
    lengths minus 5 points. That accuracy is measured on 1,000 problems per edit size.
* **Gate (per seed).** M0 must reach at least 90% on unedited problems with 3-4 events. If not,
  that seed stops before certification.
* **Training.** This is run_multi.py's plain recipe, unchanged:
  * 6,000 steps, batch 16, learning rate 2e-4, LoRA rank 16, 24,000 problems with 1-4 events;
  * half of the problems are lengthened by random certified edits up to 16 events, keeping their
    short labels. That matches cancel_spread's p = 0.5 and cap of 16.
* **Evaluation.** The final's sets, rebuilt from the same seeds: 500 problems at each of 4, 8, 12
  and 16 events, plus 500 at 12 events with held-out vocabulary. Greedy decoding with the strict
  parse. Within a seed, the test problems are identical to those of the final's conditions.
* **Settings chosen before this run, and how.**
  * The 64-problem stage 2 replaced the toy runs' 300-problem single stage, to save compute.
  * Its size came from a binomial power calculation, together with a measurement that used the
    generator's true answers only: no model and no test problem. In that measurement, some
    non-exact edits keep the answer on 1-2-event problems up to about 80% of the time.
  * With 32 problems, about one wrong edit would be expected to pass. With 64, about 0.1 would
    pass, and about 89% of the exact edits would pass.

## Code (sha256)

* `mine_multi.py`: ea2bd63f5f6005d27e4ee7a31ff4a32992a58c18d04ce97f67ddf4f76d5d9414
* `multi.py`, `run_multi.py`, `run_llm.py`, `problems.py`: unchanged from the final (hashes in
  `SPEC_MULTI.md`; checked equal).
* Runner: `kaggle_llm.py` (adds `mine_multi.py` to the kernel and mounts the final's output).
  Job: `jobs/mine_multi.json`, kernel `llm-mine-multi`, T4 x2. Lane 0 runs seeds 9 and 11; lane 1
  runs seed 10.

## Decision rule

The primary metric is accuracy at 12 events (training vocabulary), per seed. The comparison
conditions are the final's own runs on the same seeds and test problems (`results/multi_final/`).

1. **Gate.** At least two seeds must pass M0's gate, or the result is NO TEST.
2. **Validity.** Mined minus plain must be at least 15 points on every seed that passed the gate.
   This is the bar cancel_spread had to clear. If it fails: MINING FAILS HERE.
3. Otherwise, with d = mined - cancel_spread:
   * **MINING BETTER:** mean d at least +5 points, and d > 0 on every seed;
   * **HAND RULE BETTER:** mean d at most -5 points, and d < 0 on every seed;
   * **otherwise: MATCHES** (no clear difference from the hand-designed rule).
4. The claim "the model found the rule itself and trained as well as with the hand-designed rule"
   is supported by MATCHES or MINING BETTER. Under HAND RULE BETTER or MINING FAILS HERE, it is
   not.

## Secondary (reported, not decisive)

* Certification: precision and recall against the exact edits, that is same person, opposite
  kind and same amount, at spread or adjacent positions (108 edits). Also the edits certified
  wrongly, with their true keep-rate.
* Training data: the share of inherited labels that are true (from logged true answers, never
  used), and the share of problems that are valid (no amount below zero). An inserted loss
  placed before its gain can make an amount negative for a while.
* Mined minus self-improvement at 12 events. Accuracy at 4, 8 and 16 events and on held-out
  vocabulary. GPU seconds for certification and for training.

## Predictions (recorded before the run)

1. The gate passes on all three seeds (about 80%). Plain gets 91.8-92.2% at 4 events.
2. Certification precision is at least 95% on every seed (about 70%), and recall is at least 80%
   (about 65%).
3. Verdict: MATCHES about 50%, HAND RULE BETTER about 25%, MINING FAILS HERE about 15%, MINING
   BETTER about 10%.
4. Mined at 12 events, mean over seeds: 33-48% (cancel_spread 41.7%, plain 23.3%).
5. At least 99% of the training labels are true (about 75%). Invalid training problems are 5-25%.

## Known weaknesses, stated before the run

* **These are not fresh seeds.** The comparison conditions' results on seeds 9-11 are known,
  and their 12-event test problems were analysed after the final (`diagnose_multi.py`). The
  mining design was fixed on the toy tasks. Its only new settings are the stage sizes, chosen as
  described above. A fresh-seed replication with plain, cancel_spread and mined on new seeds is
  needed for a full claim. It is planned after the GPU quota refreshes on 2026-11-14.
* **Compute.** Mined costs M0's training plus certification plus a full training run, about
  twice cancel_spread's GPU time.
* **The edit space is generic but chosen by me.** Insertions of one or two events are the
  segments of this task. Mining can find only rules that the space contains.

## Budget and stop rule

* About 2.8 GPU hours of the about 4.6 left this week: one kernel with two lanes, each seed about
  1.4 hours (certification about 15 minutes, training about 68).
* If the kernel fails, I fix it and rerun once this week only if enough quota is left for the
  full job. Otherwise it waits for the refresh.

## Result (2026-11-08, written after the run; nothing above was changed)

Kernel `llm-mine-multi` completed with all three seeds. The kernel recorded `mine_multi.py`'s
sha256 (ea2bd63f…, equal to the registered one) and the three M0 adapter hashes, which are equal
to the registered ones. There is no exact overlap between training and test problems. Outputs are
in `results/mine_multi/`, without adapters. `summary.txt` is the output of `summarize_mine.py`,
which was committed before the results.

**VERDICT: MATCHES.**
* At 12 events, mined got 41.2%, 45.2% and 42.4% (mean 42.9%). The hand rule (cancel_spread)
  got 40.4%, 43.2% and 41.4% (mean 41.7%).
* So d = +0.8, +2.0 and +1.0, mean +1.27 points.
* Validity holds: mined minus plain is +19.6, +20.4 and +19.0.
* By rule 4, the claim "the model found the rule itself and trained as well as with the
  hand-designed rule" is supported here. This holds on seeds that are not fresh (see weaknesses).

| seed | M0 at 3-4 events | edits certified | of them exact | exact edits found | mined at 12 | cancel_spread at 12 | plain at 12 | self_improve at 12 |
|---|---|---|---|---|---|---|---|---|
| 9  | 93.8% | 64 | 64 | 64 / 108 | 41.2% | 40.4% | 21.6% | 24.8% |
| 10 | 94.3% | 66 | 66 | 66 / 108 | 45.2% | 43.2% | 24.8% | 24.8% |
| 11 | 91.9% | 86 | 86 | 86 / 108 | 42.4% | 41.4% | 23.4% | 22.8% |

### Predictions, scored

1. The gate passes on all three seeds: **right**.
2. Precision at least 95%: **right**. It was 100%, and no wrong edit was certified on any seed.
   Recall at least 80%: **wrong**. It was 59.3%, 61.1% and 79.6%.
3. MATCHES was the most likely verdict (about 50%): **right**.
4. Mined at 12 events, mean 33-48%: **right**, 42.9%.
5. At least 99% of the training labels are true: **right**, 100%. Invalid training problems
   5-25%: **wrong**. They were 4.6%, 4.6% and 4.1%, below the range.

### Secondary (reported, not decisive)

Means over seeds 9-11, in %:

| condition | 4 events | 8 | 12 | 16 | held-out vocabulary, 12 |
|---|---|---|---|---|---|
| plain | 91.9 | 56.4 | 23.3 | 11.7 | 23.2 |
| cancel_spread (hand rule) | 92.2 | 69.3 | 41.7 | 25.8 | 42.1 |
| mined | 90.9 | 65.2 | 42.9 | 31.3 | 39.9 |
| self_improve | 85.1 | 52.9 | 24.1 | 10.7 | 23.4 |
| oracle (true labels) | 97.7 | 93.9 | 88.5 | 79.9 | 86.5 |

* Mined was below the hand rule at 8 events on every seed (-6.0, -2.0, -4.2) and with held-out
  vocabulary on every seed (-3.6, -1.4, -1.6).
* Mined was above it at 16 events on every seed (+7.6, +2.4, +6.4).
* These were not tested in advance, and I don't know their cause.
* Mined minus self_improve at 12 events: +16.4, +20.4 and +19.6.
* **GPU seconds per seed:**
  * certification: 1,050, 983 and 1,086;
  * mined training: 4,978, 4,788 and 5,025;
  * cancel_spread training (final): 4,086-4,097;
  * plain training: 2,412-2,418.
* Mined's total, including training M0, is about 8,400 s, about 2.05 times cancel_spread.
* Mined's training data was no longer than cancel_spread's: 5.6 vs 5.7 events on average. So
  the extra training time does not come from longer problems. I did not find its cause.

### Post-hoc analysis (not registered; `recall_mine.py`, output in `results/mine_multi/recall.txt`)

* **Why recall was low.** M0 is less accurate on problems that contain a cancelling pair than on
  typical problems of the same length.
  * Exact edits that went to stage 2 scored 90.7%, 92.2% and 91.4%. M0's reference was 93.8%,
    94.3% and 91.9%. These means are biased upward, because only edits that passed stage 1 are
    included.
  * So with a 5-point margin, many exact edits fell below the threshold. Rejected at stage 1:
    16, 24 and 13; rejected at stage 2: 28, 18 and 9.
  * Most misses involve large amounts. Exact edits with amounts 7-9 were certified 13/36, 13/36
    and 23/36 times; amounts 1-3, 28/36, 27/36 and 30/36.
  * Seed 11 had the lowest reference, so the lowest threshold (86.9%) and the highest recall.
* **The precision margin was thin.** The best-scoring wrong edits reached:
  * 82.8% on seed 9 (threshold 88.8%);
  * 85.9% on seed 10 (threshold 89.3%);
  * 85.9% on seed 11 (threshold 86.9%), one point below the threshold.
  * These edits keep the answer about 83-86% of the time. For example, two losses for one person
    keep it whenever that person is not the richest.
  * With a slightly weaker M0, a fixed 5-point margin would have certified such edits. Their
    problems would then have carried wrong labels. This is a real weakness of the fixed margin.

### What this does not show

* **Not fresh seeds.** A fresh-seed replication with plain, cancel_spread and mined is still
  needed, after the quota refresh on 2026-11-14.
* **One synthetic task.** The rule was inside an edit space I chose.
* **Matching, not beating.** Mined equals the hand rule at about twice the compute. Its value, if
  it holds, is that no person writes the rule and the model never labels a long problem.
* **#33 is still not confirmed** under the tournament protocol, because its fresh-seed
  confirmation failed.

**Deviations:** none. The PyTorch warning about `lr_scheduler.step()` before `optimizer.step()`
also appears in the final's logs, from the same recipe.
