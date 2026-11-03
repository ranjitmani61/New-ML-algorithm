# Language-model test of cancelling-event insertion: plan and rules (draft, not yet registered)

## Verification log (2026-11-03, after a session restart on a new machine)

* The registered final code hash matches (`2aa08eb5...`). The committed results reproduce
  `research/final/results/report.txt` exactly (PASS, 91.6%).
* All three test suites pass: tournament, reasoning, recorder.
* The branch equals the remote, and the last 40 commits all carry this session's trailer.
* Kaggle: four kernels not made in this project (`disybo-gpu`, `disybo-gpu-benchmark`,
  `notebook77b19ea399`, `notebook1afaeb7ccf`) were confirmed by the user as theirs.
* The stuck TPU probe (`pairs-final-probe`, queued about 23 h) was deleted at the user's request.
* Quota: GPU 0.63 h and TPU 20 h, until 2026-11-07.

## Question

Does inserting relevant, net-zero event pairs into short labeled word problems let a small
instruct model solve problems with many more state-changing events? For example: "She found 4
shells. Later she lost 4 shells."

## Design (to be fixed in a registered spec before any full run)

* **Problems:** templated additive state changes (one person, one item, a start amount, then K
  events of gain or loss). The answer is the final amount, computed exactly.
  * Labeled training uses K of at most 4.
  * The test uses K = 8, 12 and 16.
  * A second family has order-dependent events ("doubles", "halves"), tested separately,
    because the toy results predict no gain there.
* **Conditions:**
  1. plain fine-tuning;
  2. irrelevant sentences (PROBLEMATHIC or NoOp style);
  3. inactive events ("she counted them"), with no change;
  4. cancelling event pairs, spread out (the method);
  5. cancelling event pairs, adjacent;
  6. true labels at all K (the ceiling).

  Conditions 2-5 are matched to condition 4 on token length.
* **Model:** a small instruct model (Qwen2.5-0.5B-Instruct class), LoRA fine-tuning, greedy
  decoding, at least 3 seeds.

## Rules for this phase

1. **Headroom gate first.** If the base model or plain fine-tuning is already near the ceiling at
   the primary K, stop. There is nothing to show.
2. **Exact labels.** Answers come from integer arithmetic in the generator. Unit tests check that
   every augmentation keeps the answer (or changes it exactly as declared), and that amounts never
   go negative.
3. **No leakage.** Training and test problems are checked for exact overlap by hash. One set of
   names and items is held out for a secondary out-of-distribution test.
4. **Length-matched controls.** Token counts per condition are measured and reported, not
   assumed.
5. **Strict grading.** One fixed rule extracts the final number. Parse failures are counted
   separately. At least 20 raw outputs per condition are read by eye.
6. **Keep everything.** Trained adapters, raw generations and per-example scores are saved, so
   metrics can be recomputed independently from the raw outputs.
7. **Registration.** The specification, code hash, seeds, metric and pass criteria are committed
   before the full run. Any deviation goes into an erratum, never a silent edit.
8. **Budget declared up front, with a stop rule.** No run outside the declared budget.

## Build log (2026-11-03)

* **Local tests**, with a tiny random Qwen2 model run under Kaggle's exact versions
  (transformers 5.20.2, PEFT 0.21.2):
  * the answer-only loss mask is checked token by token;
  * an overfit model reproduces 16/16 answers through batched generation;
  * all condition, family and zero-shot paths run, with zero leakage.
* **Bugs found and fixed before any GPU run:**
  * generate() and token_type_ids;
  * cancel_spread on the mixed family silently fell back to plain;
  * cancel_adjacent guessed the family;
  * the tiny tokenizer's spacing (a test fixture);
  * a wrapper that hid failures;
  * the jobs file lost to a blocked command.
* **Kaggle smoke v1:** setup fine, but LoRA crashed because PEFT 0.21.2 raises on Kaggle's torchao
  0.10.0. I reproduced this locally and added a LoRA preflight that removes the unused torchao
  and retries.
* **Kaggle smoke v2 (real Qwen2.5-0.5B-Instruct; smoke only, not a result):**
  * the preflight recovered as designed, both jobs exited 0, hashes matched local, there was no
    leakage, and the metrics recompute exactly;
  * base model: 5% / 1% / 2% at 4 / 8 / 16 events;
  * cancel_spread after 150 steps: 24% / 8% / 3%;
  * 0.54 s per training step, so about 15-16 minutes per full run;
  * prompts: median 68 tokens, max 168, none truncated.
* **Quota after the smoke runs:** GPU 0.55h h until 2026-11-07. The pilot, discovery and final
  runs wait for the refresh.
