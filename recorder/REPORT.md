# Experiment recorder

An append-only, hash-chained evidence journal for the research pipeline, in a single SQLite file using only the standard library. It records:
- runs, with interpreter, package versions, git commit and dirty flag, and SHA-256 hashes of the pipeline source files;
- metrics, content-addressed artifacts and candidate entities;
- per-stage decisions and preregistrations;
- every access to a sealed test set.

It also provides the cross-round **leaderboard** the pipeline was missing.

## Files

| File | Purpose |
|---|---|
| `recorder.py` | Recorder, verification, bundle export and CLI |
| `test_recorder.py` | 20 contract tests, including attacks |
| `demo_pipeline.py` | End-to-end pipeline run recorded by the recorder, plus a throughput benchmark |
| `pipeline_run.bundle.jsonl` | The recorded pipeline run (1,613 events, 7 artifacts), verifiable offline |
| `REPORT.md` | This file |

```bash
python -m unittest test_recorder -v
python demo_pipeline.py --csv heart.csv --target HeartDisease --journal run.sqlite --rounds 5 --per-round 100
python recorder.py verify run.sqlite --head SEQ:HASH
python recorder.py leaderboard run.sqlite dev_balanced_accuracy
python recorder.py verify-bundle pipeline_run.bundle.jsonl --head 1613:2c4c42576ccd58706af89955edc31b271d3f38b566aa95647f728da68a1efe3e
```

```python
import recorder as rc
rec = rc.Recorder("study.sqlite", sources=["generator_only/hypothesis_generator.py"], repo=".")
with rec.run("development round 1", "development", params={"steps": 160}) as run:
    run.entity("H-abc", "hypothesis", {"round": 1})
    run.metric("dev_balanced_accuracy", 0.87, split="development", entity="H-abc")
rec.leaderboard("dev_balanced_accuracy")          # best across all rounds
rec.preregister(["H-abc"], "best dev candidate goes to test", metric="dev_balanced_accuracy")
rec.holdout_access(test_fingerprint, purpose="final comparison")   # second call raises
rec.verify(expected_head=published_head)
```

## Guarantees and how each was tested

| Property | Mechanism | Evidence |
|---|---|---|
| Append-only | SQLite triggers abort UPDATE and DELETE on events and blobs | Test: both statements rejected |
| Tamper-evident | Each event hash covers sequence, time, run, kind, body hash and the previous hash | Tests: an edited body is caught; an edit re-hashed consistently still breaks the next link; a corrupted blob is caught on read and by `verify()` |
| Truncation and rewrite | `head()` can be published externally; `verify(expected_head=…)` checks the journal still contains it | Test: a clean truncation passes plain `verify()` (stated limit) but fails against the published head |
| Holdout single use | Second `holdout_access` for the same fingerprint raises `HoldoutReuseError` **and records the refusal**. Declared reuse needs a 20+ character justification and taints every later test metric | Tests; also a second access forged through the low-level append (chain kept valid) is caught by `verify()` |
| No selection on test | `leaderboard(split="test")` raises `SelectionOnTestError`. Test metrics carry `preregistered` and `holdout_tainted` flags, which `verify()` recomputes | Tests |
| Run lifecycle | Exceptions mark a run FAILED, with the traceback stored as a blob; events after a run ends are refused; `recover()` marks runs of dead processes ABANDONED | Tests, including a subprocess killed with `os._exit` mid-run |
| Concurrency | WAL plus BEGIN IMMEDIATE per append; sequence numbers assigned inside the transaction | Test: 4 processes × 62 events gave a gap-free, verified chain of 249 events |
| Portability | `export_bundle` writes JSON lines; `verify_bundle` re-verifies without SQLite | Tests: round trip verifies; one edited line fails |
| Reproducibility context | Every run stores packages, git commit and dirty flag, and source hashes; `diff(run_a, run_b)` lists changed parameters, source files and metrics | Test: an edited source file appears in the diff |

## Demonstration: the whole pipeline as one recorded loop

This is the first time all stages ran together. Data: `heart.csv` (Kaggle `fedesoriano/heart-failure-prediction`). Generator seed 7, split seed 7, 5 rounds × 100 hypotheses. Total time 46 s on Python 3.13.17 with NumPy 2.5.3.

1. **Data engine.** Verdict READY_WITH_WARNINGS (Cholesterol zeros treated as missing). The data-derived TaskSpec was 19 features, 2 classes, goal `target_noise`. Split: 551 / 184 / 183.
2. **Rounds.** Each round passed through the generator, the five-layer hypothesis filter and the math update gate (mathematical status PASS). Development scoring then fit `GeneratedModel` on train and scored on development, and the score was fed back to the generator with `observe`.

| Round | Generated | Filter passed | Math passed | Scored | Best dev balanced accuracy |
|---|---:|---:|---:|---:|---:|
| 1 | 100 | 85 | 28 | 28 | 0.8631 |
| 2 | 100 | 78 | 23 | 23 | **0.8741** |
| 3 | 100 | 79 | 27 | 27 | 0.8631 |
| 4 | 100 | 83 | 24 | 24 | 0.8729 |
| 5 | 100 | 82 | 27 | 27 | 0.8631 |

Feedback did not produce a steady improvement across rounds. The best candidate came from round 2 and was freshly *constructed*, not mutated or crossed from earlier winners: linear plus random-Fourier features, Brier loss, dual-memory update, identity preconditioner. In this run, the feedback loop did not produce the winner.

3. **Baselines.** Logistic regression (4 C values) and histogram gradient boosting (4 configurations) were trained on the same split. The best by development score was boosting (learning_rate 0.1, max_depth 3) at 0.8607.
4. **Preregistration**, recorded with a leaderboard snapshot hash. The two entries were the best generated candidate across all rounds (round 2, dev 0.8741) and the best baseline.
5. **One test look**, through `card.unseal_test(..., ledger=recorder)`:

| Method (preregistered) | Dev balanced accuracy | Test balanced accuracy (183 rows) |
|---|---:|---:|
| Generated candidate `745dd6ba…` | 0.8741 | **0.8861** |
| Boosting baseline | 0.8607 | **0.8602** |

The paired bootstrap of the test difference (5,000 resamples) gave **+0.027, 95% CI [−0.008, +0.064]**, positive in 92% of resamples.

6. **A second test look was refused.** The message was `test set bac167c0… was already used (event 1606); refusal recorded as event 1613`.
7. **Verification.** 1,613 events verified against the published head, 7 artifacts (3.3 MB), 23 runs SUCCEEDED. The exported bundle verifies on its own. The winner's lineage reads: entity (generator) → READY (hypothesis filter) → PASS (math gate) → dev 0.8741 → preregistration → test 0.8861.

**Attack on the real journal.** On a copy, the generated candidate's test metric was edited from 0.8861 to 0.95, with the guards dropped first. `verify()` reported `seq 1608: event body does not match its hash`. The genuine journal still verified.

### What this result does and does not show

**It shows** that the pipeline runs end to end, that every decision is traceable, and that the holdout discipline works in practice.

**It does not show** that the generated algorithm is better than standard methods:
- The CI includes zero, so the difference is not significant.
- It is one public dataset ("historically exposed").
- The test set has only 183 rows.
- The generated side searched 129 scored candidates against only 8 baseline configurations.
- The generated learner is a linear model over fixed feature maps with an adaptive optimizer, which is unsurprisingly competitive on a small tabular task.

The run was executed twice. The first run measured the slow recorder (see below), and the second ran after the speed fix. Everything is seeded, and the second run reproduced the first exactly: same winner, same dev and test numbers, same refusal. That is a deterministic recomputation, not a second selection. It does illustrate a stated limit: a **new journal does not know about holdout uses recorded in another journal**.

## Throughput (this container, 4 vCPU, WAL, synchronous=NORMAL)

| Measure | Value |
|---|---|
| Batched appends (`with rec.batch()`) | ≈ 27,000–30,000 events/s |
| One transaction per event | ≈ 6,300–7,200 events/s |
| `verify()` over 21,003 events | 0.21–0.29 s |
| Leaderboard over 20,000 metrics | ≈ 0.1 s |

**Disclosure:** the first pipeline run measured about 500 events/s batched. The check that a run is still open scanned the run's entire history on every append, which is O(n²). A composite index `(run_id, kind, seq)` and a two-query state check fixed it. All tests were re-run.

## Limits

- **Hashes are not signatures.** Whoever controls the file can rebuild a different, internally consistent journal. Only a head published somewhere else detects that. Publish `head()` with results.
- **The holdout ledger is per journal.** It cannot know about uses of the same test set in other journals or outside the recorder.
- **The recorder trusts the caller's labels** for what a dataset fingerprint, split or metric means.
- **Durability:** with `synchronous=NORMAL`, the last transactions can be lost on power failure, though not corrupted. `durable=True` uses FULL.
- **Scale:** queries scan JSON events in Python. That is fine at the tested size of about 21,000 events per journal; very large journals would need materialized indexes.
- **Crash recovery** decides "dead process" by host name and pid. It is not suitable for runs spread across machines that share a file.

## Checksums

| File | SHA-256 |
|---|---|
| recorder.py | `b49058c85d69c80b204f3549d2b5572886abf69e69c643a2c0ff692dc22fcd5e` |
| test_recorder.py | `3dfa54a04e3b8d8c6715ac7469c8787c39a6d7bcf190c93ad3d8baf890d6beaa` |
| demo_pipeline.py | `d8b0413f6d8d3d3a130ff390599705531c98aafbee6de0b8aed1679af6ba39ff` |
| pipeline_run.bundle.jsonl | `b985caf986cd1796254650968c452ff099cfa45df82b533341fbb71044c05582` |
| published head | seq 1613, `2c4c42576ccd58706af89955edc31b271d3f38b566aa95647f728da68a1efe3e` |
