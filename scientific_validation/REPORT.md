# Scientific validation engine — reproducible evidence, one frozen candidate

This engine performs a controlled experimental evaluation of one generator-selected ML candidate. It does not turn benchmark scores or mathematical admission receipts into proof of novelty, full convergence, independent replication, or usefulness in deployment. The ten requested scientific requirements remain separate, scoped decisions. Six requirements have executable experimental checks within this contract. Novelty, full mathematical foundation, independent reproduction and real-world application remain external requirements: this release deliberately cannot mark them PASS or certify a new algorithm on its own.

## Delivery and dependencies

Four files: `scientific_validation.py`, `test_scientific_validation.py`, `REPORT.md`, and `demonstration.sqlite`. Existing generator and mathematical engines are unchanged. The SQLite file holds raw data, every attempted development configuration, predictions, frozen settings, timing/memory measurements, event history, and review material. No paid service or network download is required for the built-in demonstration.

Tested environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, Linux. Native numerical work and estimators use one thread. `threadpoolctl` is installed with scikit-learn.

```bash
python -m pip install numpy==2.3.5 scipy==1.17.0 scikit-learn==1.8.0
export SCIENCE_GENERATOR=/absolute/path/to/hypothesis_generator.py
python -m unittest test_scientific_validation -v
python scientific_validation.py inspect demonstration.sqlite
# Creates a NEW experiment; never overwrites or reuses a consumed experiment.
python scientific_validation.py demo my_new_study.sqlite
```

The reviewed generator must match SHA-256 `8ce3010e741603c9627c85f37944586c4cde2155eeb24660adcfd58ca5931b3f`. By default it is loaded from `../generator_only/hypothesis_generator.py`; `--generator` or `SCIENCE_GENERATOR` overrides that location, not its required hash. The upstream handoff integration test also uses the existing math-update filter and verifier if present; it explicitly skips that optional test if they are absent.

The same protocol can be run in separate processes using `develop`, `freeze`, then `evaluate`. `inspect` recomputes decisions from the completed artifact without refitting or consuming test data again. The supplied artifact is already consumed; attempting to evaluate it again must fail.

## Supported experiment contract

Numeric finite tabular classification or single-output regression; integer public class schema; 2–20 declared dataset units of one task kind. Custom datasets are supplied through the Python `Dataset` API, including real entity/group IDs where observations are dependent. This is not an evaluator for every ML task, neural architecture, time-series forecasting protocol, or mathematical family.

`ScientificEngine.accept_handoff(packet, live_math_engine)` replays the existing engine's live admission receipt and checks its handoff binding, then returns a canonical generator record suitable for `develop(datasets, candidates=[record])`. The receipt is retained in the frozen intake history. Receipt scope is not silently enlarged: scientific tuning, feature/class adaptation, or a different dataset can exceed its original assumptions, and full mathematical validity remains unverified. Direct canonical proposals can also be evaluated as exploratory candidates; the demonstration uses that route rather than treating an arbitrary curvature screen as established scientific quality.

### Selection and leakage guards

* Duplicate feature rows and supplied entity groups stay within one approximately 60/20/20 train/development/test split. Training must contain all declared classes. Splitting is not stratified.
* The engine rejects observable cohort overlap: identical dataset content, shared feature rows, and same-source supplied entity groups. Different representations of the same hidden subjects cannot be detected without provenance/group information. Independence is an assumption, not something hashes prove.
* Candidate and baseline fitting see training labels only. Scalers live inside fitted estimators. Development selection sees development labels; final fitting uses train plus development. Test labels enter metric computation only after final model prediction. Fixed corruption recipes use training-feature statistics only; regression normalization uses the relevant training-label variance only.
* The candidate architecture and learning rate are selected globally across development datasets. Baselines are tuned per dataset; the reference family is chosen on development scores. No best-test baseline or per-dataset candidate is selected.
* SQLite commits `CONSUMED` and a hash-linked event before the public API can read the held-out partition. Parallel/reopened consumption is refused. Caught failures become terminal `FAILED` records; a hard process kill can leave `CONSUMED`, which is also non-retryable. Restarting the program does not reset the experiment.
* The source fingerprint, complete development record, settings, metrics, grids, corruption/scaling recipes and chosen references are frozen before final fitting. Replay checks model/data identities, every planned cell, raw predictions, recomputed scores, resource scope and refit evidence.

This is a trusted-local execution protocol. SQLite is not encrypted, hashes are not signatures, the owner can read data outside the API or fabricate an entirely new journal, and undisclosed earlier exposures cannot be detected. The engine does not claim a universal zero-leakage guarantee. Recreating a new artifact on the same public datasets does not create a fresh scientific holdout.

### Fair comparisons

The default development budget is 24 configurations per dataset for the generated family, each fixed-controller search, and each of five baseline families. The adaptive and fixed search both receive the same twelve generated structures and two rates (`0.02`, `0.08`). This matches the proposed mechanism's search opportunity. A configuration label is not a literature novelty claim, and different parent records can describe effectively identical fixed variants.

Baselines: scaled logistic regression/Ridge, scaled SVC/SVR, Extra Trees, histogram gradient boosting, and scaled nearest neighbors. Grids are visible in `baseline_grids`; all 24 options are attempted. Any failed development fit prevents a fair-benchmarking PASS. These are finite meaningful grids, not a claim that every competitor is optimally tuned. Equal configuration counts are not equal wall-clock budgets; all actual measurements are retained.

Final methods: selected adaptive candidate; matched constant-0.5 controller; independently development-tuned constant-0.5 search; `first_only` control; and the five tuned baseline families. In this generator, `first_only` fixes the second controller influence at 0.5; it does not remove a learner component. Each generated fit receives the same 160 update steps.

### Statistics and criteria

For comparison j, let d_ds be the candidate-minus-comparator score for dataset d and fitting seed s. First compute the within-dataset mean x_d. The reported mean effect is the equal-weight mean of x_d. Repeated seeds describe training variability but do not create new independent datasets.

The two-sided sign-flip p-value enumerates every sign assignment at dataset level, counting absolute permuted sums at least as large as the observed absolute sum (floating comparisons use a conservative tolerance). Valid inference requires independent dataset units and sign exchangeability/symmetry under the null. Across nine frozen comparisons, Holm corrects p-values. Reported intervals are Bonferroni simultaneous Student-t intervals over dataset means, requiring approximately normal independent cluster means; four clusters make these intervals especially uncertain.

Primary statistical targets are the development-chosen reference and the independently tuned fixed-controller family. They must both have positive mean effects and Holm-adjusted p <= 0.05. Accuracy additionally requires a point estimate at least 0.005, simultaneous interval lower bound above zero, and declared unexposed datasets. This is not a confidence bound exceeding the practical margin. The other seven comparisons remain reported and enter multiplicity correction; PASS does not mean superiority to every one.

Four independent datasets have minimum attainable two-sided exact p = 2/16 = 0.125. Five seeds cannot change that. This demonstration is incapable of establishing the requested 0.05 superiority claim, even if every dataset effect were positive. More genuinely unexposed datasets, chosen before evaluation, would be necessary; repeatedly re-splitting known datasets would not solve this.

Robustness conditions are clean data, Gaussian feature noise (0.15 training standard deviations), 5% feature-entry replacements at training mean +/- 6 training standard deviations, and a +0.5-standard-deviation corruption on a seeded 20% of feature columns. The same arrays are used for every method. Required mean corruption drop <= 0.10 and comparator-relative disadvantage <= 0.02 apply to every dataset/condition. This is a finite artificial stress suite on real data, not validation of true domain shift, label noise, adversarial security or all outliers.

Each final fit runs in a fresh isolated Python process, with a 120-second timeout. Fit time excludes imports and construction; prediction time is the median of one initial and four repeated clean predictions. Peak RSS includes the interpreter, libraries, input arrays, fitting and inference. The frozen size study uses nested train-only, class-covering 25%, 50%, and 100% samples and all five fitting seeds for candidate and the preselected reference; full-size fits are reused. Required median ratios: fit <= 5x, predict <= 5x, process RSS <= 2x, and candidate fit growth from quarter to full data <= 12x. These are explicit research criteria, not universal optimality constants. Shared-machine timing and a few small data sizes cannot establish asymptotic or production scalability.

## Frozen real-data results

The engine completed the study and returned **NOT_DEMONSTRATED**, with `accepted=false`. The candidate did not satisfy the requested scientific standard. Its mean score was below the preselected reference and both fixed-controller controls. These observations do not prove inferiority in general: the small retrospective study has weak statistical power.

All 672 development fits, 180 final comparison fits, 80 smaller-training-size fits and 4 deterministic refits succeeded: 936 fits in total, including 264 fresh final worker processes. Each of four datasets used five fitting seeds. All four same-machine candidate refits reproduced prediction hashes exactly. The consumed final evaluation took 375.43 seconds, including process startup and full resource studies.

Mean held-out balanced accuracy over the five fitting seeds; the same held-out rows are reused across seeds:

| Dataset | Adaptive candidate | Matched fixed | Tuned fixed search | Frozen reference | Reference family |
|---|---:|---:|---:|---:|---|
| cancer | 0.973329 | 0.966465 | 0.952881 | 0.953236 | linear |
| digits | 0.895758 | 0.925428 | 0.941778 | 0.969392 | kernel |
| iris | 0.805079 | 0.902222 | 0.919048 | 1.000000 | kernel |
| wine | 0.956863 | 0.960784 | 0.964706 | 0.960784 | boosting |

All tuned baseline families are reported, including those not chosen as the primary reference:

| Dataset | Linear | Kernel | Extra Trees | Boosting | Neighbors |
|---|---:|---:|---:|---:|---:|
| cancer | 0.953236 | 0.979908 | 0.971871 | 0.959815 | 0.959459 |
| digits | 0.955248 | 0.969392 | 0.975781 | 0.956967 | 0.958915 |
| iris | 1.000000 | 1.000000 | 0.976190 | 0.952381 | 0.880952 |
| wine | 0.960784 | 0.980392 | 0.945098 | 0.960784 | 0.921569 |

Against the frozen reference, the candidate won 6, tied 2 and lost 12 of 20 repeated-fit comparisons. These are descriptive counts, not 20 independent experiments.

| Comparison | Mean candidate effect | Two-sided p | Holm p | Simultaneous interval |
|---|---:|---:|---:|---|
| reference | -0.063096 | 0.375 | 1.000 | [-0.409604, 0.283412] |
| fixed_matched | -0.030968 | 0.375 | 1.000 | [-0.198745, 0.136810] |
| fixed_search | -0.036846 | 0.375 | 1.000 | [-0.245864, 0.172173] |
| first_only | -0.033629 | 0.250 | 1.000 | [-0.199474, 0.132215] |

The primary reference effect is **-6.31 percentage points**; the tuned fixed-controller effect is **-3.68 points**. No significant superiority is demonstrated. Baseline references were fixed on development data: for example, cancer used linear even though kernel later scored higher on test. The engine did not replace the reference after seeing that outcome.

The selected architecture uses robust scaling, linear plus Fourier features, Brier loss, RMS preconditioning, median-of-means aggregation, a memory-step mechanism and coupled controllers. Its controller is `q=(max(alignment,disagreement)+(1-time))/2`; the exact canonical record/equations/runtime are retained in the frozen manifest. This is a generated combination of existing components, not an established novel method.

The largest candidate corruption drop was 19.38 points on digits under feature outliers, exceeding the 10-point criterion. Wine also exceeded that drop criterion. Iris had baseline-relative disadvantages beyond the 2-point tolerance. Digits under outliers did show a local tradeoff: candidate 0.702000 versus reference 0.361003, despite much worse clean performance. This does not rescue the frozen primary claim or establish a new robustness discovery; it could motivate a separately preregistered future hypothesis on genuinely unused data.

| Dataset | Full-size fit-time ratio | Prediction-time ratio | Peak process RSS ratio | Quarter-to-full fit growth |
|---|---:|---:|---:|---:|
| cancer | 12.061 | 0.797 | 1.003 | 1.330 |
| digits | 8.044 | 0.036 | 1.000 | 2.740 |
| iris | 16.551 | 0.176 | 1.000 | 1.100 |
| wine | 0.665 | 0.028 | 1.000 | 1.226 |

Fit-time ratios on cancer, digits and iris exceed the prespecified 5x limit. The engine therefore fails efficiency even though prediction ratios and measured size growth are small. Near-identical process RSS largely reflects shared interpreter/library/input overhead, not proof of equal model-state memory. Timing used a shared host; part of the run overlapped a single review worker. These are recorded operational measurements, not an uncontended production performance claim.

| Requirement | Decision | Meaning here |
|---|---|---|
| accuracy_generalization | FAIL | Mean effects/interval requirements failed; datasets historically exposed. |
| efficiency | FAIL | Frozen fit-time ratios failed. |
| fair_benchmarking | PASS | Complete declared grids, matched controller search and frozen references. |
| independent_reproducibility | UNVERIFIED | Same-machine repeats and separate code replay do not establish external replication. |
| mathematical_foundation | UNVERIFIED | No full proof for this tuned learner and data contract. |
| novelty | UNVERIFIED | No independently established literature contribution. |
| real_world_validation | UNVERIFIED | Real public benchmark data, but no demonstrated domain/deployment usefulness. |
| robustness | FAIL | Frozen drop and reference-relative criteria failed. |
| statistical_validation | FAIL | No primary superiority after correction; four independent dataset units assumed. |
| working_implementation | PASS | All planned candidate fits and scoped deterministic refits executed. |

The benchmark uses real observations from breast cancer, handwritten digits, iris and wine, not synthetic performance data. All are explicitly flagged historically exposed. Synthetic datasets occur only in protocol/integrity diagnostics, not as evidence for real-world candidate performance.

A separate implementation rechecked all **1,392** development/held-out prediction-score cells and all nine comparison statistics. It uses rational balanced-accuracy arithmetic and rational sign enumeration, imports neither the scientific engine nor scikit-learn, and matched the reported means, p-values, Holm corrections and interval endpoints within 1e-12. That guards against a shared metric/statistics implementation error; it is still same-environment checking, not an independent scientific replication.

The completed artifact is permanently consumed (`holdout_uses=1`). A second real evaluation command was attempted and refused without changing it. Read-only inspection remains possible. The initial source used for development was `98bae83d…`; before freezing or accessing final results, the disclosed evidence-validation correction produced the final `7d760f43…` source. It changed no candidate proposals, training procedure, development scores, selected settings or scientific criteria.

Frozen manifest SHA-256: `94a5084f7b8d411f609e3e472343b7e7c73dd2fd54d336c8ace7f098d8146e9d`.

Canonical completed-evidence SHA-256: `0c49428b0b6c0156bd98c0c4a3f548c193bb36fc270df4e46d44a70580ea9d40`.

These bind this local experiment; they are not public-key authentication.

### Wiring passed mathematical hypotheses into scientific evaluation

The caller retains the live upstream engine that issued admissions. The following pattern selects one canonical algorithm on development data, then evaluates it; it does not require hand-editing scientific gate fields:

```python
from scientific_validation import Dataset, ScientificEngine

science = ScientificEngine("new_study.sqlite", generator_path="/path/to/hypothesis_generator.py")
admitted = []
for item in math_batch["results"]:
    if item["status"] == "READY_FOR_SCIENTIFIC_VALIDATION":
        packet = math_engine.handoff(item["candidate"], item["audit"])
        admitted.append(science.accept_handoff(packet, math_engine))
        if len(admitted) == 12:
            break
if not admitted:
    raise ValueError("No mathematically admitted hypotheses to evaluate")
# At least two suitable datasets; group IDs must represent real dependencies.
datasets = [Dataset("domain_a", X_a, y_a, classes=C_a, groups=groups_a,
                    historical_exposure=True, source="actual provenance a"),
            Dataset("domain_b", X_b, y_b, classes=C_b, groups=groups_b,
                    historical_exposure=True, source="actual provenance b")]
science.develop(datasets, candidates=admitted)
science.freeze()
decision = science.evaluate()
```

`historical_exposure=False` is a factual provenance declaration about genuinely unused data, not an optimization switch. No code can establish that declaration from a local JSON object. New experiments created after reading benchmark outcomes must keep those datasets historically exposed.

The live handoff test completes this route with the existing pinned math-update engine and verifier, confirms retained admission evidence, and rejects a forged packet. Its admission uses learning rate 0.001, because the existing filter rejects the first twelve binary-classification proposals at its 0.08 default. Scientific runs may change runtime/task dimensions; their full mathematical applicability consequently stays UNVERIFIED. This is not a hidden bypass of a convergence proof.

### Separate challenge review and disclosure

A separate reviewer agent authored 40 scenarios plus a 16-scenario contract supplement before seeing the implementation. Their original files were hashed, sealed and not opened until the first complete implementation froze. This is useful separate-agent review on the same system; it is not independent scientific replication or an external laboratory.

| Stage | Pass | Fail | Out of contract | Executed checks |
|---|---:|---:|---:|---:|
| Initial frozen source `98bae83d…` | 54 | 1 | 1 | 970 |
| Final source `7d760f43…`, exposed regression | 55 | 0 | 1 | 286 |

The genuine initial failure, E04, comprised three strict-schema bugs: Python accepted `True` as integer event sequence 1, classification normalizer 1, or development normalizer 1. No algorithm was scientifically accepted, but malformed evidence was incorrectly treated as usable negative evidence. The fix checks numeric types explicitly and uses canonical comparisons for bound structures. Tests additionally cover Boolean environment metadata and refit predictions. The final source also caches identical corruption reconstruction within each dataset/seed; the learning, search, scoring formulas and acceptance thresholds were not changed.

The initial source snapshot and original adapter remain preserved. Its workers had already imported/forked the frozen source when the correction was applied to the live file, so its results explicitly record `source_unchanged_at_end=false`; all live journal/evaluate checks had finished before that edit. The snapshot binds its actual original implementation. Fresh classification/regression controls and all three original failures were then replayed on the corrected source.

The final exposed replay covers all 56 original scenarios, with 36 representative cell omissions instead of repeating 720 exhaustive omissions. The initial full run tested all 720. An intermediate exposed exhaustive run stopped at 200/970 checks at the parent agent's request to avoid redundant work and timing contention; no complete result or additional success count is claimed for that prefix. Final regression is not presented as fresh unseen validation.

E20 is out of contract: its author assumed `first_only` removes an algorithm component, whereas the pinned generator fixes the second controller influence at 0.5. It is excluded and never counted as a pass. Six earlier source-review findings were corrected before the initial freeze: cohort aliases, frozen recipes, terminal failure receipt, failed development budgets, raw data schema/content binding, and malformed nested input handling.

Original corpora SHA-256:

* `7c568790f5abc3b97f9edba1fd287aaf9de3ca1ba558035c1d24e827a5e32dbe`
* `9aabeb9adbb6b98fd599f590712ae801e2bd5801ab7907f96ee8cf8d78bfaa77`

The SQLite `review` table archives the original source, corpora, adapters, diagnostic fixtures, exact selection manifest, initial/final results and their embedding index. It also retains the developer's first-party source snapshots and a separate metric/statistics replay program. Review records are supporting audit material; they do not grant any scientific gate PASS.

## Reproducing the four-file package in a clean directory

The SQLite artifact also contains byte-exact snapshots of its existing first-party dependencies. They are audit/dependency copies, not new mathematical engines. If the originals are unavailable, extract them to a temporary directory with the following script, then set `SCIENCE_GENERATOR` to the printed path. The expected hashes are pinned here as well as checked by the engines; no network fetch is needed.

```python
import hashlib, json, sqlite3, tempfile
from pathlib import Path
root = Path(tempfile.mkdtemp(prefix="science_dependencies_"))
snapshots = [
    ("pinned_generator_source", "generator_only/hypothesis_generator.py",
     "8ce3010e741603c9627c85f37944586c4cde2155eeb24660adcfd58ca5931b3f"),
    ("pinned_math_filter_source", "math_update_gate/math_filters.py",
     "6d3dd542732e68008eac4ed88bf133976493231745842e5e4a3916aa85b3f644"),
    ("pinned_verifier_source", "math_verifier/math_checker.py",
     "37dc50931bfe2512875558e5660de692680c8b232333a98885503da9697205fc"),
]
with sqlite3.connect("demonstration.sqlite") as db:
    for key, relative, expected in snapshots:
        source = json.loads(db.execute("SELECT payload FROM review WHERE name=?", (key,)).fetchone()[0])
        raw = source["text"].encode("utf-8")
        assert hashlib.sha256(raw).hexdigest() == expected
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
print(root / "generator_only/hypothesis_generator.py")
```

In a clean extraction of the four delivered files, the isolated command `python -I test_scientific_validation.py -v` runs the test suite without relying on the workspace import path. With the three dependency snapshots materialized, it includes the live upstream handoff test. The engine's `inspect` command and the separate replay operate on the released completed evidence; neither is a new unseen-data experiment.

To inspect the audit entries, use `SELECT name FROM review ORDER BY name`. Each archived source/log/corpus payload contains `text`, `bytes`, and `sha256`. Verify the bytes before executing extracted source. For example, extract `independent_statistics_script`, write its `text` to a temporary `.py` file, then run it with `demonstration.sqlite` as its argument. Its saved result is `independent_statistics_result`. The `scientific_decision` row is the derived decision, while `state.evidence` remains the canonical completed experimental record.


## Final package checks

A clean extraction materialized the three pinned dependency snapshots into a temporary directory. With `SCIENCE_GENERATOR` pointing there, `python -I test_scientific_validation.py -v` completed **32/32 tests, no skips, in 13.876 seconds**. Read-only engine inspection returned the same `NOT_DEMONSTRATED` decision and completed-evidence hash. The extracted separate statistics program again returned `MATCH` for 1,392 score cells and nine comparisons. These checks relied on no original workspace import path or source file. They ran on the same machine and installed dependency versions, so independent-environment scientific replication remains unverified.

The final SQLite integrity check returned `ok`. The implementation and experiment were not retuned after held-out scores were exposed. First-party source snapshots, original/final challenge records, public tests, clean-extraction logs, raw predictions and results are all included in the artifact.

| File | SHA-256 |
|---|---|
| scientific_validation.py | `7d760f43afcd37a74f402fa4f05154964a794a601a544597fa975613702fc150` |
| test_scientific_validation.py | `457b99d2f2fda3250d996f826e95dffbbba92f65400d5766f03f61939ff71e5d` |
| demonstration.sqlite | `02c83e0acff12c59629bd7c5edd86c61e596af0e1280846f8ac9e0e084b33299` |

`REPORT.md` is excluded from its own hash table. The archive contains exactly four delivered files. Whole-file release hashes bind bytes for comparison; they do not authenticate an author or establish scientific validity.
