# Data intelligence engine

Takes a raw table (a CSV file, NumPy arrays or a scikit-learn bundled dataset) and produces:

1. **Leakage-safe train / development / test partitions.** Exact duplicate rows and declared entities never cross partitions. The split is stratified, or forward in time when a time column is declared.
2. **An auditable data card.** It records every column decision (keep, encode, drop, and why), every warning with its evidence, the train-only diagnostics, and the hashes that bind all of it.
3. **A generator `TaskSpec` whose research goals come from measured evidence.** The goals are `outliers`, `target_noise`, `ill_conditioned`, `class_imbalance`, `high_dimension` and `distribution_shift`. Previously they were typed in by hand.
4. **Exports:** generator `Partition` objects, a `scientific_validation.Dataset`, and a sealed test partition. Opening the test partition goes through the recorder's holdout ledger.

Standard library plus NumPy. scikit-learn is used only to read its bundled datasets.

## Files

| File | Purpose |
|---|---|
| `data_engine.py` | Engine, data card, loaders and CLI |
| `test_data_engine.py` | 26 contract tests |
| `demo_data_engine.py` | Planted-defect benchmark, real datasets and the leakage-boundary check |
| `demonstration.json` | Full output of the final demonstration run |
| `REPORT.md` | This file |

```bash
python -m unittest test_data_engine -v
python data_engine.py analyze heart.csv --target HeartDisease --json card.json
python data_engine.py analyze builtin:wine
python demo_data_engine.py --dev-dir DEV --heldout-dir HELDOUT --seeds 20 --output demonstration.json
```

```python
import data_engine as de
card = de.DataEngine().analyze(de.read_csv("heart.csv"), "HeartDisease", time=None, groups=None)
print(card.summary())
spec = card.task_spec(generator_module)            # data-derived goals
train, dev = card.generator_partitions(generator_module)
X_test, y_test = card.unseal_test("final preregistered comparison", ledger=recorder)  # once
```

## Leakage boundary

- **Training rows only.** Every learned transformation is computed from training rows alone: column types, category vocabularies, imputation medians, disguised-missing sentinels, redundancy and leakage checks. So is every diagnostic. Development and test rows are only transformed.
- **Labels outside training are used for exactly two things.** One is proportional stratification of the split. The other is reading the target's value set (classification versus regression, and the class list). Feature–label associations outside training are never read.
- **Split-time duplicate detection reads feature values of all rows, never labels.** Identifier-like columns are excluded from the duplicate key, so a fresh ID cannot hide a re-recorded row.
- **The test partition is sealed.** `unseal_test` needs a stated purpose. With a ledger attached, a second use is refused.
- **Science export.** `science_dataset` passes all rows to `scientific_validation`, which applies its own split. Used on its own, the encoder statistics (vocabulary and medians, no labels) come from this engine's training rows, so some of the validator's test rows may have contributed to them. The pipeline (`recorder/run_pipeline.py`) removes this caveat: it declares the validator's own split to the engine (see below), so encoders are fitted on exactly the rows the validator trains on.

The boundary was **tested on real data**. The demo replaces every kept feature cell of the development rows, and separately of the test rows, with fresh values: numbers become 10⁶ plus noise, and categories become unseen strings. Only rows alone in their duplicate bucket are changed, so the split itself cannot move. Across **8 real datasets × 2 partitions**, the split, every fitted decision, every warning (except the transform-time counter that describes those rows) and the training matrix were **identical in all 16 cases**. For example, 84,656 changed cells in Adult produced zero changed decisions.

## Declared splits and validator alignment

- **`analyze(..., split={"train": rows, "development": rows, "test": rows})`** accepts an externally declared split, given as table row indices. Examples are an official benchmark split, or the split a downstream validator will use. It is **validated, never repaired**. Any of these blocks the card with `SPLIT_DECLARED_INVALID` or `CLASS_ABSENT_FROM_TRAIN`:
  - a row with no target;
  - a row placed twice, or a usable row left out;
  - an exact duplicate or declared entity spread over two partitions;
  - a class missing from training.
- **`card.partition_rows` and `card.used_rows`** expose the row mapping. `science_dataset(..., groups=labels)` lets a caller hand the validator a coarser grouping.
- **`cohort_overlap`** compares rows only when the shared columns cover at least half of the narrower table and number at least two. Otherwise it reports `comparable: False`, since two unrelated datasets that share only an `Age` column must not be called overlapping.

## What it detects (frozen policy, SHA-256 `816623cb…`)

**Columns:**
- Identifiers, recognized three ways: a unique ID-like name, unique strings, or a dense unique integer sequence.
- Free text.
- Constant columns.
- Mostly-missing columns (over 50% missing), which keep only an indicator.
- Unparseable numbers.

**Encoding and redundancy:**
- One-hot encoding with `__other__` and `__missing__` levels. Frequency encoding beyond 25 categories, with an indicator when values are missing.
- **Disguised missing values:** a fixed sentinel such as 0, −1 or 999 that forms an *isolated point mass*. The rule requires 1–50% of rows, outside the other values' range, separated by at least 0.5 IQR.
- **One-to-one redundant column pairs.**
- **Exact linear dependencies** among encoded features, naming the columns involved.

**Leakage:**
- Single-column leakage, using out-of-fold 1-D kNN accuracy at or above 0.99 for classification, and binned R² or Spearman at or above 0.99 / 0.995 for regression.
- **Multivariate linear leakage** for regression (out-of-fold linear R² at or above 0.999, for example `total = part_a + part_b`).

**Data structure:**
- Exact and conflicting duplicates.
- Files sorted by target.
- Temporal shift, via adversarial AUC between early and late training rows (only with a declared time column).

**Diagnostics behind the goals:**
- Standardized condition number (one level dropped per one-hot block) and pairs with |r| ≥ 0.98.
- Tukey far-out outlier rate. Columns dominated by one value are skipped, because their IQR is degenerate.
- Imbalance ratio.
- Train-only cross-validated probes: ridge, random-Fourier ridge and kNN. These are diagnostics, not benchmarks.

A `BLOCKED` verdict comes from any of: leakage, too few rows, a class with fewer than 5 rows, a constant target, no usable features, or an infeasible split.

## Demonstration results

Engine SHA-256 `8569205e…`. Run on Python 3.13.17 and NumPy 2.5.3, about 3 minutes. The demonstration was first run under `d0d37cba…` and re-run after the declared-split additions; every detection, false alarm, real-data finding and leakage-boundary result was identical.

### 1. Planted-defect benchmark

Each defect was injected into a clean synthetic task (600 rows, 8 features), with 20 seeds per cell. Each entry gives two counts: the original seeds, then a fresh seed set run after the last rule change.

| Planted defect | Classification | Regression |
|---|---|---|
| label noise (25% flips / noise = 1.5× signal sd) | 20 / 20 | 20 / 20 |
| outliers (4% rows, one value at 8–12 σ) | 19 / 20 | 20 / 20 |
| collinear copy (r ≈ 0.9999) | 20 / 20 | 20 / 20 |
| class imbalance (≈ 4:1) | 20 / 20 | — |
| 80 extra noise features | 20 / 20 | 20 / 20 |
| target copy + small noise (leak) | 20 / 20 | 20 / 20 |
| target = part_a + part_b (sum leak) | — | 20 / 19 |
| numeric identifier (shuffled dense integers) | 20 / 20 | 20 / 20 |
| string identifier | 20 / 20 | 20 / 20 |
| free-text column | 20 / 20 | 20 / 20 |
| constant column | 20 / 20 | 20 / 20 |
| 75%-missing column | 20 / 20 | 20 / 20 |
| disguised missing (15% zeros in a ~N(200,30) column) | 20 / 20 | 20 / 20 |
| 20% missing values | 20 / 20 | 20 / 20 |
| one-to-one redundant pair | 20 / 20 | 20 / 20 |
| 10% exact duplicate rows | 20 / 20 | 20 / 20 |
| rows sorted by target | 20 / 20 | 20 / 20 |
| temporal drift (2σ over the time span) | 19 / 19 | 19 / 18 |

**Totals:** 677 / 680 detected on the original seeds and 676 / 680 on the fresh seeds. The misses are borderline drift, outlier and sum-leak cases.

**False alarms on clean tasks** (20 tasks per kind and seed set):
- **Regression:** no flags at all.
- **Classification:** `target_noise` fired in **9 of 40** clean tasks. **This is a real weakness.** The "clean" task has 4.3% irreducible label disagreement. The flag's metric (rows misclassified out-of-fold by all three probes) ranges 0.02–0.09 on clean data against a threshold of 0.08, and 0.18–0.28 with 25% flips. It responds to any irreducible error above about 8% and cannot tell label errors from class overlap. A threshold near 0.13 would separate these cases cleanly. I left the policy frozen instead of tuning it to this benchmark.
- The **stationary temporal controls** produced no `distribution_shift` alarms.

### 2. Real datasets

All data came from Kaggle via its API. "Documented" means the issue is stated in the dataset description or is long-standing common knowledge about that dataset.

**Development datasets.** These were inspected while the rules were written, so they are not blind.

| Dataset (rows × cols) | Documented issues found | Other findings | Goals derived |
|---|---|---|---|
| Titanic (891 × 12) | PassengerId identifier ✔, Name free text ✔, Cabin 77.8% missing ✔ | Ticket high-cardinality; Age missing; 3 conflicting duplicates | outliers, target_noise |
| Telco churn (7,043 × 21) | customerID identifier ✔, blank TotalCharges ✔, "No internet/phone service" exact dependencies ✔ | 42 conflicting duplicates | class_imbalance, ill_conditioned, target_noise |
| Adult census (32,561 × 15) | education ≡ education.num ✔, `?` missing in workclass / occupation / native.country ✔ | missing occupation mirrors missing workclass (exact dependency) | class_imbalance, high_dimension, ill_conditioned, outliers, target_noise |
| Heart failure (918 × 12) | Cholesterol = 0 as missing (18.1%) ✔ | — | target_noise |

Development total: **11 / 11**. The scikit-learn sets behaved as expected: iris and wine were flagged as sorted by target, breast cancer as ill-conditioned, digits as high-dimensional, and diabetes for target noise.

**Held-out datasets.** Expectations were written into `demo_data_engine.py` before any run.

| Dataset (rows × cols) | Expected | Result |
|---|---|---|
| Pima diabetes (768 × 9) | zeros as missing in Glucose, BloodPressure, SkinThickness, Insulin, BMI; Pregnancies zeros are real | **2 / 5**: BloodPressure ✔, BMI ✔. Glucose ✘ (0.65% zeros, below the 1% minimum, predicted miss), Insulin ✘ (zeros sit 0.12 IQR from the other values, predicted miss), SkinThickness ✘ (gap under 0.5 IQR; I had predicted a catch). Pregnancies correctly **not** flagged ✔ |
| Breast cancer, Kaggle CSV (569 × 33) | `id` identifier; empty trailing column; collinearity | `id` ✔, empty column dropped ✔, ill_conditioned ✔ |
| Bike sharing, daily (731 × 16), time = `dteday` | `instant` row index; `cnt = casual + registered` leak | `instant` ✔; **BLOCKED** for linear leakage (out-of-fold R² 1.0000) ✔. Re-run with both columns dropped: distribution_shift ✔ (adversarial AUC 0.98), ill_conditioned (temp ≈ atemp) ✔ |

Held-out total: **6 / 9 documented issues**, with no wrong detections. Task type was correct on all three.

## Disclosures: changes after seeing data

1. **Smoke run on the development datasets.** The first disguised-missing rule (3-IQR fence) missed Heart's Cholesterol zeros, because on the training subset the fence fell just below 0. I replaced it with the isolated-point-mass gap rule. In the same pass:
   - The high-dimension ratio was raised from 0.10 to 0.20 features per training row (wine was flagged with 13 features).
   - The outlier rule now skips columns dominated by one value (digits pixels).
   - I added name-based identifiers, naming of empty CSV headers, exact-dependence reporting, and the multivariate linear-leak check.

   The name rule and the linear-leak rule were written knowing the *types* of issue documented for two held-out sets (an `id` column; a sum leak). They are therefore implementation checks there, not blind discoveries.
2. **The first frozen run crashed on held-out breast cancer.** Its header has a trailing comma but its rows do not. The reader now pads rows only when the missing trailing header cells are empty, and stays strict otherwise. The breast-cancer result is therefore post-fix.
3. **In the first complete benchmark, classification leakage was detected only 7 / 20 times.** Equal-frequency bins straddled the class gap. I replaced that check with out-of-fold 1-D kNN and re-ran on the original seeds plus a fresh seed set; both results are shown above.
4. **A final development-set fix:** Adult's missing `native.country` values were silently absorbed by frequency encoding. They now get an indicator and a report entry.

5. **Declared splits, validator alignment and the overlap-comparability rule** were added after the demonstration, for the pipeline. They do not change default analysis; the full demonstration was re-run and is identical.

The Pima and bike-sharing results were first seen from an already frozen engine. The only later change (item 4) does not touch those files, and their results are unchanged in the final run.

## Limits

- **These are rules with explicit thresholds, not a universal data-quality oracle.** Every threshold is in `Policy` and bound into each card by hash.
- **Disguised missing** catches only fixed sentinels that form an isolated point mass. Rare zeros (Pima Glucose) and zeros adjacent to a wide distribution (Pima Insulin) are missed.
- **Leakage detection** covers near-perfect single columns and exact linear reconstruction (regression). A post-outcome feature that is merely strong, or a nonlinear combination, is not caught.
- **`target_noise` measures irreducible error**, not label errors alone (see the false-alarm result above).
- **`distribution_shift`** is assessed only when a time column is declared.
- **Values treated as missing:** `""`, `NA`, `N/A`, `NaN`, `null`, `none`, `?`, `missing` and `#N/A`. A category literally called "None" becomes a `__missing__` level.
- **Identifier detection** uses names, uniqueness and dense integer runs. A unique, sparse, unnamed integer ID would be kept as a feature.
- **Speed and scope:** tested up to 32,561 rows (about 3 s). There is no streaming mode, and everything stays in memory.

## Data sources (Kaggle slug, file SHA-256)

**Development:**
- `yasserh/titanic-dataset`, `7d118fef…`
- `blastchar/telco-customer-churn`, `88be4b93…`
- `uciml/adult-census-income`, `250e154e…`
- `fedesoriano/heart-failure-prediction`, `948420b0…`

**Held-out:**
- `mragpavank/diabetes` (Pima), `698c203a…`
- `uciml/breast-cancer-wisconsin-data`, `1425d9af…`
- `lakshmi25npathi/bike-sharing-dataset` (`day.csv`), `a6bcf826…`

The data files are not committed. The full digests are in `demonstration.json`.

## Checksums

| File | SHA-256 |
|---|---|
| data_engine.py | `8569205ee91259b816d2488ded93709f8f026293b36fd3d4b03a7f16312cc3fd` |
| test_data_engine.py | `cba363ce720023601f4e7fc752cdda46881237762fb07b0ba17b728ae8988249` |
| demo_data_engine.py | `a365cb92eb5ebd20b37640e94567f91ac5c846f7475501ff6198638006183a80` |
| demonstration.json | `6d30c90bb16e7b739c33d2bd7fa85106e724aff825680b6d1734b73cff2bd0d3` |
