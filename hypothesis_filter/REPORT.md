# Five-layer hypothesis filter

## Accepted design and implementation plan

The user approved five ordered layers: coherence, uniqueness, prior-art novelty screening, technical potential, and research readiness. This component consumes hypotheses, not observations or benchmark results. It selects research candidates for the existing mathematical pipeline. It does not certify novelty or performance. Existing generator, math engines and scientific validator remain unchanged.

Four delivery files: one implementation, one test file, this report, and one SQLite artifact containing demonstrations, references and review material. Runtime dependencies: Python and the existing NumPy-based generator. The existing generator source is SHA-pinned. Unsupported representations are held for review rather than translated by guessing.

Implementation follows test-driven development and a separate reviewer. No further design approval is needed: the requested five-layer design was explicitly approved in this conversation. The plan is recorded here to preserve the four-file limit.

1. Test and implement strict canonical import, bounded exact rational controller normalization, full update identities, and parameter-family identities. Min/max rewriting and sparse polynomial identities must recognize algebraic aliases without using sampled equality as proof.
2. Test and implement five ordered gates, explicit unknown/skipped outcomes, candidate-specific prior-art dossiers, exact conditional signal-sensitivity witnesses and a declared Pareto research-priority policy. Standard losses or RMS scaling are not blacklisted; no score claims predictive quality.
3. Test and implement transactional persistent duplicate registration, immutable round records, deterministic replay and candidate-bound handoffs. Concurrent rounds cannot independently pass the same semantic algorithm. Keep all ten scientific gates unverified.
4. Test and demonstrate real generator batches of 100, 500 and 1,000, deliberate aliases/parameter variants, input leakage guards, reopened/concurrent operation and the existing mathematical-stage adapter. Freeze before opening the separate review corpus; disclose initial failures and exposed corrections. Package four files, byte-verified dependencies and reproducible audit material.

The review focuses on false equivalences, misleading novelty labels, parameter-only variants, evidence smuggling, tainted scientific inputs, duplicate races, forged receipts and source changes. Dataset/task-wide coverage is never inferred from domain names. The generator's invention grammar remains bounded; this filter does not extend it into every ML task.


## What each layer establishes

| Layer | Decision and evidence | What it does not establish |
|---|---|---|
| 1. Coherence | Strict regeneration through the pinned generator, canonical equations/fields, budgets, assumptions and falsification plan. Unknown representations are held. Noncanonical/unsupported payloads are discarded before persistence; only an input digest and fixed rejection/hold code remain. | Authenticated generation origin, correctness or freedom from undisclosed earlier research exposure. |
| 2. Uniqueness | Persistent exact update normal forms; recognized rational and lattice identities; unresolved response-profile collisions are held. Identity excludes seed, ancestry, name and task dimensions, which are not algorithmic inventions. | A complete functional-equivalence solver or bit-identical floating-point execution. |
| 3. Prior-art screen | Caller-registered implementation exclusions, coefficient-template variants and candidate-specific overlap dossiers against six inspected sources. Emits exact mechanism definitions and follow-up search queries. | Exhaustive literature search, patent clearance or proof that an admitted combination is new. |
| 4. Technical potential | Exact rational witnesses that at least two distinct training-feedback inputs can change update controls on the stated signal cube; executable component-domain mapping. | Real joint reachability of those inputs, convergence, accuracy or likely benchmark superiority. |
| 5. Research readiness | Falsifiable, bounded candidates retained on a like-for-like Pareto research-priority frontier. No quota forces a survivor. | An empirical or calibrated probability of success. A dominated candidate is lower priority under this policy, not scientifically refuted. |

The result is `READY_FOR_MATH_INVESTIGATION`, `HELD` or `REJECTED`. Every record has all five layer entries; later layers are `SKIPPED` after an earlier failure. Every candidate retains `novelty_status=UNVERIFIED` and all ten scientific gates `UNVERIFIED`. There is no overall confidence score that can turn unrelated family facts into support for a claim.

### Exact identities and conservative uncertainty

The normalizer uses sparse rational polynomials plus explicit `min` atoms. It uses `max(a,b)=a+b-min(a,b)`, exact polynomial cancellation, bounded-interval identities, and a canonical max-of-mins antichain. The lattice normal form includes associativity, commutativity, idempotence, absorption and distributivity. Complements are pushed through min/max using De Morgan identities before lowering. For example, `(min(a,b)+max(a,b))/2` equals `(a+b)/2`, and `1-min(a,b)` equals `max(1-a,1-b)`.

The proof assumptions are the declared signal cube `[0,1]^6` and ideal real arithmetic. This is reviewed normalization code with independently coded rational evaluation checks, not a Lean/Coq proof or a complete decision procedure for all piecewise-polynomial identities. Sparse expansion and lattice expansion are bounded; unsupported normalization is held rather than passed using a weaker identity check.

If different normal forms produce identical effective controls on 27 fixed, public rational signal points, the later candidate is **held**, not declared equivalent. These points are construction probes and include fixed anchors, axes and deterministic hash-derived rational points. They are not training, validation or test examples. Their agreement is deliberately not treated as a proof. This catches cases such as `min(x*y,x*x*y)` versus `x*x*y` without claiming a general inequality solver. Probe disagreement also does not prove distinct behavior along real training trajectories: joint signal reachability is a separate mathematical question.

Coefficient-family screening treats equal monomial/sign templates with different numerical weights as tuning variants, within this declared grammar. It is a research policy, not an equality proof. Set `reject_parameter_variants=False` when a specific theoretical reason justifies revisiting coefficients; the exact-identity and response-profile guards remain. Richer evidence/proof languages need a reviewed adapter and are currently held, not guessed into this schema.

### Cross-domain scope

Domains are inferred from executable components, not user-supplied labels: optimization, training feedback/control, robust statistics, probability, algebraic feature maps, harmonic analysis, randomized methods and information-theoretic losses. A Fourier map adds harmonic/randomized mechanisms; merely naming another domain adds nothing. This ontology does not make the generator universal. Its executable tasks remain finite numeric supervised classification and single-output regression, with six update mechanisms and a bounded controller grammar. The filter does not invent missing generator primitives or support every ML/mathematical domain.

The research frontier compares candidates only when task metadata and non-controller optimizer/loss/representation blueprints match. It favors more witnessed feedback inputs and affected controls, fewer controller nodes and fewer stored parameters. These are explicit research-allocation preferences. It does not presume that extra signals improve prediction, punish a different loss just for being nonconvex, or rank Fourier features below linear features simply because they cost more. Distinct blueprints preserve separate research opportunities. Frontier selection is within each batch; persistent identity/template checks span all rounds.

### Prior-art authority and limits

The inspected primary works are Adam, online hypergradient adaptation, YellowFin, adaptive gradient clipping in NFNets, the Lugosi–Mendelson robust-estimation survey, and Rahimi–Recht random features. The first five were inspected through alphaXiv full-text retrieval; the last through the official NeurIPS proceedings page. Retrieval notes are archived. The registry records **mechanism overlaps**, not algorithm-level non-equivalence proofs or an exhaustive search through the retrieval date.

| Source | Primary URL | Why it constrains novelty |
|---|---|---|
| Kingma & Ba, Adam | https://arxiv.org/abs/1412.6980 | Momentum and second-moment adaptive scaling are established. |
| Baydin et al., Hypergradient Descent | https://arxiv.org/abs/1703.04782 | Online gradient-based learning-rate adaptation is established. |
| Zhang & Mitliagkas, YellowFin | https://arxiv.org/abs/1706.03471 | Joint momentum/rate tuning and negative feedback are established. |
| Brock et al., NFNets/AGC | https://arxiv.org/abs/2102.06171 | Adaptive gradient clipping is established; the formulas must be compared. |
| Lugosi & Mendelson, robust-estimation survey | https://arxiv.org/abs/1906.04280 | Median-of-means and robust estimation are established. |
| Rahimi & Recht, Random Features | https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html | Random Fourier features are established. |

A different composition of these components can be worth investigating, but is not automatically a new technical contribution. This release does **not** autonomously search all current publications for every proposal. Its suggested search queries and exact controller/update dossier support further review. L3 PASS means no matching exclusion/template was found in its declared local index, not VERIFIED_NOVEL. The main demonstration starts without caller-curated full-implementation exclusions; the six papers provide component overlaps. Consequently every genuine generator proposal in that demonstration passed layer three. This fact must not be presented as evidence that they are all novel.

`register_prior_art(record, reference_id)` adds a caller-curated exclusion before the first round. Matching is exact against the supplied implementation, but the caller's mapping of that implementation to a cited paper is not authenticated or independently proven. The dossier explicitly says so. No manual PASS flag, arbitrary novelty score or unrecognized evidence field can override the five gates.

## Leakage boundary and integrity

The filter takes canonical algorithm metadata only. It has no dataset, label, measured-score, experiment-result or learning-feedback argument. It never instantiates or fits a learner. Known training/test/benchmark fields cannot be added to canonical records; strict import rejects them. Rejected and unsupported payload values are not retained in returned audits or persistent round bodies. Only their original digest and fixed outcome code remain. Such redacted entries can be replayed for internal no-pass consistency, but their discarded original contents cannot be recovered or reclassified from the artifact.

The ranking and witnesses use public declarations and fixed symbolic points only. An instrumented 100-record subprocess blocked any open under the existing scientific-validation path, dataset-provider imports, and learner construction/fitting. It completed with zero prohibited access attempts, zero learners and zero datasets loaded. Another test changed an unrelated sealed-results canary from score 1 to 0; the decisions stayed identical. These checks establish the tested data-flow boundary, not a universal zero-leakage theorem.

A caller can still encode earlier observations in a seed, component choice or purportedly public task description. The filter cannot detect undisclosed exposure, sanitize a tainted research history, protect against a malicious Python/OS owner, or make formerly seen data fresh. Those provenance obligations remain external. Public literature may contain published experiments; only mechanism notes enter this filter, not those benchmark numbers.

SQLite registers exact identities and uncertain response profiles together with the round audit in a single `BEGIN IMMEDIATE` transaction. Two concurrent instances cannot independently approve the same semantic candidate. Reopening retains duplicate history. The complete chain, policy/index binding, decisions and persistent registries are rederived before another round and on replay. Missing history, altered registries, forged scientific flags and rebound handoffs fail validation. Source files and the mechanism registry are pinned.

Hashes bind local records; they are not digital signatures or identity authentication. A fully privileged owner could rewrite source, data and an entire coherent journal. This is a trusted-local research tool, not a security boundary against its own administrator.

## Interfaces and execution

Runtime: Python 3.12.14 and NumPy 2.3.5 in the tested environment. The generator dependency must have SHA-256 `8ce3010e741603c9627c85f37944586c4cde2155eeb24660adcfd58ca5931b3f`. By default it is `../generator_only/hypothesis_generator.py`, or set `HYPOTHESIS_GENERATOR` / `--generator` to its location.

```bash
python -m unittest discover -s . -p 'test_*.py' -v
python hypothesis_filter.py demo a_new_filter_journal.sqlite
python hypothesis_filter.py inspect demonstration.sqlite
```

The demo creates a new journal and refuses to overwrite an existing file. Reopening an existing compatible journal uses `create=False`. Policy and source changes cannot silently reuse it. Inspection replays audits without fitting anything.

```python
from hypothesis_filter import FiveLayerFilter

screen = FiveLayerFilter("research.sqlite", generator_path="/path/to/hypothesis_generator.py")
# records is the existing generator's list of full canonical .to_dict() records.
audit = screen.evaluate_batch(records)
assert screen.replay_audit(audit)
selected = [d for d in audit["decisions"] if d["eligible_for_math"]]
# The complete audited batch can be checked once before bulk downstream routing.
for decision in selected:
    canonical_record = decision["record"]
    # Submit only this immutable/copy-checked record to the mathematical stage.
```

`handoff(candidate_id)` exports one candidate with a committed candidate-bound receipt. `verify_handoff(packet)` replays its journal and rejects mutated candidates or forged receipts. This convenience call replays history; for large bulk runs, verify the complete batch once rather than repeatedly replaying it for each record.

`forward_to_math(packet, existing_math_engine, runtime=..., math_policy=...)` calls only the existing public `assess()` and `verify_receipt()` for that selected candidate. It does not run predecessor hypotheses or write a private issuance ledger. The existing math gate retains origin authority: unissued candidates stay pending/rejected under its own rules. **Automatic selective origin admission is not provided by that legacy engine.** A compatible public intake or an already session-issued origin is needed before it can produce a science-ready admission. The filter does not bypass that limitation or silently claim a complete autonomous discovery pipeline.

The live integration test prevents `math_engine.run()` from being called and verifies the actual selected-record mathematical receipt. Mathematical subclaim verification is not a proof that the whole learner converges, generalizes or is novel.


## Demonstration results

The final source processed 1,700 genuine generator hypotheses without fitting a learner or loading a dataset. The first three rounds used one classification generator with seed 42, public task features=8, classes=3 and goals outliers/ill_conditioned, continuing its pure sequence across rounds. A fourth regression control used seed 99, features=8, classes=1 and the same goals. Identity history persisted across all four rounds.

| Round | Generated | L1 coherence | L2 uniqueness | L3 scoped prior art | L4 potential | L5 ready for math |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 100 | 100 | 100 | 82 | 68 |
| 2 | 500 | 500 | 500 | 500 | 435 | 309 |
| 3 | 1000 | 1000 | 1000 | 1000 | 869 | 546 |
| 4 | 100 | 100 | 100 | 100 | 87 | 76 |

Final survivors total **999 research candidates**: 68 + 309 + 546 classification and 76 regression. None is certified novel or superior. The initial three layers passed every genuine proposal in these particular rounds: their generator already avoided observed structural duplicates, and no caller-curated full-implementation exclusions were loaded. The demonstration's reduction therefore came from feedback-witness/readiness rules. Deliberate equivalence, coefficient-variant and registered-exclusion controls exercise the other rejection paths. This distinction is material; survivor counts are not novelty precision or algorithm-quality measurements.

All six update mechanisms remain represented among survivors in every round. The classification survivors include cross-entropy, Brier and focal losses; regression survivors include squared, Huber, pseudo-Huber and Cauchy. There is no blanket convex-loss or RMS-preconditioner admission rule in this screen.

| Round | Filter seconds, including prior-history validation | Generation seconds |
|---|---:|---:|
| 1 | 0.286396 | 0.037498 |
| 2 | 2.411450 | 0.183638 |
| 3 | 8.721261 | 0.398741 |
| 4 | 8.883303 | not separately timed |

The 1,000-proposal third round took about 8.72 seconds including validation of previous journal history. Timings are machine-dependent operational measurements, not proof of superiority to other systems. Replaying the entire journal is deliberately stronger and more expensive than checking a hash. Use complete-batch replay once for bulk routing.

Every audit rederived the same stored decision digest. The final run was regenerated after the supplementary equivalence fix using exactly the same generation metadata and research policy; it produced the same survivor counts as the initial run. Digests changed because normalization/profile records and the pinned source changed, not because any scientific outcomes were consulted.

| Round | Final decision digest |
|---|---|
| 1 | `7824ba3c48ed4f070113c4d700c4180f8c5547e375562848d79ae57d4ebb2dcf` |
| 2 | `5bae01b87a4f4dae4b77c2dd16e41f9b1d0ac016c61055d01496facabb39b123` |
| 3 | `d466ea670eccba5ccb20e2c498a610376b358da2360d37276b1337037f3844b8` |
| 4 | `08df3dd413bdd64996220198abebdd08cb73453ecda3a3b0e539b6d4fb87977d` |

### Review and bug disclosure

A separate reviewer wrote and sealed 40 adversarial scenarios before reading the implementation. The corpus SHA-256 is `ded9f190b3b513cdaa817563cec0287755df44bd74435bbd16f96dce0be06f7d`. Before unsealing, source review found four bugs/gaps: min associativity aliases, retained rejected payloads, unwanted math-prefix processing, and a contradictory prior-art claim. Each was reproduced and corrected before the first freeze.

The first frozen source was `e04d600a…`. The first harness execution reported 38 PASS, 1 SOURCE_REVIEW_ONLY and one harness mismatch: it compared a tuple in generator output to a JSON list using ordinary Python equality. Canonical JSON comparison corrected that harness error without changing the implementation, giving **39 PASS + 1 SOURCE_REVIEW_ONLY**. Both original and corrected harness/results are preserved.

A later supplementary check, outside the sealed 40, then found a genuine implementation miss: `1-min(a,b)` and `max(1-a,1-b)` could both be forwarded as distinct candidates. That failure is preserved separately. The final source pushes complements through the lattice and adds conservative unresolved-profile holds. The unchanged 40 scenarios were rerun on the final source as **exposed regression**, again 39 PASS + 1 SOURCE_REVIEW_ONLY. Five supplementary regressions passed, including nested De Morgan, persistent conservative profile holds, cross-blueprint separation, selected-only math routing and concurrent profile collisions. The final public suite passes **28/28** tests.

The source-review-only case uses a nonpublic SQLite abort trigger to inspect transaction behavior. It is not presented as a tested public fault-injection API. Several corpus PASS cases establish only safe refusal of unsupported schemas, rather than broad domain/evidence support. The review is separate-agent work on the same system, not independent institutional replication, and no scientific data or learner fitting was used.

### Upstream mathematical bridge

The instrumented real-math integration submits one selected canonical record through the original public math engine and replays its receipt, while prohibiting any generation-prefix run. The archived example verified 33/33 conditional subclaims and returned PENDING solely for UNATTESTED_ORIGIN. The original engine may verify conditional subclaims yet remain pending/rejected because its origin authority has not issued that record. This is an explicitly preserved constraint. It must not be described as successful automatic scientific admission or as a proved new algorithm.

## Reproduction and package layout

The four delivered files are `hypothesis_filter.py`, `test_hypothesis_filter.py`, `REPORT.md` and `demonstration.sqlite`. Runtime requirements are Python and NumPy; optional live mathematical integration uses the existing pinned math engine/verifier. Byte-exact dependency snapshots are archived in the SQLite review table so the package can be tested in a clean directory without workspace source imports.

The artifact stores candidate specifications, deterministic symbolic probes/witnesses, source/index hashes, all four audit rounds, review cases/results/source snapshots and logs. It stores no real dataset or experimental accuracy table. The archived adversarial corpus deliberately contains measurement-shaped dummy fields for refusal tests; these are synthetic canaries, not real benchmark evidence. The main `rounds` table is the committed filter evidence; `review` contains supporting reproducibility records and never grants scientific PASS.

To materialize the three dependency snapshots into a fresh temporary directory:

```python
import hashlib, json, sqlite3, tempfile
from pathlib import Path
root = Path(tempfile.mkdtemp(prefix="hypothesis_dependencies_"))
entries = [
    ("pinned_generator_source", "generator_only/hypothesis_generator.py",
     "8ce3010e741603c9627c85f37944586c4cde2155eeb24660adcfd58ca5931b3f"),
    ("pinned_math_filter_source", "math_update_gate/math_filters.py",
     "6d3dd542732e68008eac4ed88bf133976493231745842e5e4a3916aa85b3f644"),
    ("pinned_verifier_source", "math_verifier/math_checker.py",
     "37dc50931bfe2512875558e5660de692680c8b232333a98885503da9697205fc"),
]
with sqlite3.connect("demonstration.sqlite") as db:
    for key, relative, expected in entries:
        item = json.loads(db.execute("SELECT payload FROM review WHERE name=?", (key,)).fetchone()[0])
        raw = item["text"].encode("utf-8")
        assert hashlib.sha256(raw).hexdigest() == expected
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
print(root / "generator_only/hypothesis_generator.py")
```

Set `HYPOTHESIS_GENERATOR` to the printed path. `python -I test_hypothesis_filter.py -v` then tests from a clean extraction without relying on the workspace import path. `python -I hypothesis_filter.py inspect demonstration.sqlite` rederives the committed four rounds. To reproduce the primary demonstration, run `demo` against a new path; to recreate round four, generate 100 regression records from the documented seed/task and call `evaluate_batch` on the reopened new journal. Decision digests should agree under the pinned source and dependency version. Runtime timings need not.

Original dependency files and all previous generator, mathematical and scientific engines remain unchanged. This delivery builds the five-layer research screen. Full novelty investigation, universal ML-domain generation, selective upstream origin admission, and proof of scientific value are not silently claimed complete.


## Final package verification

A clean extraction restored the three pinned dependency snapshots to a temporary directory and set `HYPOTHESIS_GENERATOR` there. `python -I test_hypothesis_filter.py -v` passed **28/28 tests with no skips in 0.978 seconds**, including the real existing mathematical-engine receipt test. Read-only inspection successfully rederived all four committed rounds.

A new process then regenerated **all 1,700 proposals** from the documented seeds/public metadata and reran the filter into a different journal. All four decision digests and layer counts matched the supplied artifact exactly. The reproduction relied on no original workspace source/import path. It is same-environment reproducibility, not independent scientific validation or a benchmark of candidate quality. Reproduction source, result and logs are archived in the SQLite review table. The final SQLite integrity check returned `ok`.

| Delivered file | SHA-256 |
|---|---|
| hypothesis_filter.py | `f3766f437d5c2e6cf42d91cbc12cd71ef36348be3ac8d2237b51d1b8a39acd92` |
| test_hypothesis_filter.py | `9491c32006a45cf8e903f79b20c6e7ce749e41a5ed9889d0676bf3f78182f678` |
| demonstration.sqlite | `5a0bcd8517b0155ea54e244f374fcc1e2b53f244d6a9c1060d535568d1082daf` |

The report is excluded from its own hash table. The archive contains exactly the four delivered files. File hashes bind bytes for comparison; they do not authenticate an author or certify novel science.
