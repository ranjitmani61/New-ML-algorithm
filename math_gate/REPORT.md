# Mathematical admission: integrity and substantive filters

Scope: the two missing stages, integrated with the unchanged generator and
mathematical verifier. No scientific validation engine, data fitting, novelty
claim or acceptance-gate upgrade. Four delivery files: implementation, tests
and runners, this report, independent challenge corpus.

Demonstrated on 2026-10-31 with Python 3.12.14 and NumPy 2.3.5. On the same
800 generated candidates, the previous subclaim-only gate would pass 800; this
layer admitted 367, rejected 432 and left 1 pending. All 367 admitted candidates
completed receipt verification and actual local handoff. These are policy
admissions, not discoveries of proven new or superior ML algorithms.

Chosen default admission policy: convex per-example/mean data-fit objective in
trainable scores/weights for fixed features, plus exact existential influence
of two training-feedback signals on a used update coefficient, with a minimum
dimensionless effect change of 1/64. Time is not training feedback. Nonconvex
candidates rejected by this profile are not declared universally invalid ML
research. The policy can be explicitly configured and is bound to issuance and
receipts. No descent, convergence, trajectory-reachability or predictive theorem
is implied by admission.

Integrity precedes mathematical work: bounded strict JSON, canonical complete
generator roundtrip (including equations/gates/resources/provenance), pinned
source versions, complete record hash, runtime/policy binding, and a private
session ledger of actual generator emissions with parent membership. External
canonical records without session issuance remain PENDING. Hashes establish
consistency, not public-key authentication or an independent origin signature.
The process/filesystem and the pinned executor's mathematical binding are part
of the trusted base; source hashes do not prove Python execution semantics.

Replay verifies every emitted mathematical certificate and reconstructs the
complete admission decision, including exact rational feedback witnesses.
Handoff and model construction require a READY receipt in the issuing session.
The model is constructed with precisely the admitted runtime and remains
unfitted. Existing scientific gates remain UNVERIFIED.

Objective proofs use fixed, explicitly implemented analytic schemes: nonnegative
curvature for squared loss; continuous nondecreasing piecewise Huber gradient;
positive pseudo-Huber curvature through the square-root relation; and softmax
weighted-variance PSD via a dimension-independent merge identity. Counterexamples
use Cauchy residual 2, Brier logit probability 1/4, and focal probability 1/32.
Focal's sign uses log(2)>2/3, proved by integrating the strict tangent bound
1/t >= 4/3-4t/9 on [1,2]. Algebra, derivatives, inequalities and boundary facts
are certificate-checked; calculus/convexity inference schemes are reviewed code,
not mechanically verified Lean/Coq derivations.

The separate reviewer authored and sealed 145 challenges before inspecting the
new implementation. They were opened only after the initial freeze below.
Predeclared corpus SHA256:
`c336dc207e08850994c8e70082f5f87619ca881ba37eb326b30867b5de422f74`.

Initial core freeze before opening that corpus:
`34ab1ee0c4428c8d3e0179844499e53a02f2f6a0a7a19241b6e7bf5a67099767`.
All 23 public tests passed before this freeze. Any implementation fixes after
exposure will be disclosed; the corpus then becomes regression material.

Initial frozen-core independent result: 128/145 matching challenges, three
genuine obligation-completeness failures, 14 outside the declared API contract.
All 71 record-integrity mutations, 14 feedback cases and 21 theory cases matched.
The failures inject omission, duplication or certified-tautology substitution
into the base checker's returned obligation list. Initial code replayed supplied
proofs but failed to bind the complete required list to the candidate. A
post-exposure correction was independently regression-tested; this corpus is
not untouched evaluation of the corrected implementation.

## Run and integrate

The archive contains exactly four new files:

| File | Role |
|---|---|
| `math_filters.py` | All production integrity, filters, pipeline, receipt and handoff logic |
| `test_math_filters.py` | 26 public test methods plus reproducible demonstration/evaluation runners |
| `independent_cases.json` | All 145 unchanged predeclared challenges, including unsupported cases |
| `REPORT.md` | Contract, mathematical derivations, measurements and limitations |

The existing generator and verifier are dependencies and remain unchanged;
they are deliberately not duplicated in this four-file delivery. By default,
place this directory next to `generator_only/` and `math_verifier/`. For other
layouts, pass explicit production paths and set test-runner paths as shown.
Use the reviewed source revisions listed under Checksums. Unexpected source
revisions are rejected before import; source updates need a new review/pin.

```bash
# On Python 3.12; the generator's runtime dependency:
python -m pip install numpy==2.3.5

# Default adjacent layout:
python math_filters.py run --kind regression --features 8 --count 100 --seed 42
python math_filters.py run --kind classification --classes 3 --features 8 --count 100 --output audit.json
python -m unittest test_math_filters -v
python test_math_filters.py --external
python test_math_filters.py --demonstrate --output demonstration.json

# Alternative layout:
python math_filters.py --generator /path/hypothesis_generator.py --verifier /path/math_checker.py run --count 20
MATH_GATE_GENERATOR=/path/hypothesis_generator.py MATH_GATE_VERIFIER=/path/math_checker.py python -m unittest test_math_filters -v
```

`assess candidate.json` analyzes an external full generator record. A canonical
record can pass mathematics while staying PENDING because its origin is not
attested by this engine's generation session. `run` generates through the pinned
generator, registers actual emissions, then performs the same assessment.
It requires no dataset, experiment scores or manually changed gate statuses.

```python
from math_filters import MathFilterEngine

engine = MathFilterEngine(
    generator_path='/path/hypothesis_generator.py',
    verifier_path='/path/math_checker.py',
)
batch = engine.run({'kind': 'regression', 'features': 8}, count=100, seed=42)
for item in batch['results']:
    if item['status'] == 'READY_FOR_SCIENTIFIC_VALIDATION':
        packet = engine.handoff(item['candidate'], item['audit'])
        # Pass packet to a future scientific validator through a trusted interface.
        # No scientific validator is implemented or invoked by this package.
```

`build_model(candidate, audit)` constructs the unchanged generator's reference
executor only after successful handoff validation. It uses the admitted steps,
learning rate and seed, and returns an **unfitted** model. No fitting or training
is performed by the new engine or demonstration runner.

## Exact acceptance meaning

The route is generator -> complete record integrity -> prescribed base math
obligations -> objective/feedback filters -> verified handoff. Both new filters
are necessary under the default profile; passing easy arithmetic alone cannot
produce READY. A rejection lists concrete policy reasons and mathematical
witnesses. Unresolved proof or influence search stays PENDING.

Default policy:

```json
{"require_convex_objective":true,"min_feedback_signals":2,"min_influence":"1/64"}
```

This is a configurable research-admission profile, not a universal requirement
for every useful ML algorithm. Setting `require_convex_objective` to false is an
explicit different profile: the record still reports a NONCONVEX finding, and
all other obligations and origin checks remain required. A nonconvex candidate
is not declared contradictory or scientifically worthless. The current per-example
loss screen also does not conclude that a particular unknown dataset's mean
loss must be nonconvex: the policy requires convexity of the loss family before
data are supplied. Convex per-example losses imply convex empirical means for
fixed features; the converse is not assumed.

Runtime defaults are `steps=160`, exact `learning_rate="2/25"`, execution
`seed=0`. Steps must be strict integers in 1..20000 and seeds unsigned 32-bit
integers. Rates accept exact rational/decimal strings or integers in (0,1];
binary floats and Boolean numbers are rejected by this API. `build_model`
converts the admitted rational rate to the reference executor's floating value.
No finite-precision correctness or overflow theorem follows from real arithmetic.

Policy and runtime are bound to the issuance key, complete record hash and
receipt. Rehashing a weaker policy or changed runtime does not create a matching
issuance and cannot reuse READY. The scientific validator can choose new runtime
settings only through an explicit new, correctly issued admission contract.

## Substantive objective proofs

Fixed preprocessing/features are held constant when reasoning about weights.
All three emitted representations (linear, diagonal quadratic and Fourier)
are affine in trainable weights, even when nonlinear in raw inputs. Approved
convex losses therefore compose with the fixed representation and remain convex.
This says nothing about whether the representation predicts well.

| Loss | Mathematical result under this profile |
|---|---|
| Squared | second derivative 1 > 0 |
| Huber | continuous, globally nondecreasing piecewise gradient; C1 knots permitted |
| Pseudo-Huber | second derivative `(1+r*r)^(-3/2) > 0` |
| Cross-entropy | logit Hessian `diag(p)-p*p^T` is PSD |
| Cauchy | negative residual curvature at r=2, exactly -3/25 |
| Brier | negative logit curvature at target probability 1/4 |
| Focal, gamma=2 | negative logit curvature at target probability 1/32 |

For pseudo-Huber, set `s=sqrt(1+r*r)>0`. The checked algebra is
`(s-r*(r/s))/(s*s) - 1/(s*s*s) = (s*s-r*r-1)/(s*s*s)`.
The square-root defining relation makes the right side zero, and the reduced
Hessian is positive. The original loss-gradient identity and denominator domain
are separately certified.

For cross-entropy, the second directional derivative is a weighted variance:
`sum_i p_i*v_i^2 - (sum_i p_i*v_i)^2`, with `p_i>0` and `sum_i p_i=1`.
The dimension-independent induction uses the checked merge identity
`a*x*x+b*y*y-(a*x+b*y)^2/(a+b) = a*b*(x-y)^2/(a+b) >= 0` for a,b>0.
It covers all finite admitted class counts without pretending a small matrix
enumeration proves arbitrary dimension.

For Brier with C>=2, hold all non-target logits equal and vary the target
logit. With target probability p, the restricted loss is
`C/(2*(C-1))*(1-p)^2` and `dp/dz=p*(1-p)`. Its logit curvature is
`C/(C-1)*p*(1-p)^2*(3*p-1)`. At p=1/4 this is exactly
`-9*C/(256*(C-1))`; for C=3 it is -27/512. The probability point is attainable
at target logit `log((C-1)/3)` and other logits zero. Brier's convexity in
probabilities is not substituted for convexity in logits/weights.

For focal loss `-(1-p)^2*log(p)`, the corresponding curvature is
`p*(1-p)^2*((2-6*p)*log(p)+5*(1-p))`. At p=1/32, use `log(p)=-5*log(2)`.
The exact bound `log(2)>2/3` follows because
`1/t-(4/3-4*t/9)=(2*t-3)^2/(9*t)>=0` on [1,2], with a strict nonzero point;
integrating the tangent gives 2/3. Thus the curvature is bounded above by
the negative rational `-110515/3145728`. The code checks derivative identities,
the rational inequality, primitive and endpoint facts. The fundamental theorem
of calculus, log identities, convexity criteria and induction composition are
fixed reviewed inference schemes. They are not Lean/Coq kernel derivations.

These mathematical facts concern the loss, not the complete optimizer field.
Clipping, median aggregation, momentum, adaptive penalties and RMS scaling
remain additional mechanisms. This package asserts no global descent,
convergence, unbiased-gradient, robustness or empirical superiority theorem.

## Exact feedback filter

The new filter evaluates the actual controller AST using rational arithmetic,
independent of the generator's rounded construction probes. It searches the
fixed public grid `{0,1/2,1}^6`, comparing endpoint and interior pairs that
change exactly one training signal. Positive witnesses bind the complete
controller, unchanged other inputs and a used mechanism coefficient.

The coefficient is the clipping threshold multiplier for clip_memory/clip_step,
fast momentum for memory_step, fast/slow mix weight for dual_memory, or penalty
multiplier for penalty_step/variance_penalty. These are dimensionless effects;
the penalty multiplier's 1/64 threshold is not a 1/64 absolute lambda change.
First-only coupling still has a real first branch, so is not automatically
rejected. Requiring a second coupled branch is outside this profile.

A controller proved constant is rejected. Fewer than the requested number of
possible training inputs is rejected. For the remaining cases, sufficient exact
witnesses pass; search failure is PENDING, never a proof of no influence.
Tests cover nonmonotone functions whose endpoint values agree, cancellation,
inactive labels and receipt-witness alteration.

The witnesses are existential over an independent unit box. They do not prove
those signal combinations occur on real training trajectories or that adaptation
helps. The original conditional range assumptions and missing convergence
obligations remain visible in the supplied base audit and prior verifier report.

## Integrity, completeness and provenance

The full canonical import regenerates **every** generator field: algorithm,
task, identifiers, statement, equations, ablations, scientific gates, resources
and provenance. Forged PASS flags or equations are rejected before math checks.
JSON duplicates, nonfinite numbers, unknown fields, excessive depth/size and
expensive rational exponents are rejected. Allowed scientific gates remain
exactly the generator's UNVERIFIED values; no record is rewritten to PASS.

The corrected core separately compiles the prescribed base obligations through
the pinned declarative generator-audit compiler with a declaration-only collector.
It compares the exact ordered obligation names and full claim hashes before
checking verdicts/certificates. Missing, duplicate or substituted claims yield
PENDING `BASE_OBLIGATION_SET_MISMATCH`. A valid 1==1 proof cannot stand in for a
required controller bound. Universal obligations cannot use finite or vacuous
proof statuses. Each exact obligation must be VERIFIED and replay successfully.

The layer then adds the objective/feedback requirements. Receipt verification
replays certificates and reconstructs the decision, including bounded feedback
search; the entire receipt is not claimed to be search-free. Underlying exact
certificate replay uses the existing verifier's search-free replay API.

Session issuance is registered only from actual generator emissions. Every
mutate/crossover parent must already have been emitted in that generation run.
Full record, task/provenance, runtime and policy are bound to the session entry.
An internally consistent fabricated seed or random parent ID is not mistaken
for proven history. A separately loaded record can be analyzed but cannot
handoff without actual session issuance.

The ledger is local, in memory and capped without silent ancestry eviction.
Receipts are consistency evidence, not signed public credentials. A different
process cannot infer authenticated origin from JSON alone; use a trusted
issuing interface for downstream handoff. Mathematical subcertificates remain
individually replayable against their full claims. Self-source identity is
captured at module load, so stale loaded code cannot attest to a later edited
file. Trusted-process monkeypatching, compromised dependencies and correctness
of the Python-to-real-mathematics binding are outside record-data integrity.

## Demonstration results

Four seeds (7,19,42,2026), two task kinds, 100 hypotheses per combination,
eight features, three classification classes. Default policy/runtime above.
No training, evaluation dataset, predictive score or generator reward feedback
was consumed. Each admitted candidate attempted actual receipt-checked handoff;
every non-admitted candidate was tested to refuse handoff. One admitted model
per row was constructed and confirmed unfitted.

| Task | Seed | Generated | Admitted/handed off | Rejected | Pending |
|---|---:|---:|---:|---:|---:|
| Regression | 7 | 100 | 48 | 51 | 1 |
| Regression | 19 | 100 | 65 | 35 | 0 |
| Regression | 42 | 100 | 75 | 25 | 0 |
| Regression | 2026 | 100 | 63 | 37 | 0 |
| Classification | 7 | 100 | 22 | 78 | 0 |
| Classification | 19 | 100 | 30 | 70 | 0 |
| Classification | 42 | 100 | 46 | 54 | 0 |
| Classification | 2026 | 100 | 18 | 82 | 0 |
| Total | | 800 | 367 | 432 | 1 |

There were 373 negative-curvature policy findings: 106 Cauchy, 157 Brier and
110 focal candidates. There were 109 insufficient-training-feedback reasons
among rejections; reasons overlap. Unresolved feedback also appeared alongside
two already-rejected candidates and in the one PENDING candidate. The previous
subclaim-only rule would accept all 800 under its own conditional arithmetic
contract. This measures stronger policy discrimination, not ML performance.

14,845 mathematical facts were VERIFIED, including facts proving negative
curvature for rejected candidates. This count contains repeated library facts
and claim-cache reuse; it is not a count of new theorems. No whole algorithm
was promoted to scientific validity. All ten scientific gates stayed UNVERIFIED.

Admission median was 31.72 ms and p95 60.15 ms in the recorded run. Combined
generation/admission for 800 took 27.68 seconds across the eight rows; additional
handoff revalidation was outside that interval. Library facts use claim-specific
cache reuse. Measurements are machine/load-dependent; some verification jobs
shared the machine during the run. No speed-leadership claim is made.

Deterministic candidate/decision/reason digest, excluding timing:
`02836c7896e01cbee3bad3c97b5a5db0087cf7206db28af08dc438127d1112ff`.
The demonstration runner prints every row, all counts, environment and digest.

## Validation history and independent review

All **26 new tests, 22 generator tests and 33 verifier tests pass** (81 methods).
The four-file archive was extracted into a fresh temporary directory. All 26
new tests passed there in an isolated Python `-I` subprocess with explicit
pinned dependency paths and no PYTHONPATH. The extracted autonomous CLI generated
20 candidates: 13 admitted, 7 rejected, scientific status NOT_EVALUATED.
Tests cover complete record mutation, unissued origin, parent forgery,
nonconvexity in the correct coordinates, Huber knots, up to 256 classes,
nonmonotone and constant feedback, malformed receipts, policy/runtime/source
tampering, rehashed attacks, complete obligation binding, and actual autonomous
handoff/model construction. Separate reviewing-agent code/theory inspection
found no remaining blockers within this contract after the corrections.

The original 145-case corpus was prepared before implementation inspection.
It has not been edited. Its scope and outcome history are retained:

| Challenge category | Cases | Corrected applicable matches |
|---|---:|---:|
| Full record integrity | 71 | 71 |
| Behavioral feedback policy | 14 | 14 |
| Objective theory policy | 21 | 21 |
| Required second-branch coupling | 12 | outside this policy |
| Runtime contract | 14 | 12, plus 2 out-of-contract |
| JSON wire integrity | 4 | 4 |
| Verifier fault injection | 9 | 9 |
| Total | 145 | 131 matched; 14 out-of-contract; 0 mismatches |

The 14 exclusions are all disclosed: twelve require a mandatory second branch
not in this profile; one expects decimal-string rates to be rejected although
this API explicitly requires exact strings; one expects a seed above uint32 to
be valid. Finite floating-rate corpus examples are mapped by decimal spelling
to the public exact-string API. Unissued external records are assessed for
mathematical eligibility and remain UNATTESTED; they are not secretly added to
the ledger. Fault tests use real session-issued candidates and actual handoff.
The complete adapter is delivered in `test_math_filters.py`.

Initial result was 128 matched, 3 genuine failures, 14 exclusions. The same
reviewer independently reproduced all three completeness failures, then verified
the fix against the unchanged corpus: 131 applicable matches, no mismatches.
Public regression tests were added before the fixes. Further review found
malformed-receipt API errors; guards and a stale-loaded-source test were added.
This is **post-correction regression**, not untouched final held-out validation.

Initial core SHA256:
`34ab1ee0c4428c8d3e0179844499e53a02f2f6a0a7a19241b6e7bf5a67099767`.
Final reviewed core SHA256:
`7aee339f092dfc64823979cd2ef74d0e6d48f80d810992668d76a053feed66b0`.
Initial independent-results artifact SHA256:
`02e0197efe6dbdcaefce9bea66eb7e91931788a77f1a92a74745a3c603e8e7d2`.
Corrected independent-results artifact SHA256:
`202f6d995d457a5932894ab98726db70da8d92dedd483a96fb7f9dc4fa561bf6`.
Those workspace result artifacts are summarized here rather than adding more
delivery files; the supplied runner reproduces the corrected case-level results.
Review and reruns used the same managed machine, not an independent institution.

## Bounds and remaining work

Records are bounded to 300,000 JSON bytes, depth 24 and 20,000 structural nodes.
Numeric exponent/bit checks precede expensive exact conversion. Controller
grammar remains the generator's depth/node-bounded library. The feedback search
uses 729 exact points and fixed coordinate comparisons. The existing verifier
enforces its own polynomial, variable, operation and cooperative-time budgets.
Each run allows 1..10000 candidates, and the ledger refuses capacity above 20000
entries without eviction. A run retains its result batch, so memory grows with
batch size; this implementation does not claim unbounded streaming scalability
or OS-enforced memory/time isolation.

The layer neither accesses benchmark outcomes nor learns from the challenge
labels. Test cases were exposed for the disclosed corrections, so no zero-leakage
held-out-accuracy claim is made. Data-free architectural separation and complete
record checks are demonstrated; false provenance supplied outside the trusted
session is not magically authenticated by hashing.

Only the current pinned generator grammar has these proof schemes. It does not
establish arbitrary mathematics, literature novelty, complete optimizer
convergence, finite-machine correctness, generalization, robustness or all ten
scientific requirements. Those remain downstream scientific-validation work.

## Checksums

| File | SHA256 |
|---|---|
| math_filters.py | `7aee339f092dfc64823979cd2ef74d0e6d48f80d810992668d76a053feed66b0` |
| test_math_filters.py | `4d7e87b7efcedfb3ef2f01cf4d62ddf3a920a01b4ef29ff6dfb30b9d111f1cd9` |
| independent_cases.json | `c336dc207e08850994c8e70082f5f87619ca881ba37eb326b30867b5de422f74` |
| unchanged generator dependency | `8ce3010e741603c9627c85f37944586c4cde2155eeb24660adcfd58ca5931b3f` |
| unchanged verifier dependency | `37dc50931bfe2512875558e5660de692680c8b232333a98885503da9697205fc` |
