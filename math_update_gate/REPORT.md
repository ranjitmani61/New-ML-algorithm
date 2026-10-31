# Update-rule mathematical filters, version 2

Four files: one implementation, tests/runners, report, independent challenge
corpus. Existing generator and exact verifier remain unchanged dependencies.
Scientific validation is not implemented or performed.

Default convexity requirement is now false. Objective convexity/nonconvexity is
reported, with its existing certificates. Admission additionally requires exact
adaptive update invariants and the declared frozen positive-mode stability
envelope, as well as preserved integrity, origin, feedback and proof-completeness
checks. Default curvature upper bound 50 is an explicit reference contract,
not a Hessian or smoothness bound inferred from an unknown training dataset.

The update invariants cover varying in-range controls and the actual recurrence:
from zero initialization m_j^2<=20*v_j, nonnegative v, positive RMS/blend
denominators, and normalized step norm<=eta<=learning_rate. Frozen Jury/Schur
certificates cover only fixed symmetric feedback, controls, preconditioner and
inactive norm cap; zero modes are marginal, negative modes are outside that
contract. They never imply full nonlinear or switching stability.

certify_transition supplies outward-rational optimizer-core calculations through
clipping, the true norm quantile, mean/MoM, intercept-exempt penalty, fast/slow
memory, RMS/blend and normalization. The true per-example gradients, non-time
controller signals and segment smoothness are supplied assumptions. It does not
authenticate their data derivation or prove state trajectory reachability.
Descent certificates target the mean objective plus the current frozen lambda
regularizer. Positive directional-change certificates do not assert the finite
nonconvex objective necessarily increases. Unknown signs remain INCONCLUSIVE.

At the initial freeze, 26 public test methods passed. A conformance test observes 132 steps of the pinned
executor on public toy values, covering all six mechanisms, both preconditioners
where compatible, all maps, both aggregators and both couplings. Floating
comparisons are diagnostics, never the sign proof. Other tests check exact
curvature-envelope boundaries, adaptive RMS energy, interval square roots,
runtime/controller sensitivity, nonconvex retention, update-direction reversals,
tampering, integrity and handoff. No scientific benchmark scores are measured.

The reviewer sealed 132 independent challenges before seeing this implementation.
Predeclared corpus SHA256:
`19383c00918c5067f384908c82e0b5277425812fbb0b2d45dda015b492effd79`.
Initial implementation freeze before opening the challenges:
`60cf52e5f55aaaef7dafa4d518021bd915625a61e4c2e5a824d7fbb98dca6c10`.
Any post-exposure fixes will be disclosed and subsequent runs called regression.

Initial independent run: 94/132 complete contract matches. All mathematical
transition classifications, all 36 envelope classifications and all tampering
outcomes matched; 36 cases additionally required the promised maximal rate
field that was absent, and 2 malformed-state cases raised KeyError instead of
the documented ValueError. Those API omissions were corrected after exposure,
with red/green public tests. The strict rate threshold now carries its own exact
boundary polynomial certificate. Subsequent challenge results are regression,
not untouched final held-out evaluation. Full penalized feedback is explicitly
named in the frozen-curvature assumption.

## Delivered outcome

Built and measured on 2026-10-31, Python 3.12.14, NumPy 2.3.5. On the same
800 generated candidates used for the prior admission demonstration, v2 admitted
and handed off **134**, rejected **661** under its declared profile, and left
**5 pending**. This includes **58 nonconvex candidates** rejected by v1 solely
under its convexity requirement. Conversely 291 v1-eligible candidates did not
pass v2. All seven supported losses have v2 admissions. More rejections are not
claimed to mean better algorithms; the two profiles ask different questions.

The new contribution is update-specific mathematical analysis and replay. It
does not establish ML novelty, predictive superiority or complete optimizer
convergence. The default frozen-reference envelope is a stated research screen,
not a proved bound on an unknown dataset's full adaptive dynamics.

## Four files and execution

| File | Purpose |
|---|---|
| `math_filters.py` | All production integrity, admission, update invariants, rate/envelope analysis, transition certification and replay |
| `test_math_filters.py` | 28 test methods, executor-conformance and property checks, corpus/demonstration runners |
| `independent_cases.json` | All 132 unchanged externally prepared challenges |
| `REPORT.md` | Contracts, proof derivations, measurements, limitations and history |

The existing generator and verifier are unchanged dependencies. They are not
duplicated in this four-file delivery. Default layout places this directory
next to `generator_only/` and `math_verifier/`; explicit paths are supported.
The v1 gate is needed only for the optional paired demonstration, not production.
Unexpected generator/verifier hashes fail before import. Use the reviewed
revisions in Checksums; an update to those sources requires reviewed new pins.

```bash
# Tested environment: Python 3.12 and NumPy 2.3.5.
python -m pip install numpy==2.3.5
python math_filters.py run --kind regression --features 8 --count 100 --seed 42
python math_filters.py run --kind classification --classes 3 --features 8 --count 100
python -m unittest test_math_filters -v
python test_math_filters.py --external
python test_math_filters.py --demonstrate --output demonstration.json

# Non-default dependency layout:
python math_filters.py --generator /path/hypothesis_generator.py --verifier /path/math_checker.py run --count 20
MATH_GATE_GENERATOR=/path/hypothesis_generator.py MATH_GATE_VERIFIER=/path/math_checker.py python -m unittest test_math_filters -v
```

The same `MathFilterEngine.run`, `assess`, `verify_receipt`, `handoff` and
`build_model` APIs are retained. `run` registers actual generator emissions and
parent lineage. An external canonical record can pass mathematical conditions
while staying PENDING because it lacks session issuance. Handoff is local to
the issuing engine; hashes are consistency checks, not public authentication.
Receipts bind the complete record, source versions, runtime and policy. Their
verification reconstructs all required claims and decisions and replays every
mathematical subcertificate. No scientific gate is rewritten.

```python
from math_filters import MathFilterEngine

engine = MathFilterEngine('/path/hypothesis_generator.py', '/path/math_checker.py')
batch = engine.run({'kind':'regression', 'features':8}, count=100, seed=42)
for item in batch['results']:
    if item['status'] == 'READY_FOR_SCIENTIFIC_VALIDATION':
        packet = engine.handoff(item['candidate'], item['audit'])
        assert packet['scientific_status'] == 'NOT_EVALUATED'
        # A future scientific validator can consume this trusted handoff.
```

## Default admission profile

```json
{
  "require_convex_objective": false,
  "min_feedback_signals": 2,
  "min_influence": "1/64",
  "curvature_upper": "50",
  "require_stable_envelope": true
}
```

Convexity is diagnostic by default. Cauchy, Brier and focal retain their exact
nonconvexity findings and can pass the update checks. The original effective
feedback requirement remains explicit: two training signals need rational
influence witnesses on an actually used coefficient over the signal box. Time
alone is not counted as training feedback. This does not prove trajectory
reachability or that adaptation is beneficial.

The new default requirement is a **frozen positive-mode reference envelope**,
with full penalized aggregate feedback curvature h in (0,50]. This includes
the derivative of the frozen current lambda penalty on non-intercept weights.
It is not a loss Hessian substituted for a clipping/MoM/controller Jacobian.
The derivative is assumed fixed and symmetric; the preconditioner, controls,
regularizer and step coefficients are frozen, and the norm cap is locally
inactive. All such assumptions are carried in the audit. The caller can change
the envelope or disable that requirement only through an explicit policy bound
to a new issuance. Disabling it does not manufacture a stability proof.

The filter proves a sufficient all-control interval result when possible. It
reports FAIL only with an exact controller input and schedule-consistent time
witness violating the declared reference contract. An unsuccessful conservative
interval proof without such a witness stays PENDING. A FAIL is not asserted to
be a divergent trajectory of the complete adaptive executor.

## Adaptive invariants for the actual recurrence

Let the coordinate gradient supplied to the optimizer be g. The code uses
`f_next=b*f+(1-b)*g` and `v_next=19*v/20+g*g/20`, with 0<=b<=19/20.
Starting from f=v=0, the invariant is `f*f<=20*v`. Its induction uses the
checked exact identity

`b*f*f+(1-b)*g*g-(b*f+(1-b)*g)^2 = b*(1-b)*(f-g)^2 >= 0`.

Given the previous invariant, the remaining slack is

`20*v_next-(20*b*v+(1-b)*g*g) = (19-20*b)*v+b*g*g >= 0`.

This holds for changing momentum coefficients and arbitrary finite gradients;
it does not require a convex objective or an unbiased gradient estimator.
The slow dual-memory branch b=19/20 has the same invariant, and its convex mix
with the fast branch preserves it by the analogous square identity. Nonnegative
variance is also inductive. For ordinary RMS, `D=sqrt(v)+1e-6`, so every
pre-normalization direction coordinate satisfies `direction_j^2<=20`.
The variance-penalty blend has `D=(1-q2)+q2*(sqrt(v)+1e-6)>=1e-6`.
No uniform ordinary-RMS direction bound is falsely reused for this blend.

For all mechanisms, `delta=-eta*d/max(1,||d||)` gives `||delta||<=eta`.
The actual mechanism multipliers lie in (0,1], and the schedule gives
`eta<=learning_rate`. These are real-arithmetic finite-state bounds, not a
convergence argument. They also do not certify finite-precision overflow safety.
Every stated algebraic identity/inequality has a claim-bound subcertificate;
the initialization, induction and composition inference are reviewed code.

## Frozen-mode stability and rate limits

For one momentum state and fixed positive diagonal preconditioner D, a scalar
mode has effective k=eta*h/D and recurrence

`f_next=b*f+(1-b)*h*x; x_next=x-(eta/D)*f_next`.

The characteristic polynomial is
`z^2-(1+b-k*(1-b))*z+b`. With 0<=b<1 and h>0, strict Schur stability is
equivalent to `k*(1-b)<2*(1+b)`. For fixed positive diagonal D and genuinely
symmetric feedback, mode analysis applies through the congruence/similarity
`D^(-1/2)*H*D^(-1/2)`. It is not assumed for an arbitrary nonsymmetric Jacobian.

Dual memory uses its actual fast coefficient 1/2, slow coefficient 19/20 and
mix q. Its cubic coefficients are
`A=-49/20+k*(10-9*q)/20`, `B=77/40-k*(19-18*q)/40`, `C=-19/40`.
The remaining strict Jury constraint reduces to `k*(39-36*q)<234`; the other
conditions are positive for k>0 and q in [0,1]. This is checked separately,
rather than applying the single-memory formula to a three-state update.

The compiler preserves actual joint q/q2 relations for each mechanism and the
actual denominator floor: identity 1; ordinary RMS 1e-6; variance blend
`1-q2+q2*1e-6`. Thus `first_only` variance blending has floor 0.5000005,
which is not treated as ordinary RMS. Current-controller interval bounds are
themselves certified. After multiplying only by positive denominators, the
worst-case slack is a degree-at-most-two polynomial in q. Exact endpoint/vertex
minimization supplies a replayable interval minimum certificate.

`max_certifiable_rate` is the exact, uncapped, **strict** threshold for that
conservative controller interval and frozen envelope. Equality does not qualify.
Worst q is hi for clip_memory/clip_step/variance_penalty, and lo for
memory_step/dual_memory/penalty_step. A boundary polynomial with minimum exactly
zero certifies the rate boundary; its rate slope has the required strict sign.
This is not an optimal rate for actual trajectories or an empirical tuning
recommendation. The runtime API still caps rates at 1.

`recommended_certified_rate=min(1,max_certifiable_rate/2)` is a conservative
explicit option for this contract. It is never applied silently. A different
runtime needs a new issued receipt. Tests demonstrate the same Cauchy algorithm
failing at its default rate and passing at a smaller rate, the same clipping
mechanism passing with a narrower controller range, ordinary RMS passing with
a small rate, and blended RMS passing in a different coupling configuration.
This is not a loss or preconditioner whitelist.

Zero-curvature modes are explicitly MARGINAL, not strictly contractive. This
matters for unregularized intercepts and softmax common-logit shifts. Negative
curvature modes are outside the positive-mode envelope. Nonconvex losses are
not rejected just because they have negative curvature elsewhere.

### A limitation demonstrated by an exact counterexample

Frozen stability is **not** switching stability. For memory_step, h=19/10,
base step=1 and D=1, every frozen q in [0,1] passes the strict Schur test. But
the q=0 and q=1 matrices

`A0=[[-9/10,0],[19/10,0]]`,
`A1=[[781/800,-19/80],[19/200,19/20]]`

have `det(A1*A0)=0` and `trace(A1*A0)=-10639/8000<-1`. Alternating these
frozen matrices is unstable. The unit test checks this exact result and checks
that the filter leaves full adaptive convergence UNVERIFIED. Reachability of
that control sequence, time decay and clipping are not claimed by the example.
It prevents interpreting a frozen-envelope PASS as a full algorithm theorem.

## Conditional one-step certification

The new API accepts the **complete optimizer-core state**, not a family name:

```python
state = {
    't': 0,
    'theta': ['0','0'], 'fast': ['0','0'],
    'example_gradients': [['0','1/4'], ['0','1/4']],
    'signals': {s:'0' for s in engine.gen.SIGNALS},
    'smoothness_bound': '1'
}
# A canonical one-feature regression fixture (two parameters), analyzed without
# claiming session issuance or that this supplied state is a training trajectory:
g = engine.gen
controller = g.Controller.make('mean',
    g.Controller('signal', name='energy'), g.Controller('signal', name='dispersion'))
algorithm = g.AlgorithmSpec('regression', 'linear', 'standard', 'cauchy',
    'clip_step', 'mean', 'identity', controller)
candidate = g.Hypothesis(algorithm, g.TaskSpec('regression',1),
    'construct', (), 7, 1).to_dict()
receipt = engine.certify_transition(candidate, state)
assert engine.verify_transition(candidate, state, receipt)
```

Vectors flatten the executor's `(mapped_features+1, classes)` arrays in row-major
order. Required fields: t, theta, fast, example_gradients, all six signals and
positive smoothness_bound. Dual memory also requires slow; RMS requires
nonnegative variance. Time must equal `t/max(steps-1,1)`. The remaining controls
are explicitly supplied mathematical inputs, not certified data derivations.
There must be 2..32 gradient rows, with at most 128 flattened parameters.
Unknown fields, Boolean numeric inputs, bad shapes and invalid state domains
are rejected. Buffer magnitudes are not claimed reachable merely because they
can be analyzed. The receipt reports whether the previous EMA/RMS energy
invariant holds, without mistaking that necessary condition for reachability.

The transition reproduces the pinned executor's optimizer equations:

- Norms and the linearly interpolated 75th percentile of **norms**, not the
  square root of a percentile of squared norms.
- Clipping weights with the actual 1e-12 guard, then mean or coordinatewise
  median-of-block-means. Seeded blocks account for the earlier Fourier draw.
- The mechanism-specific current lambda penalty with the last class-width
  intercept row excluded, fast/slow memory, RMS/blend and normalized step.

Every intermediate scalar/vector is enclosed by rational intervals. Square-root
bounds are computed using integer square roots at 96 binary fractional bits.
Large denominators are rounded outward. No floating comparison determines a
mathematical sign. Receipt replay reconstructs all values and binds every input,
including unused signals, runtime seed, sources and the original full record.

The true gradient target is the mean of the **unclipped** supplied example
gradients plus the current frozen lambda penalty, not a clipped/MoM estimate.
For an L-smooth target objective on the entire step segment,

`F(theta+delta)-F(theta) <= g^T*delta + L*||delta||^2/2`.

A strictly negative certified upper bound gives PROVED_DESCENT. A strictly
positive lower bound on `g^T*delta` gives PROVED_NONDESCENT_DIRECTION. The latter
is a first-order obstruction; it is not a claim the finite nonconvex step must
increase F. Otherwise the result is INCONCLUSIVE. Both gradient correctness and
the segment smoothness bound are displayed assumptions; this API does not
authenticate gradients from a dataset or verify those assumptions from labels.
The target uses lambda_t held fixed; no decrease of a changed F_(t+1) is implied.

Tests reproduce clipping reversal with scalar gradients [4,-1,-1], reversal
from opposing momentum, and a diagonal-preconditioner reversal where positive
raw gradient/momentum alignment becomes negative after scaling. A norm cap
alone cannot repair a wrong direction.

## Demonstration and paired comparison

Same source-pinned generator, seeds 7/19/42/2026, regression and 3-class
classification, eight features, 100 proposals per combination. Default runtime:
160 steps, nominal learning_rate 2/25, execution seed 0. Production admission
and this large demonstration consume metadata only and run no fitting or
predictive scoring. Every admitted candidate was actually receipt-checked and
handed off; each rejected/pending candidate was checked to refuse handoff.

| Task | Seed | Admitted | Rejected | Pending | Nonconvex admitted |
|---|---:|---:|---:|---:|---:|
| Regression | 7 | 21 | 78 | 1 | 7 |
| Regression | 19 | 13 | 87 | 0 | 5 |
| Regression | 42 | 18 | 82 | 0 | 3 |
| Regression | 2026 | 9 | 91 | 0 | 1 |
| Classification | 7 | 15 | 83 | 2 | 10 |
| Classification | 19 | 22 | 76 | 2 | 12 |
| Classification | 42 | 24 | 76 | 0 | 12 |
| Classification | 2026 | 12 | 88 | 0 | 8 |
| Total | | 134 | 661 | 5 | 58 |

| Loss | Admitted | Rejected | Pending |
|---|---:|---:|---:|
| Cauchy | 16 | 90 | 0 |
| Huber | 15 | 82 | 0 |
| Squared | 15 | 113 | 0 |
| Pseudo-Huber | 15 | 53 | 1 |
| Brier | 20 | 134 | 3 |
| Focal | 22 | 88 | 0 |
| Cross-entropy | 31 | 101 | 1 |

Paired v1 mathematical eligibility on the identical records: 76 passed both,
58 passed only v2, 291 passed only v1, 375 passed neither. External v1 records
remain origin-unattested; comparison uses mathematical eligibility, not fake
ledger insertion. Existing real v2 issuance is used for actual handoffs.
The v1 source is optional for this comparison and is checked against its prior
published hash. No pass-rate comparison is presented as ML accuracy evidence.

26,845 claim-bound algebraic facts were VERIFIED, including repeated library
facts and cache reuse; this is not a count of new theorems. Full adaptive
convergence remains UNVERIFIED. All ten scientific acceptance gates stayed
UNVERIFIED. No literature novelty or scientific superiority was inferred.

Median admission time was 46.63 ms, p95 88.39 ms. Total generation/admission
time across rows was 40.21 seconds; additional receipt-checked handoff and v1
comparison were outside that interval. Some verification jobs shared the
machine. These are workload measurements with cache reuse, not a speed-leadership
or large-scale memory guarantee. Decision/reason digest, excluding timings:
`4a70c6e2c7cd83c590b79c63fa168b28c1077d792e46c51f2dc66210cadaed0c`.

## Tests, independent review and exposure history

**28 new test methods pass**, along with all 26 v1 admission, 22 generator and
33 verifier tests: **109 methods total**. New tests cover exact Schur boundaries,
dual cubic Jury versus an independent eigenvalue oracle, adaptive RMS energy on
exact sequences, 300 outward square-root cases, malformed states, replays,
nonmonotone feedback, controller/runtime sensitivity, rate boundaries and the
switching counterexample.

The four-file archive was extracted into a fresh temporary directory. All 28
new tests passed in a Python `-I` subprocess with explicit pinned dependencies
and no PYTHONPATH. The extracted autonomous CLI generated 24 candidates:
4 admitted, 20 rejected, scientific status NOT_EVALUATED. This is same-machine
package validation, not independent-machine scientific replication.

The executor-conformance test traces **132 actual optimizer steps** on fixed
public toy arrays across mechanisms, maps, aggregators, preconditioners and
couplings. It compares snapshots with the rational transition calculation.
This test executes the pinned reference fit loop on toy TRAIN-tagged inputs;
it measures no predictive score and is not a scientific experiment. Float
tolerances are diagnostics and never certify mathematical signs.

The reviewing agent separately inspected the update equations and proof scopes.
An independent 160-digit Decimal oracle checked 2,902 scalar/vector enclosures
over 60 valid challenge transitions, with zero misses. This is diagnostic
evidence, not a proof of the interval implementation or a new held-out score.
Review and reruns occurred in the same managed workspace, not an independent
institution or formally verified execution environment.

The original 132 challenges were sealed before the new implementation was
inspected. Categories: 84 transition requests (valid and invalid), 36 envelope
assessments, 12 replay-binding mutations. The first frozen run had 94 complete
contract matches. All mathematical verdicts and mutations matched, but 36
assessments lacked the promised rate-limit output and 2 invalid buffer requests
raised KeyError rather than ValueError. Those defects were fixed after exposure,
with public red/green regression tests. The H assumption was also clarified to
include the full penalized feedback. **Final unchanged-corpus regression is
132/132.** It is not untouched final held-out validation.

Initial core hash:
`60cf52e5f55aaaef7dafa4d518021bd915625a61e4c2e5a824d7fbb98dca6c10`.
Final independently reviewed core hash:
`6d3dd542732e68008eac4ed88bf133976493231745842e5e4a3916aa85b3f644`.
Initial independent summary hash:
`16cf5ba42764fecbf331bd5b3e40796c8abe20e78859b6bd84aee3fbc43bd3e4`.
Corrected independent summary hash:
`a1eac8ae46c6724bd1adb2915d99c9fccbcb204dbd28ffdc32a7152764e9bed5`.
Summaries are recorded here rather than adding more delivery files. The supplied
runner reproduces every corrected challenge outcome without modifying labels.

## Trust and limits

This is a fixed, reviewed mathematical model of the pinned optimizer core.
Algebraic subcertificates use the existing exact verifier; interval evaluation,
Schur/Jury, induction and calculus inference are reviewed Python, not Lean/Coq.
Source consistency does not prove finite-precision executor equivalence or
authenticate external observations. The process, installed dependencies and
source-to-mathematics binding remain trusted assumptions.

Explicit transitions are bounded to 32 examples and 128 flattened parameters,
with the original JSON structural/byte budgets, bounded numeric exponents and
4096-bit interval arithmetic guard. Interval widening may cause INCONCLUSIVE.
Admission's invariant and mode schemas do not enumerate all parameter
coordinates. Runs remain bounded batches (up to 10000), retain their results
and grow in memory with batch size; the session ledger refuses silent eviction.
No unbounded scalability or hard OS memory/time isolation is asserted.

Neither convexity nor frozen stability proves an algorithm is good or new.
Gradient correctness, signal derivation, curvature applicability, reachability,
full adaptive convergence, finite-machine behavior and the ten scientific
requirements are not silently inferred. This deliverable strengthens the
mathematical screen and exposes these remaining obligations for downstream work.

## Checksums

| File | SHA256 |
|---|---|
| math_filters.py | `6d3dd542732e68008eac4ed88bf133976493231745842e5e4a3916aa85b3f644` |
| test_math_filters.py | `d7a6ff9cb5e330346c58b1c2242e9c05b7819965d39b90f0ac59a1d0352ac7a1` |
| independent_cases.json | `19383c00918c5067f384908c82e0b5277425812fbb0b2d45dda015b492effd79` |
| unchanged generator | `8ce3010e741603c9627c85f37944586c4cde2155eeb24660adcfd58ca5931b3f` |
| unchanged verifier | `37dc50931bfe2512875558e5660de692680c8b232333a98885503da9697205fc` |
| unchanged v1, comparison only | `7aee339f092dfc64823979cd2ef74d0e6d48f80d810992668d76a053feed66b0` |
