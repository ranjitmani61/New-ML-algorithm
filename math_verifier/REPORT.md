# Exact mathematical checker and verification layer

Demonstrated on 2026-10-31 (client date). Mathematical verification only; the
scientific evaluation engine and the existing generator were not modified.
Delivery: one implementation file and four supporting files.

The measured contribution is exact, claim-specific certificate production and
replay for supported mathematical expressions. This is a scoped improvement
over the previous literal-probe checker. Global superiority over theorem
provers or verification across all mathematics has not been established.

## Files and reproduction

| File | Purpose |
|---|---|
| `math_checker.py` | Entire production implementation, API and CLI; standard library only |
| `test_math_checker.py` | Public contracts, symbolic-oracle tests and demonstration runners |
| `verification_cases.json` | Original 58 externally authored cases and unchanged expected labels |
| `REPORT.md` | Design, measurements, assumptions and limitations |
| `requirements.txt` | SymPy for oracle tests; no production runtime dependencies |

Requires Python 3.10+. Tested on CPython 3.12.14, Linux x86_64, SymPy 1.14.0.
The production module imports no CAS, ML library, network client or model.

```bash
python math_checker.py check --statement 'For all real numbers x, x^2 >= 0.'
python math_checker.py coverage
python -m pip install -r requirements.txt
python -m unittest test_math_checker -v
python test_math_checker.py --external verification_cases.json
python test_math_checker.py --memory-demo
# Optional unchanged files from the existing workspace:
python test_math_checker.py --compare ../math_discovery/math_checker.py
python test_math_checker.py --generator-demo ../generator_only/hypothesis_generator.py
```

JSON claims: `python math_checker.py check claim.json`, or `-` for stdin.
Replay: `python math_checker.py replay receipt.json`, where the receipt has
exactly `claim` and `certificate` fields. The previous baseline needs its
adjacent `claim_probes.py`. The generator demo imports explicitly supplied,
trusted Python source; the production checker consumes generator JSON without
importing or executing that source.

```python
from math_checker import MathCheckerEngine, verify_certificate

claim = {
    "kind": "scalar", "variables": {"x": {"domain": "real"}},
    "lhs": "x*x+2*x+2", "op": ">", "rhs": "0"
}
result = MathCheckerEngine().check(claim)
assert result.status == "VERIFIED"
assert verify_certificate(claim, result.certificate)
```

## Actual verification interfaces

| Kind | Scope |
|---|---|
| `scalar` | Universally quantified real/integer relations under declared bounds and premises; polynomial/rational identities, interval/sign/SOS inequalities |
| `derivative` | Equality to a generated symbolic derivative with explicit domain and regularity obligations |
| `finite` | Complete enumeration of an explicitly bounded integer box after decidable premises |
| `logic` | Complete classical Boolean truth tables |
| `psd` | Exact rational symmetric matrix PSD certificate or negative quadratic-form witness |
| `matrix_identity` | Shape-checked symbolic matrix identities for the supplied dimensions |
| `ode` | Explicit first-order scalar ODE solution identity and initial condition |
| generator record | Conditional mathematical subclaims of a typed `generator-3.0` definition |

Scalar `variables` maps names to `domain` (`real` or `integer`), optional exact
`lower`/`upper`, and Boolean `lower_open`/`upper_open`. Optional `assumptions`
contains `{lhs, op, rhs}` relations. Operators: `==`, `!=`, `<`, `>`, `<=`, `>=`,
with `=` as an input alias. Inputs use integers or exact decimal/rational
strings; binary floating-point inputs and Boolean numbers are rejected.

Admitted functions: `exp`, `log`, `sqrt`, `abs`, `min`, `max`, `sin`, `cos`,
`gcd`, and `binomial`/`comb`; constants include `pi` and `e`. Recognition does
not imply a proof rule for every identity. Integer functions require explicit
finite enumeration. Original exponents are integers -8..8, with `^` accepted
as `**`. The algebraic convention `a^0=1`, including `0^0=1`, is explicit.
Square roots use nonnegative real roots; logarithms require positive arguments.
Complex values are outside this API.

The limited whole-statement grammar supports `For all integers a and b,
a+b=b+a.` and `forall real x,y: (x+y)^2=x*x+2*x*y+y*y`. Arbitrary prose stays
unverified. Clauses are never discarded to fit a template. Typed examples for
every backend are present in tests and the external corpus.

| Status | Meaning |
|---|---|
| VERIFIED | Replayed proof of the exact supplied claim under its assumptions |
| VERIFIED_FINITE | Exhaustive proof over the declared finite population |
| DISPROVEN | Exact admissible counterexample with replay |
| ILL_DEFINED | Replayed admissible point where a submitted expression is undefined |
| VACUOUS | Proved empty domain; not positive theorem evidence |
| UNVERIFIED | No supported proof or exact refutation was established |
| INVALID / RESOURCE_LIMIT | Malformed input / computational bound reached |

For Boolean formulas, the finite truth table includes **every** Boolean
valuation, so it proves the complete Boolean claim. Integer-box results retain
their explicitly finite scope. There is no evidence score or truth probability.
Family names, unrelated theorems and successful samples cannot create approval.

## Mathematical foundation and trusted base

Polynomial identities use sparse exact rational normalization. Elementary
functions become opaque real atoms in polynomial algebra: an identity valid
for all atom values is sufficient, while additional analytic identities may
remain unresolved. All original division, logarithm and square-root domain
obligations are checked before verification, including canceled expressions.

Nonnegativity certificates establish `p = sum_i c_i*q_i^2`, with `c_i>=0`.
Replay re-expands the entire identity. Strict positivity additionally requires
inconsistent simultaneous zeros of positive-weight linear forms, checked by
exact rational ranks, or another strict sign/interval argument. Product
certificates re-multiply factors and replay their individual signs. Rational
interval enclosures include endpoint openness and conservative infinities.

PSD certificates establish exact `A=L*diag(D)*L^T`, unit lower-triangular `L`,
and nonnegative `D`. Hence `x^T A x=sum_i D_i*(L^T x)_i^2>=0`. Refutations
supply rational `x` with exact `x^T A x<0`. Matrix identities reduce to exact
scalar identities for the stated shapes, never arbitrary unspecified sizes.

Derivative/ODE checks use standard sum/product/quotient/chain rules. Square-root
differentiation requires certified strict positivity of the argument. This
sufficient regularity condition intentionally abstains on some valid boundary
cases. ODE verification checks the given solution and initial condition; it
does not assert uniqueness, global existence or numerical-solver accuracy.

Public certificate replay invokes no discovery routine. A regression disables
sign/SOS discovery while replaying a square-root derivative. Replay reconstructs
the claim and obligations, checks its hash, and verifies the receipt algebra.
It never trusts a verdict label alone. The trusted base is this Python code,
Python exact integers/Fraction, and the stated primitive mathematical rules.
The implementation is not itself mechanically verified, and no Lean/Coq proof
kernel is integrated.

## Public tests and review

All **33 public test methods pass**, including **304 symbolic-oracle decisions**:
80 generated polynomial identities and 80 false mutations, 40 generated
derivatives and 40 false mutations, and 64 exact PSD decisions checked against
SymPy. These are declared development tests, not held-out data. Another test
checks exact enclosures on 500 seeded rational interval pairs.

The five-file archive was also extracted into a fresh temporary directory and
all 33 tests passed in an isolated Python `-I` subprocess without PYTHONPATH.
The extracted CLI correctly rejected cancellation through an undefined `x/x`
at zero. This demonstrates package self-containment on the same machine;
it is not independent-machine replication.

Contracts cover cancellation at division by zero, exact decimal traps,
fractional counterexamples, zero-pivot indefinite matrices, wrong initial
conditions, cache/assumption isolation, altered domain/SOS certificates,
executable-expression injection, malformed signatures and resource attempts.
Mock tests confirm huge binomial work is rejected before calling native comb.

A separate reviewing agent inspected replay and generator linkage and ran
in-memory arithmetic probes. Review found unbounded binomial work, an incorrect
dual-memory linkage coefficient and discovery during square-root replay. All
were fixed and regression-tested before freezing the implementation. This is
second-agent review in the same workspace, not human formal validation or
independent-machine replication.

## Sealed external evaluation and leakage safeguards

A reviewer authored the **58 cases before inspecting implementation**. The
parent developed against public tests, froze the implementation, and only then
opened the corpus. The production module reads no corpus, expected labels,
benchmark rows or case IDs. No implementation changes followed exposure.
The corpus is now public regression material; future tuning on it cannot be
called untouched evaluation.

First-run and delivered implementation SHA256:
`37dc50931bfe2512875558e5660de692680c8b232333a98885503da9697205fc`

Predeclared and unchanged corpus SHA256:
`302afc6cede16397db4601a88bcc6c9eeb7e5f46a99f1aedaabc10e6a1d52094`

| Actual status | Count |
|---|---:|
| VERIFIED | 20 |
| VERIFIED_FINITE | 6 |
| DISPROVEN | 16 |
| ILL_DEFINED | 4 |
| VACUOUS | 2 |
| UNVERIFIED | 6 |
| INVALID | 3 |
| RESOURCE_LIMIT | 1 |

Raw result: **53/58 exact expected-status matches (91.4%)**, with **48/48
emitted certificates replayed**. No mathematically false positive approval
was observed in these cases. This does not establish a global zero-error
guarantee. Median cold latency was 0.244 ms, p95 3.446 ms and maximum 20.924 ms
on the recorded rerun, including internal replay. First-run status counts and
matches were identical. Original expected labels were never changed.

| Discrepancy | Original expected | Actual | Explanation |
|---|---|---|---|
| ext_037 | VERIFIED | VERIFIED_FINITE | Complete Boolean implication truth table |
| ext_038 | VERIFIED | VERIFIED_FINITE | Complete Boolean contrapositive truth table |
| ext_040 | VERIFIED | VERIFIED_FINITE | Complete Boolean excluded-middle truth table |
| ext_055 | INVALID or UNVERIFIED | VERIFIED | Corpus planned to exclude sin; frozen parser explicitly admits real sin, and sin(x)=sin(x) has a valid identity proof |
| ext_058 | INVALID or UNVERIFIED | RESOURCE_LIMIT | Exponent 9 exceeds -8..8; fails closed without a certificate |

Examples left unverified include `sqrt(x^2)=abs(x)`, `log(exp(x))=x`, and
`x^2>=1` under `x>=1`. These are genuine incompleteness limitations, not unknown
mathematical truths. Absolute-value differentiation across zero and derivative
claims with unresolved domains also abstain. No numerical fallback promotes them.

No training/validation/test datasets, predictive scores or reward feedback are
consumed by this layer. Safeguards are architectural and procedural; software
cannot prevent an external caller from misrepresenting supplied assumptions.

## Matched comparison with the previous workspace checker

The declared development cohort gives **42 identical full scalar statements**
to both engines: two canonical registry statements plus shifted squares,
expanded two-variable squares and false mutations generated by the runner.
The baseline is the existing conservative **12-literal-probe checker**, not a
recreation of the older family-score implementation.

| Metric | Previous | New |
|---|---:|---:|
| Positive results | 2 TESTED-PASS | 22 VERIFIED |
| Exact refutations | 0 | 20 DISPROVEN |
| Undecided | 40 | 0 |
| Replayed global proof certificates | 0 | 22 |
| Median latency | 0.513 ms | 0.926 ms |
| p95 latency | 0.664 ms | 5.233 ms |

Timings use five fresh-engine repeats; unique claims receive no cache-hit
benefit. Previous positive results were finite tests, not proofs. The new
engine resolves the declared cohort while doing extra proof/replay work.
These results show no latency win or general theorem-prover superiority.
This cohort favors expression generalization beyond a literal registry.
The old graph, set and probability phrases have not all been ported, so this
comparison must not be extrapolated to those domains or all mathematics.

Baseline SHA256:
`07dd908331aa35cd934bfa4ec729182b5af79aef94e1056b728441ef931741f0`

## Integration with the unchanged generator

`MathCheckerEngine().check(hypothesis)` accepts the current generator object or
its `to_dict()` JSON. It validates schema/components and the exact definition
signature, then extracts mathematical obligations. It reads no dataset and
trains no candidate. Caller annotations are not substituted for typed claims.

The metadata demo used classification/regression, seeds 7 and 42, and 100
hypotheses per task/seed: **400 hypotheses, 5,342 conditional subclaims**.
Every extracted subclaim was VERIFIED and replayed again by the runner.
All 400 complete empirical hypotheses remained UNVERIFIED. Subclaim instances
include repeated lemmas reused through a claim-specific cache; they are not
5,342 distinct new theorems.

Obligations cover conditional controller bounds, clipping weights,
mechanism-specific coefficients/step factors, declared loss derivatives,
softmax coordinate Jacobians and scalar step-cap branches. The dual-memory
fast coefficient is its actual 1/2. The declared q2 coupling is substituted
as q or 1/2. Each result exposes its assumptions and partial scope.

Signal-range derivations, complete vector-gradient composition, Huber boundary
matching, finite-precision implementation semantics, convergence and empirical
improvement are not silently certified. A scalar norm-cap consequence does not
verify an entire numerical update program. No scientific engine was built.

Median full metadata audit: 9.276 ms; maximum 29.602 ms here, with cache reuse.
The original generator is unchanged, SHA256:
`8ce3010e741603c9627c85f37944586c4cde2155eeb24660adcfd58ca5931b3f`.

## Resources and mathematical-family coverage

Default limits: 4,096 expression characters; 192 parsed nodes; 24 variables;
4,096 polynomial terms; degree 32; 200,000 budget operations; 1,024 coefficient
bits; 4,096 enumerated assignments; 192 refutation trials; 300,000 payload
bytes; 512 LRU cache entries. Symbolic matrix shapes are at most 4 per side,
subject to the variable cap; numeric PSD matrices are at most 16 by 16.
Derived expressions are also bounded. Exact power/binomial work is bounded
before computation, and intermediate arithmetic growth is checked.

The 3-second default cooperative budget applies separately to discovery and
replay, not OS-enforced process isolation or a hard service wall-clock cap.
Generator audits invoke several separately budgeted subclaims. Configuration
is trusted. Worst-case expansion may reach RESOURCE_LIMIT, never approval.

Separate `tracemalloc` run: **1,000 distinct proven identities**, cache capped
at 512 entries, current tracked allocation 946,471 bytes, peak 1,006,422 bytes.
These are Python allocation measurements on this small workload, not full
process RSS or a universal memory/scalability guarantee. Run `--memory-demo`.

The coverage endpoint lists **all 63 top-level MSC2020 sections** from
[Mathematical Reviews/zbMATH's MSC2020](https://mathscinet.ams.org/msc/pdfs/classifications2020.pdf)
and [zbMATH's classification](https://zbmath.org/classification/).
This is full top-level taxonomy coverage, not verification of arbitrary
claims in every section. Diagnostic whole-word routing supplies no evidence.
Actual verification is limited to the eight interfaces above. Arbitrary
geometry, topology, category/measure theory, PDEs, open problems and research
claims require additional formalization/backends and remain unverified here.

## Reproduction checksums

| File | SHA256 |
|---|---|
| math_checker.py | `37dc50931bfe2512875558e5660de692680c8b232333a98885503da9697205fc` |
| test_math_checker.py | `7211817ded6f80a051975772c6fa3129d56524f4b5029449b6521e4081bb3c2c` |
| verification_cases.json | `302afc6cede16397db4601a88bcc6c9eeb7e5f46a99f1aedaabc10e6a1d52094` |
| requirements.txt | `1a7baeeb23476d4e1f104545e6fd91ce252257cafc2517c7450965274644c01f` |

The runner prints case-level outcomes and measured metadata. Timing and memory
depend on hardware. These results assert mathematical/engineering capabilities;
they do not establish ML novelty, predictive advantage or ten-gate scientific
acceptance.
