"""Math verification contracts. No scientific evaluation or predictive data."""

import copy
import hashlib
import importlib.util
import json
import pathlib
import platform
import random
import statistics
import sys
import time
import unittest
from unittest.mock import patch
from fractions import Fraction


def scalar(lhs, op, rhs, variables=None, **kw):
    return dict(
        kind="scalar",
        variables=variables if variables is not None else {"x": {"domain": "real"}},
        lhs=lhs,
        op=op,
        rhs=rhs,
        **kw,
    )


class ExactContracts(unittest.TestCase):
    def setUp(self):
        from math_checker import MathCheckerEngine

        self.engine = MathCheckerEngine()

    def assert_verified(self, c, finite=False):
        from math_checker import verify_certificate

        r = self.engine.check(c)
        self.assertEqual(
            r.status, "VERIFIED_FINITE" if finite else "VERIFIED", r.to_dict()
        )
        self.assertTrue(verify_certificate(c, r.certificate))
        return r

    def test_polynomial_identity(self):
        self.assert_verified(
            scalar("(x+y)^2", "==", "x*x+2*x*y+y*y", {"x": {}, "y": {}})
        )

    def test_complete_quantified_integer_statement(self):
        r = self.engine.check("For all integers a and b, a + b = b + a.")
        self.assertEqual(r.status, "VERIFIED", r.to_dict())

    def test_quadratic_nonnegative(self):
        self.assert_verified(scalar("x*x+2*x+2", ">", 0))

    def test_counterexample_exact(self):
        from math_checker import verify_certificate

        c = scalar("x*x", ">=", "x")
        r = self.engine.check(c)
        self.assertEqual(r.status, "DISPROVEN", r.to_dict())
        self.assertTrue(verify_certificate(c, r.certificate))
        x = Fraction(r.counterexample["x"])
        self.assertLess(x * x, x)

    def test_undefined_does_not_cancel(self):
        c = scalar("x/x", "==", 1)
        r = self.engine.check(c)
        self.assertEqual(r.status, "ILL_DEFINED", r.to_dict())
        self.assert_verified(
            scalar("x/x", "==", 1, {"x": {"lower": "0", "lower_open": True}})
        )

    def test_decimal_literals_exact(self):
        r = self.engine.check(scalar("1.0000000000000001", "==", "1", {}))
        self.assertEqual(r.status, "DISPROVEN", r.to_dict())

    def test_finite_scope(self):
        c = scalar(
            "n*n", ">=", "n", {"n": {"domain": "integer", "lower": "-4", "upper": "4"}}
        )
        c["kind"] = "finite"
        self.assert_verified(c, finite=True)

    def test_derivative_and_ode(self):
        self.assert_verified(
            {
                "kind": "derivative",
                "variables": {"x": {}},
                "expression": "log(1+x*x)",
                "variable": "x",
                "proposed": "2*x/(1+x*x)",
            }
        )
        self.assert_verified(
            {
                "kind": "ode",
                "variables": {"t": {}},
                "variable": "t",
                "solution": "exp(t)",
                "rhs": "y",
                "initial": {"at": "0", "value": "1"},
            }
        )

    def test_psd_and_matrix_identity(self):
        self.assert_verified({"kind": "psd", "matrix": [["2", "1"], ["1", "2"]]})
        self.assert_verified(
            {
                "kind": "matrix_identity",
                "matrices": {"A": [2, 2], "B": [2, 2]},
                "lhs": "trace(A*B)",
                "rhs": "trace(B*A)",
            }
        )

    def test_logic(self):
        self.assert_verified(
            {"kind": "logic", "formula": "implies(p and q,p)"}, finite=True
        )
        r = self.engine.check({"kind": "logic", "formula": "implies(p,q)"})
        self.assertEqual(r.status, "DISPROVEN")

    def test_tamper_and_changed_claim(self):
        from math_checker import verify_certificate

        c = scalar("x*x", ">=", 0)
        r = self.assert_verified(c)
        bad = copy.deepcopy(r.certificate)
        bad["claim_hash"] = "0" * 64
        self.assertFalse(verify_certificate(c, bad))
        self.assertFalse(verify_certificate(scalar("x*x", "<", 0), r.certificate))

    def test_unstructured_and_injection(self):
        self.assertEqual(
            self.engine.check(
                "An algorithm proves every possible mathematical result."
            ).status,
            "UNVERIFIED",
        )
        self.assertEqual(
            self.engine.check(
                scalar("__import__('os').system('echo hacked')", "==", "0")
            ).status,
            "INVALID",
        )

    def test_vacuous_and_cache_isolation(self):
        c = scalar("x", ">", 0, {"x": {"lower": "2", "upper": "1"}})
        self.assertEqual(self.engine.check(c).status, "VACUOUS")
        c = scalar("x*x", ">=", 0)
        r = self.engine.check(c)
        r.status = "BAD"
        self.assertEqual(self.engine.check(c).status, "VERIFIED")

    def test_arithmetic_work_is_bounded_before_computation(self):
        c = scalar("binomial(1000000000,500000000)", ">=", "0", {})
        with patch(
            "math_checker.math.comb",
            side_effect=AssertionError("unbounded comb called"),
        ):
            self.assertEqual(self.engine.check(c).status, "RESOURCE_LIMIT")
        c = scalar("(((2^8)^8)^8)^8", ">", 0, {})
        self.assertEqual(self.engine.check(c).status, "RESOURCE_LIMIT")

    def test_uncertain_premises_are_not_empty_domains(self):
        for premise in (
            {"lhs": "exp(n)", "op": ">", "rhs": "0"},
            {"lhs": "1/(n-1)", "op": ">", "rhs": "0"},
        ):
            c = scalar(
                "n",
                "==",
                "n",
                {"n": {"domain": "integer", "lower": "1", "upper": "1"}},
                assumptions=[premise],
            )
            c["kind"] = "finite"
            self.assertEqual(self.engine.check(c).status, "UNVERIFIED")

    def test_polynomial_product_bound(self):
        self.assert_verified(
            scalar(
                "eta*s",
                "<=",
                "eta",
                {"eta": {"lower": "0"}, "s": {"lower": "0", "upper": "1"}},
            )
        )

    def test_square_root_derivative_boundary_abstains(self):
        c = {
            "kind": "derivative",
            "variables": {"x": {}},
            "expression": "sqrt(x^4)",
            "variable": "x",
            "proposed": "2*x",
        }
        self.assertEqual(self.engine.check(c).status, "UNVERIFIED")

    def test_smooth_pseudo_huber_derivative(self):
        self.assert_verified(
            {
                "kind": "derivative",
                "variables": {"r": {}},
                "expression": "sqrt(1+r*r)-1",
                "variable": "r",
                "proposed": "r/sqrt(1+r*r)",
            }
        )

    def test_payload_tamper_rejected(self):
        from math_checker import verify_certificate

        c = scalar("x*x+2*x+2", ">", 0)
        r = self.assert_verified(c)
        bad = copy.deepcopy(r.certificate)
        bad["payload"]["proof"]["sos"]["terms"][0][0] = "-1"
        self.assertFalse(verify_certificate(c, bad))

    def test_replay_never_calls_certificate_discovery(self):
        from math_checker import verify_certificate

        c = {
            "kind": "derivative",
            "variables": {"x": {}},
            "expression": "sqrt(1+x*x)",
            "variable": "x",
            "proposed": "x/sqrt(1+x*x)",
        }
        r = self.assert_verified(c)
        with patch(
            "math_checker.sign_proof",
            side_effect=AssertionError("proof search during replay"),
        ), patch(
            "math_checker.sos_discover",
            side_effect=AssertionError("SOS discovery during replay"),
        ):
            self.assertTrue(verify_certificate(c, r.certificate))
        c = scalar("x/x", "==", 1, {"x": {"lower": "0", "lower_open": True}})
        r = self.assert_verified(c)
        bad = copy.deepcopy(r.certificate)
        bad["payload"]["domain"] = []
        self.assertFalse(verify_certificate(c, bad))

    def test_claim_cache_binds_all_assumptions(self):
        good = scalar("x", ">", 0, {"x": {"lower": "0", "lower_open": True}})
        self.assertEqual(self.engine.check(good).status, "VERIFIED")
        bad = copy.deepcopy(good)
        bad["variables"]["x"] = {}
        self.assertEqual(self.engine.check(bad).status, "DISPROVEN")

    def test_floats_booleans_unknown_fields_rejected(self):
        for value in (0.1, True):
            self.assertEqual(
                self.engine.check(scalar("x", ">=", value)).status, "INVALID"
            )
        c = scalar("x", "==", "x")
        c["evidence_score"] = 1
        self.assertEqual(self.engine.check(c).status, "INVALID")

    def test_finite_binomial_and_gcd(self):
        c = {
            "kind": "finite",
            "variables": {
                "n": {"domain": "integer", "lower": "2", "upper": "18"},
                "k": {"domain": "integer", "lower": "1", "upper": "17"},
            },
            "assumptions": [{"lhs": "k", "op": "<", "rhs": "n"}],
            "lhs": "binomial(n,k)",
            "op": "==",
            "rhs": "binomial(n-1,k-1)+binomial(n-1,k)",
        }
        self.assert_verified(c, finite=True)
        self.assert_verified(
            {
                "kind": "finite",
                "variables": {
                    "a": {"domain": "integer", "lower": "-8", "upper": "8"},
                    "b": {"domain": "integer", "lower": "-8", "upper": "8"},
                },
                "lhs": "gcd(a,b)",
                "op": "==",
                "rhs": "gcd(b,a)",
            },
            finite=True,
        )

    def test_empty_integer_interval_not_a_real_proof(self):
        c = {
            "kind": "finite",
            "variables": {
                "n": {
                    "domain": "integer",
                    "lower": "0",
                    "upper": "1",
                    "lower_open": True,
                    "upper_open": True,
                }
            },
            "lhs": "n",
            "op": "<",
            "rhs": "0",
        }
        self.assertEqual(self.engine.check(c).status, "VACUOUS")

    def test_matrices_shape_and_noncommutativity(self):
        c = {
            "kind": "matrix_identity",
            "matrices": {"A": [2, 2], "B": [2, 2]},
            "lhs": "A*B",
            "rhs": "B*A",
        }
        r = self.engine.check(c)
        self.assertEqual(r.status, "DISPROVEN", r.to_dict())
        self.assert_verified(
            {
                "kind": "matrix_identity",
                "matrices": {"A": [2, 2]},
                "lhs": "det(transpose(A))",
                "rhs": "det(A)",
            }
        )
        c["matrices"]["A"] = [2, 3]
        self.assertEqual(self.engine.check(c).status, "INVALID")

    def test_psd_zero_pivot_witness(self):
        from math_checker import verify_certificate

        c = {"kind": "psd", "matrix": [[0, 1], [1, 100]]}
        r = self.engine.check(c)
        self.assertEqual(r.status, "DISPROVEN", r.to_dict())
        self.assertTrue(verify_certificate(c, r.certificate))

    def test_ode_wrong_initial_condition(self):
        c = {
            "kind": "ode",
            "variables": {"t": {}},
            "variable": "t",
            "solution": "exp(t)",
            "rhs": "y",
            "initial": {"at": "0", "value": "2"},
        }
        self.assertEqual(self.engine.check(c).status, "DISPROVEN")

    def test_coverage_taxonomy_is_not_proof_support(self):
        from math_checker import coverage_report

        report = coverage_report()
        self.assertEqual(report["recognized_top_level_sections"], 63)
        self.assertFalse(report["arbitrary_claim_verification"])
        for s in (
            "The Riemann hypothesis is true.",
            "P = NP.",
            "Every even integer greater than two is a sum of two primes.",
        ):
            self.assertEqual(self.engine.check(s).status, "UNVERIFIED")

    def test_generator_linkage_and_provenance(self):
        from math_checker import digest, verify_certificate

        algorithm = {
            "kind": "regression",
            "feature_map": "linear",
            "scaling": "standard",
            "loss": "cauchy",
            "mechanism": "dual_memory",
            "aggregation": "mean",
            "preconditioner": "identity",
            "controller": {"op": "half", "name": "", "children": []},
            "coupling": "coupled",
        }
        c = {
            "schema_version": "generator-3.0",
            "algorithm": algorithm,
            "task": {"kind": "regression", "features": 4, "classes": 1},
            "signature": digest(algorithm),
            "statement": "A generated candidate might improve learning.",
        }
        r = self.engine.check(c)
        self.assertEqual(r.status, "UNVERIFIED")
        self.assertGreater(len(r.subclaims), 8)
        for sub in r.subclaims:
            self.assertEqual(sub["verdict"]["status"], "VERIFIED")
            self.assertTrue(
                verify_certificate(sub["claim"], sub["verdict"]["certificate"])
            )
            if sub["obligation"].startswith("momentum_coefficient"):
                self.assertEqual(sub["claim"]["lhs"], "1/2")
        c["signature"] = "0" * 64
        self.assertEqual(self.engine.check(c).status, "INVALID")
        c["signature"] = digest(algorithm)
        c["test_scores"] = [1]
        self.assertEqual(self.engine.check(c).status, "INVALID")


class OracleProperties(unittest.TestCase):
    """Development cases: an external CAS supplies expected algebra, never proof data."""

    def test_polynomial_identities_and_mutations(self):
        import sympy as sp
        from math_checker import MathCheckerEngine, verify_certificate

        rng = random.Random(8073)
        x, y = sp.symbols("x y")
        engine = MathCheckerEngine()
        for _ in range(80):
            a, b, c = [rng.randint(-6, 6) for _ in range(3)]
            expression = (a * x + b * y + c) ** 2
            claim = scalar(
                str(expression), "==", str(sp.expand(expression)), {"x": {}, "y": {}}
            )
            result = engine.check(claim)
            self.assertEqual(result.status, "VERIFIED", result.to_dict())
            self.assertTrue(verify_certificate(claim, result.certificate))
            claim["rhs"] = "(" + claim["rhs"] + ")+1"
            result = engine.check(claim)
            self.assertEqual(result.status, "DISPROVEN", result.to_dict())

    def test_symbolic_derivatives_against_sympy(self):
        import sympy as sp
        from math_checker import MathCheckerEngine

        rng = random.Random(517)
        x = sp.symbols("x")
        engine = MathCheckerEngine()
        for _ in range(40):
            expression = sum(rng.randint(-4, 4) * x**j for j in range(6))
            derivative = sp.diff(expression, x)
            c = {
                "kind": "derivative",
                "variables": {"x": {}},
                "expression": str(expression),
                "variable": "x",
                "proposed": str(derivative),
            }
            self.assertEqual(engine.check(c).status, "VERIFIED")
            c["proposed"] = "(" + c["proposed"] + ")+1"
            self.assertEqual(engine.check(c).status, "DISPROVEN")

    def test_psd_decisions_against_exact_sympy(self):
        import sympy as sp
        from math_checker import MathCheckerEngine, verify_certificate

        rng = random.Random(271)
        engine = MathCheckerEngine()
        for i in range(64):
            n = 1 + i % 4
            b = sp.Matrix(n, n, lambda r, c: rng.randint(-4, 4))
            if i % 2:
                a = b.T * b
            else:
                a = b + b.T
            expected = "VERIFIED" if a.is_positive_semidefinite else "DISPROVEN"
            c = {
                "kind": "psd",
                "matrix": [[str(a[r, k]) for k in range(n)] for r in range(n)],
            }
            result = engine.check(c)
            self.assertEqual(result.status, expected, result.to_dict())
            self.assertTrue(verify_certificate(c, result.certificate))

    def test_interval_hulls_contain_exact_points(self):
        from math_checker import Interval, imul, iadd, ineg, reciprocal

        rng = random.Random(139)
        for _ in range(500):
            a, b = sorted(rng.sample(range(-5, 6), 2))
            c, d = sorted(rng.sample(range(-5, 6), 2))
            u = Interval(
                Fraction(a), Fraction(b), bool(rng.randrange(2)), bool(rng.randrange(2))
            )
            v = Interval(
                Fraction(c), Fraction(d), bool(rng.randrange(2)), bool(rng.randrange(2))
            )
            xs = [Fraction(a) + (Fraction(b - a) * j) / 4 for j in range(5)]
            ys = [Fraction(c) + (Fraction(d - c) * j) / 4 for j in range(5)]
            for x in xs:
                if not u.contains(x):
                    continue
                self.assertTrue(ineg(u).contains(-x))
                if u.nonzero:
                    self.assertTrue(reciprocal(u).contains(1 / x))
                for y in ys:
                    if v.contains(y):
                        self.assertTrue(imul(u, v).contains(x * y))
                        self.assertTrue(iadd(u, v).contains(x + y))


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def generator_demo(source, count=100):
    """Metadata-only integration, no datasets or scientific evaluation."""
    from math_checker import MathCheckerEngine, verify_certificate

    gen = load_source(source, "trusted_generator_for_math_audit")
    engine = MathCheckerEngine()
    totals = {}
    obligations = {}
    elapsed = []
    whole = []
    for kind in ("classification", "regression"):
        for seed in (7, 42):
            for h in gen.HypothesisGenerator(seed).generate(
                gen.TaskSpec(kind, 8), count
            ):
                start = time.perf_counter()
                result = engine.check(h)
                elapsed.append((time.perf_counter() - start) * 1000)
                if result.status != "UNVERIFIED":
                    raise AssertionError("whole empirical hypothesis promoted")
                whole.append(result.status)
                for entry in result.subclaims:
                    v = entry["verdict"]
                    totals[v["status"]] = totals.get(v["status"], 0) + 1
                    byname = obligations.setdefault(entry["obligation"], {})
                    byname[v["status"]] = byname.get(v["status"], 0) + 1
                    if v["status"] == "VERIFIED" and not verify_certificate(
                        entry["claim"], v["certificate"]
                    ):
                        raise AssertionError("generator lemma replay failed")
    return {
        "generator_sha256": hashlib.sha256(
            pathlib.Path(source).read_bytes()
        ).hexdigest(),
        "metadata_only": True,
        "whole_hypotheses": len(whole),
        "whole_status": "UNVERIFIED",
        "subclaim_statuses": totals,
        "obligations": obligations,
        "latency_ms_median": statistics.median(elapsed),
        "latency_ms_max": max(elapsed),
    }


def matched_text_comparison(baseline_source):
    """Declared development comparison: identical full statements to both engines.

    The old checker has a literal registry; this deliberately measures general
    mathematical expressions beyond that registry, not its graph/set coverage.
    It is a scope comparison, not a universal theorem-prover leaderboard.
    """
    from math_checker import MathCheckerEngine, verify_certificate
    from types import SimpleNamespace

    fixtures = [
        ("For all real numbers x, x^2 >= 0.", "VERIFIED"),
        ("For all integers a and b, a + b = b + a.", "VERIFIED"),
    ]
    for k in range(1, 11):
        expanded = f"x^2+y^2+{k*k}+2*x*y+{2*k}*x+{2*k}*y"
        fixtures.extend(
            [
                (f"For all real numbers x, (x+{k})^2 >= 0.", "VERIFIED"),
                (f"For all real numbers x, (x+{k})^2 < 0.", "DISPROVEN"),
                (f"For all real numbers x,y, (x+y+{k})^2 = {expanded}.", "VERIFIED"),
                (f"For all real numbers x,y, (x+y+{k})^2 = {expanded}+1.", "DISPROVEN"),
            ]
        )
    folder = str(pathlib.Path(baseline_source).resolve().parent)
    sys.path.insert(0, folder)
    try:
        baseline = load_source(baseline_source, "previous_workspace_math_checker")
    finally:
        sys.path.pop(0)
    rows = []
    times = {"previous": [], "new": []}
    counts = {"previous": {}, "new": {}}
    correct = {"previous": 0, "new": 0}
    # Five fresh-engine repeats exclude claim-cache benefits and include replay.
    for repeat in range(5):
        previous = baseline.MathCheckerEngine()
        current = MathCheckerEngine()
        for statement, expected in fixtures:
            start = time.perf_counter()
            old = previous.check(SimpleNamespace(statement=statement, components=()))
            times["previous"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            new = current.check(statement)
            times["new"].append((time.perf_counter() - start) * 1000)
            if repeat == 0:
                for name, status in (("previous", old.status), ("new", new.status)):
                    counts[name][status] = counts[name].get(status, 0) + 1
                    if status == expected or (
                        status == "TESTED-PASS" and expected == "VERIFIED"
                    ):
                        correct[name] += 1
                if new.certificate:
                    from math_checker import parse_statement

                    if not verify_certificate(
                        parse_statement(statement), new.certificate
                    ):
                        raise AssertionError("comparison certificate replay failed")
                rows.append(
                    {
                        "statement": statement,
                        "expected": expected,
                        "previous": old.status,
                        "new": new.status,
                    }
                )
    return {
        "baseline_sha256": hashlib.sha256(
            pathlib.Path(baseline_source).read_bytes()
        ).hexdigest(),
        "fixtures": len(fixtures),
        "cold_repeats": 5,
        "status_counts": counts,
        "correct_decisive_or_finite_test": correct,
        "latency_ms": {
            name: {
                "median": statistics.median(values),
                "p95": sorted(values)[int(0.95 * (len(values) - 1))],
                "max": max(values),
            }
            for name, values in times.items()
        },
        "rows": rows,
        "scope_note": "Development scalar-expression cohort; previous TESTED-PASS is finite testing, new VERIFIED requires exact replay. No global superiority claim.",
    }


def external_evaluation(path):
    """Run the externally authored cases without modifying their original labels."""
    from collections import Counter
    from math_checker import MathCheckerEngine, verify_certificate

    raw = pathlib.Path(path).read_bytes()
    corpus = json.loads(raw)
    engine = MathCheckerEngine()
    rows = []
    elapsed = []
    for case in corpus["cases"]:
        result = engine.check(case["claim"])
        elapsed.append(result.elapsed_ms)
        expected = case["expected"]
        allowed = expected if isinstance(expected, list) else [expected]
        rows.append(
            {
                "id": case["id"],
                "expected": expected,
                "actual": result.status,
                "exact_status_match": result.status in allowed,
                "certificate_replayed": (
                    None
                    if result.certificate is None
                    else verify_certificate(case["claim"], result.certificate)
                ),
                "reasons": result.reasons,
            }
        )
    return {
        "source_sha256": hashlib.sha256(
            pathlib.Path(__file__).with_name("math_checker.py").read_bytes()
        ).hexdigest(),
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "cases": len(rows),
        "exact_status_matches": sum(r["exact_status_match"] for r in rows),
        "status_counts": dict(Counter(r["actual"] for r in rows)),
        "certificates_replayed": sum(r["certificate_replayed"] is True for r in rows),
        "certificate_failures": sum(r["certificate_replayed"] is False for r in rows),
        "latency_ms": {
            "median": statistics.median(elapsed),
            "p95": sorted(elapsed)[int(0.95 * (len(elapsed) - 1))],
            "max": max(elapsed),
        },
        "rows": rows,
    }


def cache_memory_demo(count=1000):
    """Separate allocation measurement, not a latency benchmark or an RSS cap."""
    import tracemalloc
    from math_checker import MathCheckerEngine

    tracemalloc.start()
    engine = MathCheckerEngine()
    for k in range(count):
        c = scalar(f"(x+{k})^2", "==", f"x*x+{2*k}*x+{k*k}")
        if engine.check(c).status != "VERIFIED":
            raise AssertionError("streamed identity failed")
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "unique_claims": count,
        "cache_entries": len(engine._cache),
        "cache_entry_limit": engine.limits.cache_entries,
        "python_current_bytes": current,
        "python_peak_bytes": peak,
        "scope": "tracemalloc Python allocations; not total process RSS or a hard memory isolation guarantee",
    }


if __name__ == "__main__":
    if "--generator-demo" in sys.argv:
        p = sys.argv.index("--generator-demo")
        print(json.dumps(generator_demo(sys.argv[p + 1]), indent=2))
    elif "--compare" in sys.argv:
        p = sys.argv.index("--compare")
        print(json.dumps(matched_text_comparison(sys.argv[p + 1]), indent=2))
    elif "--external" in sys.argv:
        p = sys.argv.index("--external")
        print(json.dumps(external_evaluation(sys.argv[p + 1]), indent=2))
    elif "--memory-demo" in sys.argv:
        print(json.dumps(cache_memory_demo(), indent=2))
    else:
        unittest.main()
