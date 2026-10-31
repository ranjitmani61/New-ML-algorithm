"""Executable contracts for actual optimizer-core filtering, without ML fitting."""

import copy
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
import random
import platform
import statistics
from pathlib import Path
import sys
import unittest
import time

ROOT = Path(__file__).resolve().parent
GENERATOR = Path(
    os.environ.get(
        "MATH_GATE_GENERATOR",
        ROOT.parent / "generator_only" / "hypothesis_generator.py",
    )
)
VERIFIER = Path(
    os.environ.get(
        "MATH_GATE_VERIFIER", ROOT.parent / "math_verifier" / "math_checker.py"
    )
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class UpdateContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = load(GENERATOR, "update_test_generator")

    def setUp(self):
        self.assertTrue(
            (ROOT / "math_filters.py").exists(), "missing update-rule filter"
        )
        self.api = load(ROOT / "math_filters.py", "update_under_test")
        self.assertTrue(
            hasattr(self.api.MathFilterEngine, "certify_transition"),
            "missing actual-update certification",
        )
        self.engine = self.api.MathFilterEngine(GENERATOR, VERIFIER)

    def record(
        self,
        loss="squared",
        mechanism="clip_step",
        preconditioner="identity",
        coupling="coupled",
        controller=None,
        kind="regression",
        aggregation="mean",
    ):
        g = self.gen
        c = controller or g.Controller.make(
            "mean",
            g.Controller("signal", name="energy"),
            g.Controller("signal", name="dispersion"),
        )
        a = g.AlgorithmSpec(
            kind,
            "linear",
            "standard",
            loss,
            mechanism,
            aggregation,
            preconditioner,
            c,
            coupling,
        )
        return g.Hypothesis(a, g.TaskSpec(kind, 1), "construct", (), 7, 1).to_dict()

    def state(self, gradients=None, t=0):
        return {
            "t": t,
            "theta": ["0", "0"],
            "fast": ["0", "0"],
            "example_gradients": gradients or [["0", "1/4"], ["0", "1/4"]],
            "signals": {
                s: ("0" if s != "time" else str(self.api.F(t, 159)))
                for s in self.api.SIGNALS
            },
            "smoothness_bound": "1",
        }

    def test_convexity_is_diagnostic_by_default(self):
        self.assertFalse(self.api.Policy().require_convex_objective)
        for loss, kind in [
            ("cauchy", "regression"),
            ("brier", "classification"),
            ("focal", "classification"),
        ]:
            result = self.engine.assess(self.record(loss, kind=kind))
            self.assertEqual(result["objective"]["status"], "NONCONVEX")
            self.assertEqual(result["update"]["envelope"]["status"], "PASS", result)
            self.assertEqual(result["mathematical_status"], "PASS", result)

    def test_same_convex_loss_can_fail_for_unstable_update(self):
        result = self.engine.assess(self.record(mechanism="clip_memory"))
        self.assertEqual(result["objective"]["status"], "CONVEX")
        self.assertEqual(result["update"]["envelope"]["status"], "FAIL")
        self.assertEqual(result["status"], "REJECTED")

    def test_runtime_changes_decision_for_same_nonconvex_algorithm(self):
        record = self.record("cauchy", mechanism="clip_memory")
        unsafe = self.engine.assess(record)
        safe = self.engine.assess(record, runtime={"learning_rate": "1/1000"})
        self.assertEqual(unsafe["update"]["envelope"]["status"], "FAIL")
        self.assertEqual(safe["update"]["envelope"]["status"], "PASS")

    def test_exact_runtime_canonicalization_roundtrips_small_rates(self):
        from dataclasses import asdict

        runtime = self.api.Runtime.parse({"learning_rate": "1e-100"})
        self.assertEqual(self.api.Runtime.parse(asdict(runtime)), runtime)
        audit = self.engine.assess(self.record(preconditioner="rms"), runtime=runtime)
        self.assertEqual(audit["update"]["envelope"]["status"], "PASS")

    def test_controller_range_changes_update_stability(self):
        g = self.gen
        c = g.Controller.make(
            "mean",
            g.Controller("signal", name="energy"),
            g.Controller("signal", name="dispersion"),
        )
        for _ in range(3):
            c = g.Controller.make("mean", g.Controller("half"), c)
        wide = self.engine.assess(self.record(mechanism="clip_memory"))
        narrow = self.engine.assess(self.record(mechanism="clip_memory", controller=c))
        self.assertEqual(wide["update"]["envelope"]["status"], "FAIL")
        self.assertEqual(narrow["update"]["envelope"]["status"], "PASS", narrow)

    def test_rms_and_blended_rms_are_not_interchangeable(self):
        raw = self.engine.assess(self.record(preconditioner="rms"))
        blend = self.engine.assess(
            self.record(
                mechanism="variance_penalty",
                preconditioner="rms",
                coupling="first_only",
            )
        )
        self.assertEqual(raw["update"]["envelope"]["status"], "FAIL")
        self.assertEqual(blend["update"]["envelope"]["status"], "PASS")

    def test_dual_memory_uses_cubic_jury_not_single_memory_formula(self):
        record = self.record(mechanism="dual_memory")
        result = self.engine.assess(record)
        self.assertEqual(result["update"]["envelope"]["model"], "dual_memory_cubic")
        self.assertEqual(result["update"]["envelope"]["status"], "PASS")
        self.assertEqual(
            self.engine.assess(record, policy={"curvature_upper": "100"})["update"][
                "envelope"
            ]["status"],
            "FAIL",
        )

    def test_schur_boundary_is_not_strict_stability(self):
        result = self.engine.assess(
            self.record(mechanism="clip_memory"), runtime={"learning_rate": "1/25"}
        )
        self.assertEqual(result["update"]["envelope"]["status"], "FAIL")

    def test_strict_maximal_certifiable_rate_and_recommendation(self):
        record = self.record(mechanism="clip_memory")
        audit = self.engine.assess(record)
        envelope = audit["update"]["envelope"]
        self.assertEqual(envelope["max_certifiable_rate"], "1/25")
        self.assertTrue(envelope["rate_boundary"]["strict"])
        self.assertEqual(
            envelope["rate_boundary"]["minimum_certificate"]["minimum"], "0"
        )
        recommended = envelope["recommended_certified_rate"]
        self.assertLess(
            self.api.F(recommended), self.api.F(envelope["max_certifiable_rate"])
        )
        safe = self.engine.assess(record, runtime={"learning_rate": recommended})
        self.assertEqual(safe["update"]["envelope"]["status"], "PASS")

    def test_missing_required_optimizer_buffers_are_value_errors(self):
        for record in (
            self.record(mechanism="dual_memory"),
            self.record(preconditioner="rms"),
        ):
            with self.assertRaises(ValueError):
                self.engine.certify_transition(record, self.state())

    def test_rms_energy_proof_handles_adaptive_beta_and_dual_memory(self):
        for mechanism in self.gen.MECHANISMS:
            result = self.engine.assess(
                self.record(mechanism=mechanism, preconditioner="rms"),
                runtime={"learning_rate": "1/1000000000"},
            )
            invariants = result["update"]["invariants"]
            self.assertEqual(invariants["status"], "PASS")
            self.assertEqual(
                invariants["rms_memory_bound"],
                "m_j^2 <= 20*v_j from zero initialization",
            )
            self.assertTrue(
                all(
                    x["status"] == "VERIFIED"
                    for x in result["math_checks"]
                    if x["name"].startswith("update/")
                )
            )

    def test_quadratic_minimum_checks_interior_vertex(self):
        result = self.api.quadratic_minimum(["1/5", "-1", "1"], "0", "1")
        self.assertEqual(result["minimum"], "-1/20")
        self.assertEqual(result["argument"], "1/2")
        self.assertTrue(
            self.api.verify_quadratic_minimum(["1/5", "-1", "1"], "0", "1", result)
        )
        result["minimum"] = "1/5"
        self.assertFalse(
            self.api.verify_quadratic_minimum(["1/5", "-1", "1"], "0", "1", result)
        )

    def test_actual_transition_has_rigorous_descent_bound(self):
        record = self.record("cauchy")
        state = self.state()
        result = self.engine.certify_transition(record, state)
        self.assertEqual(result["status"], "PROVED_DESCENT", result)
        self.assertTrue(self.engine.verify_transition(record, state, result))
        self.assertIn("supplied", result["scope"].lower())

    def test_clipping_can_reverse_mean_gradient(self):
        record = self.record(mechanism="clip_memory")
        state = self.state([["0", "4"], ["0", "-1"], ["0", "-1"]])
        state["signals"]["energy"] = state["signals"]["dispersion"] = "1"
        result = self.engine.certify_transition(record, state)
        self.assertEqual(result["status"], "PROVED_NONDESCENT_DIRECTION", result)

    def test_momentum_can_reverse_current_gradient(self):
        record = self.record(mechanism="memory_step")
        state = self.state([["0", "1"], ["0", "1"]], t=1)
        state["signals"]["energy"] = state["signals"]["dispersion"] = "1"
        state["fast"] = ["0", "-1"]
        result = self.engine.certify_transition(record, state)
        self.assertEqual(result["status"], "PROVED_NONDESCENT_DIRECTION", result)

    def test_alignment_is_checked_after_preconditioning(self):
        record = self.record(mechanism="memory_step", preconditioner="rms")
        state = self.state([["1", "1"], ["1", "1"]], t=1)
        state["signals"]["energy"] = state["signals"]["dispersion"] = "1"
        state["fast"] = ["39/19", "-21/19"]
        state["variance"] = ["1999/19", "1"]
        result = self.engine.certify_transition(record, state)
        self.assertEqual(result["status"], "PROVED_NONDESCENT_DIRECTION", result)

    def test_intercept_penalty_mask_matches_reference(self):
        record = self.record(mechanism="penalty_step")
        state = self.state()
        state["theta"] = ["2", "2"]
        result = self.engine.certify_transition(record, state)
        self.assertEqual(result["regularizer_mask"], [1, 0])

    def test_quantile_is_of_norms_not_square_root_of_quantile(self):
        state = self.state([["0", "0"], ["0", "1"], ["0", "2"], ["0", "3"]])
        state["signals"]["energy"] = state["signals"]["dispersion"] = "1"
        receipt = self.engine.certify_transition(self.record(), state)
        self.assertEqual(receipt["clip_threshold"], {"lo": "9/16", "hi": "9/16"})

    def test_optimizer_core_matches_snapshots_of_the_pinned_executor(self):
        # Public toy values exercise code paths, not predictive performance.
        # The float comparison is a conformance diagnostic; it is not used as
        # mathematical proof by either admission or transition certification.
        g = self.engine.gen
        np = g.np
        lines = GENERATOR.read_text().splitlines()
        begin = next(
            i + 1
            for i, s in enumerate(lines)
            if "losses,residual=loss_and_residual" in s
        )
        end = next(
            i + 1
            for i, s in enumerate(lines)
            if "previous=raw;previous_loss=mean_loss" in s
        )
        x = np.array([[-2.0], [-1.0], [-0.2], [0.1], [0.7], [1.3], [2.1]])
        y = np.array([-1.0, 0.5, -0.2, 0.9, -0.4, 0.8, -0.6])
        partition = g.Partition(
            "train", x, y, tuple("toy" + str(i) for i in range(len(x)))
        )
        observations = 0
        for feature in g.MAPS:
            for mechanism in g.MECHANISMS:
                for preconditioner in ("identity", "rms"):
                    if mechanism == "variance_penalty" and preconditioner != "rms":
                        continue
                    for aggregation in ("mean", "median_of_means"):
                        coupling = "coupled" if aggregation == "mean" else "first_only"
                        controller = g.Controller.make(
                            "mean",
                            g.Controller("signal", name="energy"),
                            g.Controller("signal", name="dispersion"),
                        )
                        a = g.AlgorithmSpec(
                            "regression",
                            feature,
                            "standard",
                            "cauchy",
                            mechanism,
                            aggregation,
                            preconditioner,
                            controller,
                            coupling,
                        )
                        h = g.Hypothesis(
                            a, g.TaskSpec("regression", 1), "construct", (), 7, 1
                        )
                        model = g.GeneratedModel(
                            h, steps=2, learning_rate=0.08, seed=11
                        )
                        previous = {}
                        snapshots = []

                        def trace(frame, event, arg):
                            if (
                                event != "line"
                                or frame.f_code.co_name != "fit"
                                or Path(frame.f_code.co_filename).resolve()
                                != GENERATOR.resolve()
                            ):
                                return trace
                            v = frame.f_locals
                            if frame.f_lineno == begin:
                                previous.clear()
                                previous.update(
                                    {
                                        k: None if v[k] is None else v[k].copy()
                                        for k in ("theta", "fast", "slow", "v")
                                    }
                                )
                            elif frame.f_lineno == end:
                                gradients = np.einsum(
                                    "ni,nj->nij", v["z"], v["residual"]
                                ).reshape(len(x), -1)
                                state = {
                                    "t": int(v["t"]),
                                    "theta": list(map(str, previous["theta"].ravel())),
                                    "fast": list(map(str, previous["fast"].ravel())),
                                    "example_gradients": [
                                        [str(z) for z in row] for row in gradients
                                    ],
                                    "signals": {
                                        s: str(v["signals"][s])
                                        for s in self.api.SIGNALS
                                    },
                                    "smoothness_bound": "1000000",
                                }
                                state["signals"]["time"] = str(v["t"])
                                if previous["slow"] is not None:
                                    state["slow"] = list(
                                        map(str, previous["slow"].ravel())
                                    )
                                if previous["v"] is not None:
                                    state["variance"] = list(
                                        map(str, previous["v"].ravel())
                                    )
                                snapshots.append(
                                    (
                                        state,
                                        {
                                            k: None if v[k] is None else v[k].copy()
                                            for k in ("theta", "fast", "slow", "v")
                                        },
                                    )
                                )
                            return trace

                        old_trace = sys.gettrace()
                        try:
                            sys.settrace(trace)
                            model.fit(partition)
                        finally:
                            sys.settrace(old_trace)
                        self.assertEqual(len(snapshots), 2)
                        for state, reference in snapshots:
                            receipt = self.engine.certify_transition(
                                h.to_dict(), state, runtime={"steps": 2, "seed": 11}
                            )
                            for field, key in [
                                ("theta_next", "theta"),
                                ("fast_next", "fast"),
                                ("slow_next", "slow"),
                                ("variance_next", "v"),
                            ]:
                                if reference[key] is None:
                                    self.assertIsNone(receipt[field])
                                    continue
                                values = [
                                    float(
                                        (self.api.F(b["lo"]) + self.api.F(b["hi"])) / 2
                                    )
                                    for b in receipt[field]
                                ]
                                np.testing.assert_allclose(
                                    values,
                                    reference[key].ravel(),
                                    rtol=5e-11,
                                    atol=5e-12,
                                )
                            observations += 1
        self.assertEqual(observations, 132)

    def test_rms_requires_nonnegative_variance(self):
        state = self.state()
        state["variance"] = ["-1", "0"]
        with self.assertRaises(ValueError):
            self.engine.certify_transition(self.record(preconditioner="rms"), state)

    def test_rational_interval_square_roots_are_outward(self):
        rng = random.Random(901)
        for _ in range(300):
            x = self.api.F(rng.randrange(0, 10000), rng.randrange(1, 1000))
            lo, hi = self.api.sqrt_rational(x)
            self.assertLessEqual(lo * lo, x)
            self.assertGreaterEqual(hi * hi, x)
            self.assertLessEqual(hi - lo, self.api.F(1, 1 << 96))

    def test_adaptive_rms_energy_invariant_on_exact_sequences(self):
        rng = random.Random(143)
        F = self.api.F
        for _ in range(10):
            fast = slow = variance = F(0)
            for _ in range(40):
                beta = F(rng.randrange(20), 20)
                g = F(rng.randrange(-100, 101), 7)
                q = F(rng.randrange(11), 10)
                fast = beta * fast + (1 - beta) * g
                slow = F(19, 20) * slow + F(1, 20) * g
                variance = F(19, 20) * variance + F(1, 20) * g * g
                mix = (1 - q) * fast + q * slow
                self.assertLessEqual(fast * fast, 20 * variance)
                self.assertLessEqual(slow * slow, 20 * variance)
                self.assertLessEqual(mix * mix, 20 * variance)

    def test_dual_jury_formula_matches_independent_eigenvalues(self):
        np = self.gen.np
        rng = random.Random(332)
        for _ in range(100):
            q = rng.random()
            gain = 12 * rng.random()
            matrix = np.array(
                [
                    [
                        1 - gain * ((1 - q) / 2 + q / 20),
                        -gain * (1 - q) / 2,
                        -gain * q * 19 / 20,
                    ],
                    [1 / 2, 1 / 2, 0],
                    [1 / 20, 0, 19 / 20],
                ]
            )
            stable = max(abs(np.linalg.eigvals(matrix))) < 1
            expected = 0 < gain < 234 / (39 - 36 * q)
            self.assertEqual(stable, expected)

    def test_frozen_stability_is_not_promoted_to_switching_convergence(self):
        record = self.record(mechanism="memory_step")
        audit = self.engine.assess(
            record, runtime={"learning_rate": "1"}, policy={"curvature_upper": "19/10"}
        )
        self.assertEqual(audit["update"]["envelope"]["status"], "PASS")
        self.assertEqual(audit["update"]["full_algorithm_convergence"], "UNVERIFIED")
        F = self.api.F
        A0 = [[F(-9, 10), F(0)], [F(19, 10), F(0)]]
        A1 = [[F(781, 800), F(-19, 80)], [F(19, 200), F(19, 20)]]
        product = [
            [sum(A1[i][k] * A0[k][j] for k in range(2)) for j in range(2)]
            for i in range(2)
        ]
        self.assertEqual(product[0][0] + product[1][1], F(-10639, 8000))
        self.assertLess(product[0][0] + product[1][1], -1)

    def test_time_signal_cannot_be_forged(self):
        state = self.state()
        state["signals"]["time"] = "1"
        with self.assertRaises(ValueError):
            self.engine.certify_transition(self.record(), state)

    def test_bad_state_and_dimensions_fail_closed(self):
        for field, value in [
            ("fast", ["0"]),
            ("theta", [True, "0"]),
            ("t", True),
            ("smoothness_bound", "0"),
        ]:
            state = self.state()
            state[field] = value
            with self.assertRaises(ValueError):
                self.engine.certify_transition(self.record(), state)

    def test_transition_receipt_binds_every_input_and_output(self):
        record = self.record()
        state = self.state()
        receipt = self.engine.certify_transition(record, state)
        for field in ("status", "q", "step", "runtime", "sources"):
            bad = copy.deepcopy(receipt)
            bad[field] = "forged"
            self.assertFalse(self.engine.verify_transition(record, state, bad))
        state["example_gradients"][0][1] = "1/3"
        self.assertFalse(self.engine.verify_transition(record, state, receipt))

    def test_integrity_and_science_status_are_preserved(self):
        record = self.record()
        record["acceptance_gates"]["mathematical_foundation"] = "PASS"
        self.assertEqual(self.engine.assess(record)["status"], "REJECTED")
        batch = self.engine.run(
            {"kind": "regression", "features": 1}, count=24, seed=42
        )
        ready = [
            x
            for x in batch["results"]
            if x["status"] == "READY_FOR_SCIENTIFIC_VALIDATION"
        ]
        self.assertTrue(ready)
        for item in ready:
            self.assertTrue(
                all(
                    x == "UNVERIFIED"
                    for x in item["candidate"]["acceptance_gates"].values()
                )
            )
            self.assertEqual(
                self.engine.handoff(item["candidate"], item["audit"])[
                    "scientific_status"
                ],
                "NOT_EVALUATED",
            )


def external_evaluation(path):
    """Original unchanged challenges; final run is disclosed regression testing."""
    api = load(ROOT / "math_filters.py", "update_external_runner")
    engine = api.MathFilterEngine(GENERATOR, VERIFIER)
    raw = Path(path).read_bytes()
    cases = json.loads(raw)["cases"]
    rows = []
    for case in cases:
        expected = case["expected"]
        runtime = case.get("runtime")
        record = case["record"]
        row = {"id": case["id"], "operation": case["operation"]}
        try:
            if case["operation"] == "assess":
                result = engine.assess(
                    record, runtime=runtime, policy=case.get("policy")
                )
                envelope = result.get("update", {}).get("envelope", {})
                row.update(
                    envelope_status=envelope.get("status"),
                    max_certifiable_rate=envelope.get("max_certifiable_rate"),
                    integrity_status=result["integrity"]["status"],
                )
                matched = all(row.get(key) == value for key, value in expected.items())
            else:
                result = engine.certify_transition(
                    record, case["state"], runtime=runtime
                )
                row["transition_status"] = result["status"]
                row["verify_original"] = engine.verify_transition(
                    record, case["state"], result, runtime=runtime
                )
                if case["operation"] == "tamper_transition":
                    arguments = {
                        "record": copy.deepcopy(record),
                        "state": copy.deepcopy(case["state"]),
                        "runtime": copy.deepcopy(runtime),
                        "receipt": copy.deepcopy(result),
                    }
                    mutation = case["mutation"]
                    target = arguments[mutation["component"]]
                    for key in mutation["path"][:-1]:
                        target = target[key]
                    target[mutation["path"][-1]] = mutation["value"]
                    row["verify_tampered"] = engine.verify_transition(
                        arguments["record"],
                        arguments["state"],
                        arguments["receipt"],
                        runtime=arguments["runtime"],
                    )
                matched = not expected.get("rejected", False) and all(
                    row.get(k) == v for k, v in expected.items()
                )
        except ValueError as exc:
            row.update(rejected=True, error=str(exc))
            matched = expected.get("rejected", False)
        except Exception as exc:
            row["unexpected_error"] = type(exc).__name__ + ": " + str(exc)
            matched = False
        row["matched"] = matched
        rows.append(row)
    return {
        "core_sha256": api.file_hash(ROOT / "math_filters.py"),
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "cases": len(rows),
        "matched": sum(r["matched"] for r in rows),
        "operations": dict(Counter(r["operation"] for r in rows)),
        "phase": "post-exposure regression after documented API corrections",
        "rows": rows,
    }


def demonstrate(count=100):
    """Metadata-only autonomous admission, paired against the unchanged v1 gate."""
    api = load(ROOT / "math_filters.py", "update_demo_runner")
    old_path = ROOT.parent / "math_gate" / "math_filters.py"
    old = None
    if old_path.exists():
        assert (
            api.file_hash(old_path)
            == "7aee339f092dfc64823979cd2ef74d0e6d48f80d810992668d76a053feed66b0"
        )
        old = load(old_path, "legacy_admission_comparison").MathFilterEngine(
            GENERATOR, VERIFIER
        )
    rows = []
    statuses = Counter()
    by_loss = {}
    paired = Counter()
    all_times = []
    checks = Counter()
    digests = []
    for kind in ("regression", "classification"):
        for seed in (7, 19, 42, 2026):
            engine = api.MathFilterEngine(GENERATOR, VERIFIER)
            original = engine.assess
            elapsed = []

            def timed(*args, **kwargs):
                start = time.perf_counter()
                result = original(*args, **kwargs)
                elapsed.append((time.perf_counter() - start) * 1000)
                return result

            engine.assess = timed
            task = {"kind": kind, "features": 8}
            if kind == "classification":
                task["classes"] = 3
            start = time.perf_counter()
            batch = engine.run(task, count=count, seed=seed)
            seconds = time.perf_counter() - start
            admission_times = list(elapsed)
            all_times.extend(admission_times)
            handoffs = 0
            nonconvex_handoffs = 0
            row_paired = Counter()
            for item in batch["results"]:
                record, audit = item["candidate"], item["audit"]
                ready = item["status"] == "READY_FOR_SCIENTIFIC_VALIDATION"
                loss = record["algorithm"]["loss"]
                by_loss.setdefault(loss, Counter()).update([item["status"]])
                checks.update(c["status"] for c in audit["math_checks"])
                assert all(
                    v == "UNVERIFIED" for v in record["acceptance_gates"].values()
                )
                if ready:
                    packet = engine.handoff(record, audit)
                    assert packet["scientific_status"] == "NOT_EVALUATED"
                    handoffs += 1
                    nonconvex_handoffs += audit["objective"]["status"] == "NONCONVEX"
                else:
                    try:
                        engine.handoff(record, audit)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("non-admitted candidate handed off")
                if old is not None:
                    old_ready = old.assess(record)["mathematical_status"] == "PASS"
                    row_paired[f"v1_{old_ready}_v2_{ready}"] += 1
                digests.append([audit["record_hash"], item["status"], audit["reasons"]])
            statuses.update(batch["status_counts"])
            paired.update(row_paired)
            row = {
                "kind": kind,
                "seed": seed,
                "generated": count,
                "statuses": batch["status_counts"],
                "handoffs": handoffs,
                "nonconvex_handoffs": nonconvex_handoffs,
                "paired": dict(row_paired),
                "generation_admission_seconds": seconds,
                "admission_ms_median": statistics.median(admission_times),
            }
            rows.append(row)
            print(json.dumps({"progress": row}), flush=True)
    return {
        "version": api.VERSION,
        "scope": "metadata, exact mathematics and verified local handoff; no predictive benchmark",
        "python": sys.version.split()[0],
        "numpy": engine.gen.np.__version__,
        "platform": platform.platform(),
        "sources": engine.sources,
        "policy": asdict(api.Policy()),
        "runtime": asdict(api.Runtime()),
        "generated": len(digests),
        "statuses": dict(statuses),
        "by_loss": {k: dict(v) for k, v in by_loss.items()},
        "handoffs": sum(r["handoffs"] for r in rows),
        "nonconvex_handoffs": sum(r["nonconvex_handoffs"] for r in rows),
        "paired": dict(paired),
        "proof_statuses": dict(checks),
        "admission_ms_median": statistics.median(all_times),
        "admission_ms_p95": sorted(all_times)[int(0.95 * (len(all_times) - 1))],
        "decision_digest": api.fingerprint(digests),
        "rows": rows,
    }


if __name__ == "__main__":
    if "--external" in sys.argv:
        result = external_evaluation(ROOT / "independent_cases.json")
        print(json.dumps(result, indent=2))
    elif "--demonstrate" in sys.argv:
        result = demonstrate()
        if "--output" in sys.argv:
            Path(sys.argv[sys.argv.index("--output") + 1]).write_text(
                json.dumps(result, indent=2)
            )
        print(json.dumps(result, indent=2))
    else:
        unittest.main()
