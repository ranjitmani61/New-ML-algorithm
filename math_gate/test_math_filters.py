"""Contracts and demonstrations for the four-file mathematical admission layer."""

import copy
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import platform
import statistics
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parent
GENERATOR = pathlib.Path(
    os.environ.get(
        "MATH_GATE_GENERATOR",
        ROOT.parent / "generator_only" / "hypothesis_generator.py",
    )
)
VERIFIER = pathlib.Path(
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


class AdmissionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = load(GENERATOR, "gate_test_generator")

    def setUp(self):
        self.assertTrue(
            (ROOT / "math_filters.py").exists(),
            "missing integrity/filter implementation",
        )
        self.api = load(ROOT / "math_filters.py", "gate_under_test")
        self.assertTrue(
            hasattr(self.api, "MathFilterEngine"),
            "missing autonomous mathematical admission API",
        )
        self.engine = self.api.MathFilterEngine(GENERATOR, VERIFIER)

    def record(self, loss="squared", kind="regression", controller=None, classes=None):
        g = self.gen
        controller = controller or g.Controller.make(
            "mean",
            g.Controller("signal", name="energy"),
            g.Controller("signal", name="dispersion"),
        )
        a = g.AlgorithmSpec(
            kind,
            "linear",
            "standard",
            loss,
            "clip_memory",
            "mean",
            "identity",
            controller,
        )
        return g.Hypothesis(
            a, g.TaskSpec(kind, 4, classes=classes), "construct", (), 7, 1
        ).to_dict()

    def test_forged_scientific_gate_is_rejected(self):
        record = self.record()
        record["acceptance_gates"]["mathematical_foundation"] = "PASS"
        result = self.engine.assess(record)
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("RECORD_INTEGRITY", result["reasons"])
        self.assertEqual(result["math_checks"], [])

    def test_forged_equation_is_rejected_before_math(self):
        record = self.record()
        record["equations"][0] = "theta_next=theta+gradient"
        self.assertEqual(self.engine.assess(record)["integrity"]["status"], "REJECTED")

    def test_full_record_roundtrip_and_unattested_origin(self):
        record = self.record()
        result = self.engine.assess(record)
        self.assertEqual(result["integrity"]["status"], "PASS")
        self.assertEqual(result["mathematical_status"], "PASS")
        self.assertEqual(result["status"], "PENDING")
        self.assertIn("UNATTESTED_ORIGIN", result["reasons"])
        self.assertEqual(
            record["acceptance_gates"], {k: "UNVERIFIED" for k in self.gen.GATES}
        )

    def test_cauchy_negative_curvature_is_substantive_rejection(self):
        result = self.engine.assess(self.record("cauchy"))
        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(result["objective"]["status"], "NONCONVEX")
        self.assertEqual(result["objective"]["witness"]["curvature"], "-3/25")
        self.assertTrue(result["objective"]["witness"]["exact"])

    def test_brier_convex_in_probability_is_not_convex_in_logits(self):
        for classes in (2, 3, 7):
            result = self.engine.assess(
                self.record("brier", "classification", classes=classes)
            )
            self.assertEqual(result["objective"]["status"], "NONCONVEX")
            self.assertEqual(result["status"], "REJECTED")

    def test_focal_uses_rigorous_bound_not_float_sign_test(self):
        result = self.engine.assess(self.record("focal", "classification"))
        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(result["objective"]["witness"]["p"], "1/32")
        self.assertTrue(result["objective"]["witness"]["exact"])
        self.assertTrue(any("log2" in c["name"] for c in result["math_checks"]))

    def test_all_supported_convex_losses_discharge_obligations(self):
        for kind, loss in [
            ("regression", "squared"),
            ("regression", "huber"),
            ("regression", "pseudo_huber"),
            ("classification", "cross_entropy"),
        ]:
            result = self.engine.assess(self.record(loss, kind))
            self.assertEqual(result["objective"]["status"], "CONVEX", result)
            self.assertEqual(result["mathematical_status"], "PASS", result)
            self.assertTrue(
                all(c["status"] == "VERIFIED" for c in result["math_checks"])
            )

    def test_large_class_count_uses_dimension_independent_convexity_lemma(self):
        result = self.engine.assess(
            self.record("cross_entropy", "classification", classes=256)
        )
        self.assertEqual(result["objective"]["status"], "CONVEX")
        self.assertEqual(result["mathematical_status"], "PASS")

    def test_exact_constant_controller_is_rejected(self):
        g = self.gen
        a = g.Controller("signal", name="energy")
        b = g.Controller("signal", name="dispersion")
        c = g.Controller.make(
            "mean",
            g.Controller.make("mean", a, g.Controller.make("complement", a)),
            g.Controller.make("mean", b, g.Controller.make("complement", b)),
        )
        result = self.engine.assess(self.record(controller=c))
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("CONSTANT_CONTROLLER", result["reasons"])

    def test_time_is_not_counted_as_training_feedback(self):
        g = self.gen
        c = g.Controller.make(
            "mean",
            g.Controller("signal", name="time"),
            g.Controller("signal", name="energy"),
        )
        result = self.engine.assess(self.record(controller=c))
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("INSUFFICIENT_TRAINING_FEEDBACK", result["reasons"])

    def test_feedback_witnesses_change_only_one_training_signal(self):
        result = self.engine.assess(self.record())
        self.assertEqual(result["feedback"]["status"], "PASS")
        for w in result["feedback"]["witnesses"]:
            changes = [s for s in w["low"] if w["low"][s] != w["high"][s]]
            self.assertEqual(changes, [w["signal"]])
            self.assertNotEqual(w["signal"], "time")

    def test_unknown_fields_and_bad_runtime_fail_closed(self):
        record = self.record()
        record["external_test_accuracy"] = 1
        self.assertEqual(self.engine.assess(record)["status"], "REJECTED")
        for runtime in (
            {"steps": True},
            {"steps": 0},
            {"learning_rate": "0"},
            {"seed": -1},
            {"test_scores": [1]},
        ):
            self.assertEqual(
                self.engine.assess(self.record(), runtime=runtime)["status"], "REJECTED"
            )

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            self.api.read_json(
                '{"schema_version":"generator-3.0","schema_version":"forged"}'
            )

    def test_numeric_exponents_are_bounded_before_fraction_construction(self):
        with patch.object(
            self.api, "F", side_effect=AssertionError("unbounded Fraction conversion")
        ):
            with self.assertRaises(ValueError):
                self.api.Runtime.parse({"learning_rate": "1e99999999"})
        with self.assertRaises(ValueError):
            self.api.read_json('{"value":1e9999}')

    def test_rehashed_weaker_policy_or_runtime_cannot_reuse_issued_admission(self):
        batch = self.engine.run(
            {"kind": "regression", "features": 4}, count=12, seed=19
        )
        item = next(
            x
            for x in batch["results"]
            if x["status"] == "READY_FOR_SCIENTIFIC_VALIDATION"
        )
        for field, key, value in [
            ("policy", "min_feedback_signals", 1),
            ("runtime", "steps", 200),
        ]:
            bad = copy.deepcopy(item["audit"])
            bad[field][key] = value
            bad["receipt_hash"] = self.api.fingerprint(
                {k: v for k, v in bad.items() if k != "receipt_hash"}
            )
            self.assertFalse(self.engine.verify_receipt(item["candidate"], bad))
            with self.assertRaises(ValueError):
                self.engine.handoff(item["candidate"], bad)

    def test_rehashed_feedback_and_source_tampering_are_rejected(self):
        result = self.engine.assess(self.record())
        for kind in ("feedback", "sources", "objective", "record_hash"):
            bad = copy.deepcopy(result)
            if kind == "feedback":
                bad[kind]["witnesses"][0]["q_high"] = "999"
            elif kind == "sources":
                bad[kind]["verifier"] = "f" * 64
            elif kind == "objective":
                bad[kind]["scheme"] = "invented convergence theorem"
            else:
                bad[kind] = "0" * 64
            bad["receipt_hash"] = self.api.fingerprint(
                {k: v for k, v in bad.items() if k != "receipt_hash"}
            )
            self.assertFalse(self.engine.verify_receipt(self.record(), bad))

    def test_nonlinear_frozen_features_preserve_convex_loss(self):
        g = self.gen
        for feature in ("linear", "diagonal_quadratic", "linear_fourier"):
            original = g.Hypothesis.from_dict(self.record())
            body = original.algorithm.to_dict()
            body["feature_map"] = feature
            a = g.AlgorithmSpec.from_dict(body)
            h = g.Hypothesis(a, original.task, "construct", (), 7, 1)
            self.assertEqual(
                self.engine.assess(h.to_dict())["objective"]["status"], "CONVEX"
            )

    def test_relaxed_policy_is_explicit_and_preserves_nonconvex_finding(self):
        result = self.engine.assess(
            self.record("cauchy"), policy={"require_convex_objective": False}
        )
        self.assertEqual(result["objective"]["status"], "NONCONVEX")
        self.assertEqual(result["mathematical_status"], "PASS")
        self.assertEqual(result["status"], "PENDING")
        self.assertFalse(result["policy"]["require_convex_objective"])

    def test_no_witness_is_pending_instead_of_false_rejection(self):
        # min(a,1-a) has variation 1/2; both data channels have variation <1
        # for this mean. Failure to meet the declared threshold is unresolved.
        g = self.gen
        a = g.Controller("signal", name="energy")
        b = g.Controller("signal", name="dispersion")
        c = g.Controller.make(
            "mean",
            g.Controller.make("minimum", a, g.Controller.make("complement", a)),
            b,
        )
        result = self.engine.assess(
            self.record(controller=c), policy={"min_influence": "1"}
        )
        self.assertEqual(result["feedback"]["status"], "PENDING")
        self.assertIn("INFLUENCE_WITNESS_UNRESOLVED", result["reasons"])

    def test_checker_omission_duplication_and_tautology_substitution_fail_closed(self):
        batch = self.engine.run(
            {"kind": "regression", "features": 4}, count=12, seed=19
        )
        item = next(
            x
            for x in batch["results"]
            if x["status"] == "READY_FOR_SCIENTIFIC_VALIDATION"
        )
        original = self.engine.checker.check
        for fault in ("omit", "duplicate", "tautology"):

            def broken(claim):
                result = original(claim)
                if (
                    isinstance(claim, dict)
                    and claim.get("schema_version") == "generator-3.0"
                ):
                    if fault == "omit":
                        result.subclaims = result.subclaims[1:]
                    elif fault == "duplicate":
                        result.subclaims = [
                            copy.deepcopy(result.subclaims[0]) for _ in result.subclaims
                        ]
                    else:
                        replacement = {
                            "kind": "scalar",
                            "variables": {},
                            "lhs": "1",
                            "op": "==",
                            "rhs": "1",
                        }
                        result.subclaims[0]["claim"] = replacement
                        result.subclaims[0]["verdict"] = original(replacement).to_dict()
                return result

            with patch.object(self.engine.checker, "check", side_effect=broken):
                audit = self.engine.assess(item["candidate"])
                self.assertNotEqual(
                    audit["status"], "READY_FOR_SCIENTIFIC_VALIDATION", fault
                )
                self.assertIn("BASE_OBLIGATION_SET_MISMATCH", audit["reasons"])
                with self.assertRaises(ValueError):
                    self.engine.handoff(item["candidate"], audit)

    def test_malformed_receipts_return_false_and_handoff_raises_value_error(self):
        for receipt in (None, [], {"integrity": []}, {"integrity": None}):
            with self.subTest(receipt=receipt):
                self.assertFalse(self.engine.verify_receipt(self.record(), receipt))
                with self.assertRaises(ValueError):
                    self.engine.handoff(self.record(), receipt)

    def test_loaded_code_and_source_file_must_match(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = pathlib.Path(scratch) / "math_filters.py"
            path.write_bytes((ROOT / "math_filters.py").read_bytes())
            loaded = load(path, "stale_gate_source_test")
            path.write_text(path.read_text() + "\n# changed after module load\n")
            engine = loaded.MathFilterEngine(GENERATOR, VERIFIER)
            self.assertEqual(engine.assess(self.record())["status"], "REJECTED")

    def test_nonmonotone_feedback_uses_interior_witnesses(self):
        g = self.gen
        a = g.Controller("signal", name="energy")
        b = g.Controller("signal", name="dispersion")
        c = g.Controller.make(
            "mean",
            g.Controller.make("minimum", a, g.Controller.make("complement", a)),
            b,
        )
        result = self.engine.assess(self.record(controller=c))
        self.assertEqual(result["feedback"]["status"], "PASS")
        self.assertEqual(
            {w["signal"] for w in result["feedback"]["witnesses"]},
            {"energy", "dispersion"},
        )

    def test_autonomous_run_selects_and_hands_off_without_upgrading_science(self):
        task = {"kind": "regression", "features": 4}
        batch = self.engine.run(task, count=30, seed=42)
        statuses = {x["status"] for x in batch["results"]}
        self.assertIn("READY_FOR_SCIENTIFIC_VALIDATION", statuses)
        self.assertIn("REJECTED", statuses)
        item = next(
            x
            for x in batch["results"]
            if x["status"] == "READY_FOR_SCIENTIFIC_VALIDATION"
        )
        self.assertTrue(self.engine.verify_receipt(item["candidate"], item["audit"]))
        handoff = self.engine.handoff(item["candidate"], item["audit"])
        self.assertEqual(handoff["scientific_status"], "NOT_EVALUATED")
        self.assertTrue(
            all(
                v == "UNVERIFIED"
                for v in handoff["candidate"]["acceptance_gates"].values()
            )
        )
        model = self.engine.build_model(item["candidate"], item["audit"])
        self.assertEqual(model.steps, item["audit"]["runtime"]["steps"])
        self.assertFalse(model._fitted)

    def test_receipt_binds_record_policy_runtime_and_certificates(self):
        result = self.engine.assess(self.record())
        self.assertTrue(self.engine.verify_receipt(self.record(), result))
        for key in ("runtime", "policy", "math_checks", "status"):
            bad = copy.deepcopy(result)
            if key == "runtime":
                bad[key]["steps"] += 1
            elif key == "policy":
                bad[key]["min_feedback_signals"] = 1
            elif key == "math_checks":
                bad[key][0]["certificate"]["claim_hash"] = "0" * 64
            else:
                bad[key] = "READY_FOR_SCIENTIFIC_VALIDATION"
            self.assertFalse(self.engine.verify_receipt(self.record(), bad))

    def test_unissued_parent_lineage_cannot_be_handed_off(self):
        g = self.gen
        record = self.record()
        h = g.Hypothesis.from_dict(record)
        forged = g.Hypothesis(
            h.algorithm, h.task, "mutate", ("f" * 64,), 7, 2
        ).to_dict()
        result = self.engine.assess(forged)
        self.assertEqual(result["status"], "PENDING")
        with self.assertRaises(ValueError):
            self.engine.handoff(forged, result)


def demonstrate(count=100):
    """Multiple-seed metadata-only selection and verified handoff demonstration."""
    api = load(ROOT / "math_filters.py", "admission_demonstration")
    rows = []
    totals = Counter()
    reasons = Counter()
    proofs = Counter()
    all_latencies = []
    runtime = api.Runtime()
    policy = api.Policy()
    digest_rows = []
    for kind in ("regression", "classification"):
        for seed in (7, 19, 42, 2026):
            engine = api.MathFilterEngine(GENERATOR, VERIFIER)
            real_assess = engine.assess
            latencies = []

            def timed(*args, **kwargs):
                start = time.perf_counter()
                answer = real_assess(*args, **kwargs)
                latencies.append((time.perf_counter() - start) * 1000)
                return answer

            engine.assess = timed
            task = {"kind": kind, "features": 8}
            if kind == "classification":
                task["classes"] = 3
            start = time.perf_counter()
            batch = engine.run(task, count=count, seed=seed)
            generation_seconds = time.perf_counter() - start
            admission_latencies = list(latencies)
            all_latencies.extend(admission_latencies)
            handoffs = 0
            baseline = 0
            row_proofs = Counter()
            negative = Counter()
            for item in batch["results"]:
                audit = item["audit"]
                candidate = item["candidate"]
                assert all(
                    v == "UNVERIFIED" for v in candidate["acceptance_gates"].values()
                )
                assert audit["integrity"]["status"] == "PASS"
                base = [
                    x for x in audit["math_checks"] if x["name"].startswith("base/")
                ]
                baseline += bool(base) and all(x["status"] == "VERIFIED" for x in base)
                row_proofs.update(x["status"] for x in audit["math_checks"])
                if audit["objective"]["status"] == "NONCONVEX":
                    negative[audit["objective"]["loss"]] += 1
                if item["status"] == "READY_FOR_SCIENTIFIC_VALIDATION":
                    handoff = engine.handoff(candidate, audit)
                    assert handoff["scientific_status"] == "NOT_EVALUATED"
                    handoffs += 1
                    if handoffs == 1:
                        model = engine.build_model(candidate, audit)
                        assert not model._fitted
                else:
                    try:
                        engine.handoff(candidate, audit)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("non-admitted record handed off")
                digest_rows.append(
                    [audit["record_hash"], item["status"], audit["reasons"]]
                )
            row = {
                "kind": kind,
                "seed": seed,
                "generated": count,
                "status_counts": batch["status_counts"],
                "rejection_counts": batch["rejection_counts"],
                "negative_curvature_counts": dict(negative),
                "previous_subclaim_only_passes": baseline,
                "actual_handoffs": handoffs,
                "proof_statuses": dict(row_proofs),
                "generation_and_admission_seconds": generation_seconds,
                "candidates_per_second": count / generation_seconds,
                "admission_ms_median": statistics.median(admission_latencies),
                "admission_ms_p95": sorted(admission_latencies)[
                    int(0.95 * (len(admission_latencies) - 1))
                ],
            }
            rows.append(row)
            totals.update(batch["status_counts"])
            reasons.update(batch["rejection_counts"])
            proofs.update(row_proofs)
            print(json.dumps({"progress": row}), flush=True)
    return {
        "version": api.VERSION,
        "scope": "metadata, exact mathematics and local session handoff; no training or scientific validation",
        "python": sys.version.split()[0],
        "numpy": engine.gen.np.__version__,
        "platform": platform.platform(),
        "sources": engine.sources,
        "policy": asdict(policy),
        "runtime": asdict(runtime),
        "generated": sum(r["generated"] for r in rows),
        "status_counts": dict(totals),
        "rejection_counts": dict(reasons),
        "proof_statuses": dict(proofs),
        "previous_subclaim_only_passes": sum(
            r["previous_subclaim_only_passes"] for r in rows
        ),
        "actual_handoffs": sum(r["actual_handoffs"] for r in rows),
        "admission_ms_median": statistics.median(all_latencies),
        "admission_ms_p95": sorted(all_latencies)[int(0.95 * (len(all_latencies) - 1))],
        "deterministic_decision_digest": api.fingerprint(digest_rows),
        "rows": rows,
    }


def external_evaluation(path):
    """Reproduce the independent reviewer's semantic adapter; preserve all labels.

    This is regression material after the disclosed three-fault correction.
    External records are never secretly inserted in the issuance ledger.
    Genuine generator issuance is used for fault injection and actual handoff.
    """
    api = load(ROOT / "math_filters.py", "admission_external_regression")
    raw = pathlib.Path(path).read_bytes()
    corpus = json.loads(raw)
    engine = api.MathFilterEngine(GENERATOR, VERIFIER)
    rows = []
    for case in corpus["cases"]:
        category = case["category"]
        row = {"id": case["id"], "category": category}
        if category == "verifier_fault":
            continue
        if category == "coupling_policy":
            row.update(
                matched=None,
                note="Out of contract: a mandatory second coupled branch is not part of this policy.",
            )
            rows.append(row)
            continue
        if category == "wire_integrity":
            try:
                api.read_json(case["wire_text"])
            except ValueError:
                matched = True
            else:
                matched = False
            row["matched"] = matched
            rows.append(row)
            continue
        policy = {}
        if "minimum_effective_feedback_signals" in case.get("policy", {}):
            policy["min_feedback_signals"] = case["policy"][
                "minimum_effective_feedback_signals"
            ]
        if "require_convex_loss_in_logits" in case.get("policy", {}):
            policy["require_convex_objective"] = case["policy"][
                "require_convex_loss_in_logits"
            ]
        runtime = copy.deepcopy(case.get("runtime"))
        if category == "runtime_integrity":
            if case["id"] == "string_lr":
                row.update(
                    matched=None,
                    note="Out of contract: the public rate API requires exact strings; corpus expected string rejection.",
                )
                rows.append(row)
                continue
            if case["id"] == "boundary_max_steps":
                row.update(
                    matched=None,
                    note="Out of contract: the proposed seed exceeds the public uint32 bound.",
                )
                rows.append(row)
                continue
            if type(runtime["learning_rate"]) is float and math.isfinite(
                runtime["learning_rate"]
            ):
                runtime["learning_rate"] = str(runtime["learning_rate"])
        out = engine.assess(case["record"], runtime=runtime, policy=policy)
        if category == "record_integrity":
            matched = (
                out["integrity"]["status"] == "REJECTED"
                and out["math_checks"] == []
                and out["status"] == "REJECTED"
            )
        elif category == "runtime_integrity":
            matched = (
                (out["mathematical_status"] == "PASS" and out["status"] == "PENDING")
                if case["expected"]["runtime_valid"]
                else (
                    out["status"] == "REJECTED"
                    and out["mathematical_status"] == "NOT_RUN"
                )
            )
        else:
            eligible = out["mathematical_status"] == "PASS"
            matched = (
                out["integrity"]["status"] == "PASS"
                and eligible == case["expected"]["handoff"]
            )
            if category == "theory_policy":
                matched = matched and (
                    (out["objective"]["status"] == "CONVEX")
                    == case["expected"]["convex_loss_in_logits"]
                )
        row.update(
            matched=matched,
            status=out["status"],
            mathematical_status=out["mathematical_status"],
            reasons=out["reasons"],
        )
        rows.append(row)
    batch = engine.run({"kind": "regression", "features": 4}, count=12, seed=19)
    item = next(
        x for x in batch["results"] if x["status"] == "READY_FOR_SCIENTIFIC_VALIDATION"
    )
    real = engine.checker.check
    for case in corpus["cases"]:
        if case["category"] != "verifier_fault":
            continue
        fault = case["id"]

        def injected(claim):
            verdict = real(claim)
            if not (
                isinstance(claim, dict)
                and claim.get("schema_version") == "generator-3.0"
            ):
                return verdict
            verdict = copy.deepcopy(verdict)
            if fault == "missing_required_subclaim":
                verdict.subclaims.pop(0)
            elif fault == "duplicated_required_subclaim":
                verdict.subclaims = [
                    copy.deepcopy(verdict.subclaims[0]) for _ in verdict.subclaims
                ]
            elif fault == "verified_without_certificate":
                verdict.subclaims[0]["verdict"]["certificate"] = None
            elif fault == "verified_bad_certificate":
                verdict.subclaims[0]["verdict"]["certificate"]["claim_hash"] = "0" * 64
            elif fault in (
                "false_tautology_substitution",
                "finite_proof_substitution",
                "vacuous_proof_substitution",
            ):
                if fault == "false_tautology_substitution":
                    replacement = {
                        "kind": "scalar",
                        "variables": {},
                        "lhs": "1",
                        "op": "==",
                        "rhs": "1",
                    }
                elif fault == "finite_proof_substitution":
                    replacement = {
                        "kind": "finite",
                        "variables": {
                            "x": {"domain": "integer", "lower": "0", "upper": "1"}
                        },
                        "lhs": "x",
                        "op": "==",
                        "rhs": "x",
                    }
                else:
                    replacement = {
                        "kind": "scalar",
                        "variables": {"x": {"lower": "1", "upper": "0"}},
                        "lhs": "x",
                        "op": "==",
                        "rhs": "x",
                    }
                verdict.subclaims[0]["claim"] = replacement
                verdict.subclaims[0]["verdict"] = real(replacement).to_dict()
            elif fault == "resource_limit":
                verdict.subclaims[0]["verdict"]["status"] = "RESOURCE_LIMIT"
                verdict.subclaims[0]["verdict"]["certificate"] = None
            return verdict

        context = (
            patch.object(engine.verifier, "verify_certificate", return_value=False)
            if fault == "replay_failure"
            else patch.object(engine.checker, "check", side_effect=injected)
        )
        with context:
            out = engine.assess(item["candidate"])
            try:
                engine.handoff(item["candidate"], out)
            except ValueError:
                handed = False
            else:
                handed = True
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "matched": out["status"] != "READY_FOR_SCIENTIFIC_VALIDATION"
                and not handed,
                "status": out["status"],
                "reasons": out["reasons"],
                "handoff_allowed": handed,
            }
        )
    return {
        "core_sha256": api.file_hash(ROOT / "math_filters.py"),
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "total": len(rows),
        "matched": sum(r["matched"] is True for r in rows),
        "mismatched": sum(r["matched"] is False for r in rows),
        "out_of_contract": sum(r["matched"] is None for r in rows),
        "categories": dict(Counter(r["category"] for r in rows)),
        "adapter_notes": [
            "Finite floating rates converted by decimal spelling to exact-string API, except conflicting string-rejection case.",
            "Unissued external record eligibility compared with mathematical_status; actual origin stays UNATTESTED.",
            "Fault injection uses genuine issued record and attempts actual handoff; corpus labels are unchanged.",
            "Corrected implementation is evaluated as regression after disclosed initial failures.",
        ],
        "rows": rows,
    }


if __name__ == "__main__":
    if "--demonstrate" in sys.argv:
        result = demonstrate()
        if "--output" in sys.argv:
            pathlib.Path(sys.argv[sys.argv.index("--output") + 1]).write_text(
                json.dumps(result, indent=2)
            )
        print(json.dumps(result, indent=2))
    elif "--external" in sys.argv:
        print(
            json.dumps(external_evaluation(ROOT / "independent_cases.json"), indent=2)
        )
    else:
        unittest.main()
