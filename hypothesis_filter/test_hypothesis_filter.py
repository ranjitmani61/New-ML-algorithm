"""Five-layer research screen contracts. No performance data or learners used."""

import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
GENERATOR = Path(
    os.environ.get(
        "HYPOTHESIS_GENERATOR",
        ROOT.parent / "generator_only" / "hypothesis_generator.py",
    )
)


def api():
    path = ROOT / "hypothesis_filter.py"
    assert path.exists(), "five-layer filter implementation is missing"
    spec = importlib.util.spec_from_file_location("hypothesis_screen_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Contracts(unittest.TestCase):
    def setUp(self):
        self.s = api()
        self.g = self.s.load_generator(GENERATOR)
        self.a = self.g.Controller("signal", name="alignment")
        self.b = self.g.Controller("signal", name="disagreement")
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)

    def record(self, controller=None, **changes):
        controller = controller or self.g.Controller.make("mean", self.a, self.b)
        algorithm = self.g.AlgorithmSpec(
            "classification",
            "linear",
            "standard",
            "cross_entropy",
            "memory_step",
            "mean",
            "identity",
            controller,
        )
        algorithm = replace(algorithm, **changes)
        return self.g.Hypothesis(
            algorithm,
            self.g.TaskSpec("classification", 4, 2, goals=("outliers",)),
            "construct",
            (),
            42,
            1,
        ).to_dict()

    def engine(self, **kwargs):
        return self.s.FiveLayerFilter(
            Path(self.scratch.name) / "journal.sqlite", GENERATOR, **kwargs
        )

    def test_exact_arithmetic_recognizes_nontrivial_identity(self):
        mean = self.g.Controller.make("mean", self.a, self.b)
        minmax = self.g.Controller.make(
            "mean",
            self.g.Controller.make("minimum", self.a, self.b),
            self.g.Controller.make("maximum", self.a, self.b),
        )
        self.assertEqual(
            self.s.normalize_controller(mean.to_dict()),
            self.s.normalize_controller(minmax.to_dict()),
        )
        self.assertNotEqual(
            self.s.normalize_controller(mean.to_dict()),
            self.s.normalize_controller(
                self.g.Controller.make("product", self.a, self.b).to_dict()
            ),
        )

    def test_complement_pair_is_recognized_as_constant(self):
        c = self.g.Controller.make(
            "mean", self.a, self.g.Controller.make("complement", self.a)
        )
        self.assertEqual(
            self.s.normalize_controller(c.to_dict()),
            self.s.normalize_controller(self.g.Controller("half").to_dict()),
        )

    def test_signed_polynomials_do_not_merge_distinct_rules(self):
        p = self.g.Controller.make("product", self.a, self.b)
        q = self.g.Controller.make("mean", p, self.g.Controller("half"))
        self.assertNotEqual(
            self.s.normalize_controller(p.to_dict()),
            self.s.normalize_controller(q.to_dict()),
        )

    def test_canonical_record_with_evidence_injected_is_rejected(self):
        record = self.record()
        record["test_score"] = 0.99
        audit = self.engine().evaluate_batch([record])
        self.assertEqual(audit["decisions"][0]["layers"][0]["status"], "REJECT")
        self.assertEqual(audit["decisions"][0]["status"], "REJECTED")

    def test_forged_generator_gate_is_rejected(self):
        record = self.record()
        record["acceptance_gates"]["novelty"] = "PASS"
        self.assertEqual(
            self.engine().evaluate_batch([record])["decisions"][0]["status"], "REJECTED"
        )

    def test_unsupported_domain_is_held_not_invented(self):
        record = {
            "schema_version": "open-research-1",
            "domain": "quantum_graph_learning",
            "novelty": "verified",
        }
        decision = self.engine().evaluate_batch([record])["decisions"][0]
        self.assertEqual(decision["status"], "HELD")
        self.assertFalse(decision["eligible_for_math"])

    def test_constant_control_does_not_pass_as_new_feedback(self):
        decision = self.engine().evaluate_batch(
            [self.record(self.g.Controller("half"))]
        )["decisions"][0]
        self.assertFalse(decision["eligible_for_math"])
        self.assertEqual(decision["layers"][2]["status"], "REJECT")

    def test_exact_duplicates_and_semantic_aliases_are_removed(self):
        equivalent = self.g.Controller.make(
            "mean",
            self.g.Controller.make("minimum", self.a, self.b),
            self.g.Controller.make("maximum", self.a, self.b),
        )
        audit = self.engine().evaluate_batch(
            [self.record(), self.record(), self.record(equivalent)]
        )
        self.assertEqual(
            sum(x["layers"][1]["status"] == "PASS" for x in audit["decisions"]), 1
        )

    def test_parameter_only_variants_do_not_become_new_algorithms(self):
        weighted = self.g.Controller.make(
            "mean", self.a, self.g.Controller.make("mean", self.a, self.b)
        )
        audit = self.engine().evaluate_batch([self.record(), self.record(weighted)])
        self.assertEqual(
            sum(x["layers"][2]["status"] == "PASS" for x in audit["decisions"]), 1
        )

    def test_data_free_pass_has_five_layers_and_unverified_science(self):
        audit = self.engine().evaluate_batch([self.record()])
        d = audit["decisions"][0]
        self.assertTrue(d["eligible_for_math"], d)
        self.assertEqual([x["status"] for x in d["layers"]], ["PASS"] * 5)
        self.assertEqual(d["novelty_status"], "UNVERIFIED")
        self.assertEqual(set(d["scientific_gates"].values()), {"UNVERIFIED"})
        self.assertGreaterEqual(len(d["dossier"]["domains"]), 2)
        self.assertTrue(d["dossier"]["prior_art"])
        self.assertGreaterEqual(len(d["potential"]["feedback_witnesses"]), 2)

    def test_dominated_research_cost_is_a_policy_rejection(self):
        complicated = self.g.Controller.make("product", self.a, self.b)
        audit = self.engine().evaluate_batch(
            [self.record(), self.record(complicated, feature_map="linear_fourier")]
        )
        self.assertTrue(any(d["eligible_for_math"] for d in audit["decisions"]))
        self.assertEqual(len(audit["layer_counts"]), 5)
        self.assertTrue(
            all(
                a >= b for a, b in zip(audit["layer_counts"], audit["layer_counts"][1:])
            )
        )

    def test_persistent_deduplication_survives_reopening(self):
        engine = self.engine()
        first = engine.evaluate_batch([self.record()])
        reopened = self.s.FiveLayerFilter(engine.path, GENERATOR, create=False)
        second = reopened.evaluate_batch([self.record()])
        self.assertTrue(first["decisions"][0]["eligible_for_math"])
        self.assertEqual(second["decisions"][0]["layers"][1]["status"], "REJECT")

    def test_audit_is_replayed_not_just_rehashed(self):
        engine = self.engine()
        audit = engine.evaluate_batch([self.record()])
        self.assertTrue(engine.replay_audit(audit))
        forged = copy.deepcopy(audit)
        forged["decisions"][0]["novelty_status"] = "VERIFIED_NOVEL"
        forged["audit_hash"] = self.s.digest(
            {k: v for k, v in forged.items() if k != "audit_hash"}
        )
        self.assertFalse(engine.replay_audit(forged))

    def test_handoff_cannot_be_forged_or_rebind_a_candidate(self):
        engine = self.engine()
        audit = engine.evaluate_batch([self.record()])
        d = audit["decisions"][0]
        packet = engine.handoff(d["candidate_id"])
        self.assertTrue(engine.verify_handoff(packet))
        forged = copy.deepcopy(packet)
        forged["candidate"]["task"]["features"] = 5
        self.assertFalse(engine.verify_handoff(forged))
        self.assertEqual(
            packet["candidate"]["acceptance_gates"]["novelty"], "UNVERIFIED"
        )

    def test_concurrent_rounds_cannot_each_pass_the_same_algorithm(self):
        engine = self.engine()
        path = engine.path
        record = self.record()

        def run(_):
            return self.s.FiveLayerFilter(path, GENERATOR, create=False).evaluate_batch(
                [record]
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            audits = list(pool.map(run, range(2)))
        self.assertEqual(
            sum(d["eligible_for_math"] for a in audits for d in a["decisions"]), 1
        )

    def test_unknown_policy_and_boolean_threshold_fail_closed(self):
        for policy in (
            {"test_accuracy": 0.9},
            {"min_feedback": True},
            {"min_feedback": 0},
        ):
            with self.assertRaises(ValueError):
                self.s.Policy.parse(policy)

    def test_external_artifacts_and_fit_are_not_accessed(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("learner fit must not run during filtering")

        with tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / "sealed_results.json"
            secret.write_text('{"test_accuracy":1}')
            engine = self.engine()
            engine.gen.GeneratedModel.fit = forbidden
            engine.gen.GeneratedModel.__init__ = forbidden
            audit = engine.evaluate_batch([self.record()])
            secret.write_text('{"test_accuracy":0}')
            other = self.s.FiveLayerFilter(Path(outside) / "separate.sqlite", GENERATOR)
            again = other.evaluate_batch([self.record()])
            self.assertEqual(audit["decision_digest"], again["decision_digest"])

    def test_missing_prior_round_cannot_replay_a_later_round(self):
        engine = self.engine()
        engine.evaluate_batch([self.record()])
        audit = engine.evaluate_batch([self.record(loss="brier")])
        with engine.connect() as db:
            db.execute("DELETE FROM rounds WHERE round_no=1")
        self.assertFalse(engine.replay_audit(audit))

    def test_partial_registry_corruption_cannot_bypass_persistence(self):
        engine = self.engine()
        audit = engine.evaluate_batch([self.record()])
        with engine.connect() as db:
            db.execute("DELETE FROM seen")
        self.assertFalse(engine.replay_audit(audit))
        with self.assertRaises(ValueError):
            engine.evaluate_batch([self.record(loss="brier")])

    def test_registered_known_algorithm_is_excluded(self):
        engine = self.engine()
        record = self.record()
        engine.register_prior_art(record, "yellowfin")
        d = engine.evaluate_batch([record])["decisions"][0]
        self.assertEqual(d["layers"][2]["status"], "REJECT")
        self.assertIn("registered", d["layers"][2]["reason"])
        self.assertNotIn("no known equivalent", d["dossier"]["screening_claim"])
        with self.assertRaises(ValueError):
            engine.register_prior_art(record, "adam")

    def test_associativity_absorption_and_distributivity_of_lattice(self):
        c = self.g.Controller("signal", name="dispersion")
        make = self.g.Controller.make
        pairs = [
            (
                make("minimum", self.a, make("minimum", self.b, c)),
                make("minimum", make("minimum", self.a, self.b), c),
            ),
            (make("minimum", self.a, make("maximum", self.a, self.b)), self.a),
            (
                make("minimum", self.a, make("maximum", self.b, c)),
                make(
                    "maximum",
                    make("minimum", self.a, self.b),
                    make("minimum", self.a, c),
                ),
            ),
        ]
        for left, right in pairs:
            self.assertEqual(
                self.s.normalize_controller(left.to_dict()),
                self.s.normalize_controller(right.to_dict()),
            )

    def test_complements_are_pushed_through_lattice_before_normalization(self):
        make = self.g.Controller.make
        left = make("complement", make("minimum", self.a, self.b))
        right = make("maximum", make("complement", self.a), make("complement", self.b))
        self.assertEqual(
            self.s.normalize_controller(left.to_dict()),
            self.s.normalize_controller(right.to_dict()),
        )
        c = self.g.Controller("signal", name="dispersion")
        self.assertEqual(
            self.s.normalize_controller(make("minimum", left, c).to_dict()),
            self.s.normalize_controller(make("minimum", right, c).to_dict()),
        )

    def test_identical_profiles_are_held_when_equivalence_is_unresolved(self):
        make = self.g.Controller.make
        xy = make("product", self.a, self.b)
        xxy = make("product", xy, self.a)
        restricted = make("minimum", xy, xxy)
        # x*x*y <= x*y on [0,1]. A profile cannot prove this identity.
        engine = self.engine()
        a = engine.evaluate_batch([self.record(xxy), self.record(restricted)])
        statuses = [d["layers"][1]["status"] for d in a["decisions"]]
        self.assertEqual(sorted(statuses), ["HOLD", "PASS"])
        self.assertEqual(sum(d["eligible_for_math"] for d in a["decisions"]), 1)

    def test_rejected_and_unsupported_payloads_are_not_journaled(self):
        secret = "FORBIDDEN_BENCHMARK_VALUE_012345"
        record = self.record()
        record["benchmark_results"] = {"heldout_metric": secret}
        unsupported = {"schema_version": "other", "benchmark_results": secret}
        engine = self.engine()
        audit = engine.evaluate_batch([record, unsupported])
        self.assertNotIn(secret, self.s.canonical(audit))
        self.assertNotIn("benchmark_results", self.s.canonical(audit))
        with engine.connect() as db:
            self.assertNotIn(
                secret, db.execute("SELECT body FROM rounds").fetchone()[0]
            )
        self.assertTrue(engine.replay_audit(audit))

    def test_nested_measurement_fields_are_rejected(self):
        for location in ("task", "algorithm"):
            record = self.record()
            record[location]["heldout_score"] = 0.99
            engine = self.s.FiveLayerFilter(
                Path(self.scratch.name) / (location + ".sqlite"), GENERATOR
            )
            d = engine.evaluate_batch([record])["decisions"][0]
            self.assertEqual(d["layers"][0]["status"], "REJECT")

    def test_exact_normalizer_agrees_with_independent_fraction_evaluation(self):
        from fractions import Fraction as F

        def direct(node, values):
            op = node["op"]
            v = [direct(x, values) for x in node["children"]]
            if op == "half":
                return F(1, 2)
            if op == "signal":
                return values[node["name"]]
            return {
                "complement": lambda: 1 - v[0],
                "mean": lambda: sum(v) / 2,
                "product": lambda: v[0] * v[1],
                "minimum": lambda: min(v),
                "maximum": lambda: max(v),
            }[op]()

        def normalized(normal, values):
            def poly(terms):
                total = F(0)
                for powers, coefficient in terms:
                    value = F(coefficient)
                    for token in powers:
                        value *= (
                            values[token[2:]]
                            if token.startswith("s:")
                            else min(poly(t) for t in normal["min_atoms"][token])
                        )
                    total += value
                return total

            return poly(normal["terms"])

        records = self.g.HypothesisGenerator(721).generate(
            self.g.TaskSpec("classification", 4, 2), 60
        )
        for h in records:
            node = h.algorithm.controller.to_dict()
            normal = self.s.normalize_controller(node)
            for k in range(7):
                point = {
                    name: F((i + k) % 5, 4) for i, name in enumerate(self.s.SIGNALS)
                }
                self.assertEqual(direct(node, point), normalized(normal, point))

    def test_real_origin_can_enter_existing_math_engine_without_forging_ledger(self):
        path = GENERATOR.parent.parent / "math_update_gate" / "math_filters.py"
        if not path.exists():
            self.skipTest("optional existing mathematical engine unavailable")
        spec = importlib.util.spec_from_file_location(
            "hyp_filter_math_integration", path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        math_engine = module.MathFilterEngine(
            GENERATOR, GENERATOR.parent.parent / "math_verifier" / "math_checker.py"
        )
        records = [
            h.to_dict()
            for h in self.g.HypothesisGenerator(42).generate(
                self.g.TaskSpec("classification", 4, 2), 12
            )
        ]
        engine = self.engine()
        audit = engine.evaluate_batch(records)
        ready = next(d for d in audit["decisions"] if d["eligible_for_math"])
        packet = engine.handoff(ready["candidate_id"])

        def forbidden_prefix(*args, **kwargs):
            raise AssertionError("unselected hypotheses must never enter mathematics")

        math_engine.run = forbidden_prefix
        result = engine.forward_to_math(
            packet, math_engine, runtime={"learning_rate": "1/1000"}
        )
        self.assertEqual(self.s.digest(result["candidate"]), ready["candidate_id"])
        self.assertTrue(
            math_engine.verify_receipt(result["candidate"], result["math_admission"])
        )
        self.assertEqual(result["novelty_status"], "UNVERIFIED")

    def test_round_order_does_not_change_batch_decisions(self):
        records = [
            self.record(),
            self.record(self.g.Controller.make("product", self.a, self.b)),
        ]
        a = self.engine().evaluate_batch(records)
        b = self.s.FiveLayerFilter(
            Path(self.scratch.name) / "other.sqlite", GENERATOR
        ).evaluate_batch(list(reversed(records)))
        self.assertEqual(a["decision_digest"], b["decision_digest"])


if __name__ == "__main__":
    unittest.main()
