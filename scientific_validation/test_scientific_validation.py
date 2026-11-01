"""Scientific protocol, leakage, statistics and evidence tests; no discovery claims."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import contextlib
import io

ROOT = Path(__file__).resolve().parent
GENERATOR = Path(
    os.environ.get(
        "SCIENCE_GENERATOR", ROOT.parent / "generator_only" / "hypothesis_generator.py"
    )
)


def load_api():
    path = ROOT / "scientific_validation.py"
    spec = importlib.util.spec_from_file_location("science_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ProtocolContracts(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            (ROOT / "scientific_validation.py").exists(), "scientific engine missing"
        )
        self.api = load_api()

    def test_four_datasets_do_not_become_twenty_independent_seeds(self):
        result = self.api.paired_statistics(
            {"a": [1] * 5, "b": [1] * 5, "c": [1] * 5, "d": [1] * 5}
        )
        self.assertEqual(result["independent_units"], 4)
        self.assertEqual(result["paired_runs"], 20)
        self.assertEqual(result["p_value"], 0.125)

    def test_exact_null_and_holm_are_calibrated(self):
        self.assertEqual(self.api.paired_statistics({"a": [0], "b": [0]})["p_value"], 1)
        corrected = self.api.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(corrected["a"], 0.03)
        self.assertAlmostEqual(corrected["b"], 0.06)
        self.assertAlmostEqual(corrected["c"], 0.06)

    def test_numeric_bool_and_nonfinite_rejected(self):
        for x in (True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.api.paired_statistics({"a": [x], "b": [1]})

    def test_signed_permutation_p_values_are_not_monte_carlo_guesses(self):
        a = self.api.paired_statistics(
            {"a": [0.1, 0.2], "b": [0.3, 0.4], "c": [0.5, 0.6]}
        )
        self.assertEqual(a["p_value"], 0.25)
        self.assertIn("dataset", a["unit"])

    def test_valid_class_prediction_is_allowed_when_test_has_one_class(self):
        self.assertEqual(
            self.api.score_predictions("classification", [0, 0], [0, 1]), 0.5
        )

    def test_one_shot_registry_commits_before_evaluation(self):
        with tempfile.TemporaryDirectory() as scratch:
            journal = self.api.Journal(Path(scratch) / "study.sqlite")
            journal.freeze({"a": 1}, {"trials": []})
            self.assertEqual(journal.phase(), "FROZEN")
            journal.consume()
            self.assertEqual(journal.phase(), "CONSUMED")
            with self.assertRaises(ValueError):
                journal.consume()
            reopened = self.api.Journal(Path(scratch) / "study.sqlite", create=False)
            with self.assertRaises(ValueError):
                reopened.consume()

    def test_cannot_consume_before_freeze_or_refreeze(self):
        with tempfile.TemporaryDirectory() as scratch:
            journal = self.api.Journal(Path(scratch) / "study.sqlite")
            with self.assertRaises(ValueError):
                journal.consume()
            journal.freeze({"a": 1}, {"trials": []})
            with self.assertRaises(ValueError):
                journal.freeze({"a": 2}, {"trials": []})

    def test_simultaneous_consumers_have_exactly_one_winner(self):
        from concurrent.futures import ThreadPoolExecutor

        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "race.sqlite"
            journal = self.api.Journal(path)
            journal.freeze({"a": 1}, {"trials": []})

            def consume(_):
                try:
                    self.api.Journal(path, create=False).consume()
                    return True
                except ValueError:
                    return False

            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(consume, range(2)))
            self.assertEqual(sum(outcomes), 1)
            self.assertEqual(
                [e["kind"] for e in journal.events()], ["FREEZE", "CONSUME"]
            )

    def test_strict_protocol_types(self):
        for change in (
            {"seeds": [True]},
            {"budget": 3},
            {"alpha": True},
            {"steps": 0},
            {"primary_minimum": float("nan")},
        ):
            with self.assertRaises(ValueError):
                self.api.Protocol.parse(change)

    def test_baseline_grids_are_declared_and_distinct(self):
        for kind in ("classification", "regression"):
            grids = self.api.baseline_grids(kind)
            self.assertEqual(len(grids), 5)
            for family, grid in grids.items():
                self.assertEqual(len(grid), 24, (kind, family))
                self.assertEqual(len({self.api.digest(x) for x in grid}), 24)

    def test_label_storage_is_portable_across_numeric_dtypes(self):
        import numpy as np

        for kind, labels, classes in (
            ("classification", np.arange(40, dtype=np.int32) % 2, 2),
            ("regression", np.arange(40, dtype=np.float32), 1),
        ):
            data = self.api.Dataset(
                "portable", np.arange(120).reshape(40, 3), labels, kind, classes
            )
            parts = self.api.prepare_dataset(
                data, self.api.Protocol.parse({}), self.api.load_generator(GENERATOR)
            )
            for role in ("train", "development", "test"):
                self.assertEqual(
                    self.api.array_hash(np.asarray(parts[role]["y"])),
                    parts["meta"]["partitions"][role]["y_hash"],
                )

    def test_aliased_datasets_cannot_inflate_independent_units(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as scratch:
            x = np.arange(240).reshape(80, 3)
            y = np.arange(80) % 2
            data = [self.api.Dataset(name, x, y) for name in ("a", "b")]
            engine = self.api.ScientificEngine(
                Path(scratch) / "aliases.sqlite",
                GENERATOR,
                {"budget": 4, "steps": 3, "seeds": [0], "isolated": False},
            )
            with self.assertRaisesRegex(ValueError, "cohort|dataset|overlap"):
                engine.develop(data)

    def test_fresh_worker_executes_without_import_path_assumptions(self):
        import numpy as np

        gen = self.api.load_generator(GENERATOR)
        record = (
            gen.HypothesisGenerator(5)
            .generate(gen.TaskSpec("classification", 3, 2), 1)[0]
            .to_dict()
        )
        spec = {
            "family": "generated",
            "record": record,
            "learning_rate": "1/50",
            "steps": 3,
            "mode": "adaptive",
        }
        x = np.arange(90).reshape(30, 3).tolist()
        request = {
            "generator_path": str(GENERATOR),
            "train": {
                "role": "train",
                "x": x,
                "y": (np.arange(30) % 2).tolist(),
                "ids": [str(i) for i in range(30)],
            },
            "conditions": [{"name": "clean", "x": x[:2]}],
            "kind": "classification",
            "classes": 2,
            "model": spec,
            "seed": 0,
        }
        result = self.api.execute_fit(request, True, 30)
        self.assertTrue(result["success"], result)
        self.assertEqual(len(result["predictions"]["clean"]), 2)

    def test_duplicate_rows_and_groups_do_not_cross_splits(self):
        import numpy as np

        x = np.arange(120, dtype=float).reshape(40, 3)
        x[1] = x[0]
        y = np.arange(40) % 2
        groups = [str(i // 2) for i in range(40)]
        data = self.api.Dataset(
            "toy", x, y, "classification", 2, groups=groups, historical_exposure=False
        )
        parts = self.api.prepare_dataset(
            data, self.api.Protocol.parse({}), self.api.load_generator(GENERATOR)
        )
        sets = [
            set(parts["meta"]["partitions"][role]["groups"])
            for role in ("train", "development", "test")
        ]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_invalid_evidence_cannot_claim_any_scientific_approval(self):
        result = self.api.validate_evidence(
            {"accepted": True, "gates": {"novelty": "PASS"}}
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["status"], "INVALID_EVIDENCE")
        self.assertEqual(set(result["gates"]), set(self.api.GATES))

    def test_test_partition_has_no_access_before_consume(self):
        with tempfile.TemporaryDirectory() as scratch:
            engine = self.api.ScientificEngine(
                Path(scratch) / "study.sqlite", GENERATOR
            )
            with self.assertRaises(ValueError):
                engine.read_heldout("anything")

    def test_complete_pipeline_freezes_before_test_and_cannot_repeat(self):
        import numpy as np

        self.assertTrue(
            hasattr(self.api.ScientificEngine, "develop"),
            "development/frozen experiment pipeline missing",
        )
        with tempfile.TemporaryDirectory() as scratch:
            rng = np.random.default_rng(14)
            data = [
                self.api.Dataset(
                    "toy" + str(i),
                    rng.normal(size=(60, 3)),
                    np.arange(60) % 2,
                    "classification",
                    2,
                    historical_exposure=True,
                )
                for i in range(2)
            ]
            engine = self.api.ScientificEngine(
                Path(scratch) / "study.sqlite",
                GENERATOR,
                {"budget": 4, "steps": 3, "seeds": [0, 1], "isolated": False},
            )
            engine.develop(data)
            engine.freeze()
            with self.assertRaises(ValueError):
                engine.read_heldout("toy0")
            result = engine.evaluate()
            self.assertFalse(result["accepted"])
            self.assertEqual(engine.journal.phase(), "COMPLETE")
            with self.assertRaises(ValueError):
                engine.evaluate()
            evidence = engine.journal.evidence()
            self.assertEqual(len(evidence["runs"]), 2 * 2 * len(self.api.METHODS))
            self.assertNotEqual(result["status"], "INVALID_EVIDENCE", result)


class EvidenceReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np

        cls.api = load_api()
        cls.scratch = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(441)
        data = [
            cls.api.Dataset(
                "audit" + str(i), rng.normal(size=(80, 4)), np.arange(80) % 2
            )
            for i in range(2)
        ]
        engine = cls.api.ScientificEngine(
            Path(cls.scratch.name) / "audit.sqlite",
            GENERATOR,
            {"budget": 4, "steps": 3, "seeds": [0], "isolated": False},
        )
        with contextlib.redirect_stdout(io.StringIO()):
            engine.develop(data)
        reopened = cls.api.ScientificEngine(
            engine.journal.path, GENERATOR, create=False
        )
        assert reopened.protocol == engine.protocol
        reopened.freeze()
        with contextlib.redirect_stdout(io.StringIO()):
            result = reopened.evaluate()
        assert result["status"] != "INVALID_EVIDENCE", result
        cls.original = reopened.journal.evidence()

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def evidence(self):
        return copy.deepcopy(self.original)

    def reseal(self, body):
        body["manifest_hash"] = self.api.digest(body["manifest"])
        previous = ""
        for row in body["events"]:
            row["manifest_hash"] = body["manifest_hash"]
            row["previous_hash"] = previous
            row["event_hash"] = self.api.digest(
                {k: v for k, v in row.items() if k != "event_hash"}
            )
            previous = row["event_hash"]
        body["artifact_hash"] = self.api.digest(
            {k: v for k, v in body.items() if k != "artifact_hash"}
        )
        return body

    def invalid(self, body):
        result = self.api.validate_evidence(self.reseal(body), GENERATOR)
        self.assertEqual(result["status"], "INVALID_EVIDENCE", result)

    def test_modified_score_rejected_even_after_rehashing(self):
        body = self.evidence()
        body["runs"][0]["scores"]["clean"] += 0.1
        self.invalid(body)

    def test_missing_cell_rejected(self):
        body = self.evidence()
        body["runs"].pop()
        self.invalid(body)

    def test_declared_corruption_cannot_disagree_with_execution(self):
        body = self.evidence()
        body["manifest"]["corruption_recipe"]["noise_train_std"] = 0.9
        self.invalid(body)

    def test_pasted_scientific_approvals_rejected(self):
        body = self.evidence()
        body["scientific_gates_before"]["novelty"] = "PASS"
        self.invalid(body)

    def test_boolean_is_not_an_event_number_or_metric_normalizer(self):
        body = self.evidence()
        body["events"][0]["seq"] = True
        self.invalid(body)
        for field in ("normalizer", "development_normalizer"):
            body = self.evidence()
            name = next(iter(body["manifest"]["datasets"]))
            body["manifest"]["datasets"][name][field] = True
            body["data"][name]["meta"][field] = True
            self.invalid(body)
        body = self.evidence()
        body["manifest"]["environment"]["threads"] = True
        for row in body["runs"] + body["size_study"]:
            row["environment"]["threads"] = True
        for row in body["replications"]:
            row["fit"]["environment"]["threads"] = True
        self.invalid(body)
        body = self.evidence()
        fit = body["replications"][0]["fit"]
        fit["predictions"]["clean"][0] = bool(fit["predictions"]["clean"][0])
        self.invalid(body)

    def test_refit_flags_cannot_replace_replayed_predictions(self):
        body = self.evidence()
        p = body["replications"][0]["fit"]["predictions"]["clean"]
        p[0] = 1 - p[0]
        self.invalid(body)

    def test_resource_scope_cannot_be_relabelled(self):
        body = self.evidence()
        body["runs"][0]["resource_scope"] = "isolated per-model measurement"
        self.invalid(body)

    def test_feature_schema_cannot_be_forged(self):
        body = self.evidence()
        name = next(iter(body["manifest"]["datasets"]))
        body["manifest"]["datasets"][name]["features"] += 1
        body["data"][name]["meta"]["features"] += 1
        self.invalid(body)

    def test_ablation_cannot_silently_change_learning_rate(self):
        body = self.evidence()
        body["manifest"]["methods"]["fixed_matched"]["learning_rate"] = "1/1000"
        self.invalid(body)

    def test_nested_wrong_types_fail_closed(self):
        for replacement in (None, [], True, {"protocol": None}):
            body = self.evidence()
            body["manifest"] = replacement
            self.invalid(body)

    def test_missing_size_study_cannot_pass_efficiency(self):
        body = self.evidence()
        body["size_study"].pop()
        self.invalid(body)

    def test_failed_development_budget_is_not_fair_success(self):
        body = self.evidence()
        dev = body["development"]
        row = next(
            t
            for t in dev["trials"]
            if t["family"] == "linear"
            and t["config_id"] != dev["choices"]["baselines"][t["dataset"]]["linear"]
        )
        row["success"] = False
        row["error"] = "failed development fit"
        row["score"] = None
        body["manifest"]["development_hash"] = self.api.digest(dev)
        result = self.api.validate_evidence(self.reseal(body), GENERATOR)
        self.assertNotEqual(result["status"], "INVALID_EVIDENCE", result)
        self.assertEqual(result["gates"]["fair_benchmarking"]["status"], "FAIL")

    def test_terminal_failure_preserves_consumption(self):
        with tempfile.TemporaryDirectory() as scratch:
            engine = self.api.ScientificEngine(
                Path(scratch) / "failed.sqlite", GENERATOR
            )
            engine.journal.freeze({"protocol": {}}, {})

            def crash():
                engine.journal.consume()
                raise RuntimeError("interruption simulation")

            engine._evaluate = crash
            with self.assertRaises(RuntimeError):
                engine.evaluate()
            self.assertEqual(engine.journal.phase(), "FAILED")
            self.assertEqual(engine.journal.events()[-1]["kind"], "FAILED")
            with self.assertRaises(ValueError):
                engine.journal.consume()

    def test_live_upstream_handoff_is_bound_and_a_forgery_is_rejected(self):
        filter_path = GENERATOR.parent.parent / "math_update_gate" / "math_filters.py"
        verifier_path = GENERATOR.parent.parent / "math_verifier" / "math_checker.py"
        if not filter_path.exists() or not verifier_path.exists():
            self.skipTest("optional existing mathematical pipeline sources unavailable")
        spec = importlib.util.spec_from_file_location(
            "upstream_math_for_science_test", filter_path
        )
        upstream = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = upstream
        spec.loader.exec_module(upstream)
        math_engine = upstream.MathFilterEngine(GENERATOR, verifier_path)
        batch = math_engine.run(
            {"kind": "classification", "features": 4, "classes": 2},
            count=12,
            seed=42,
            runtime={"learning_rate": "1/1000"},
        )
        admitted = next(
            row
            for row in batch["results"]
            if row["status"] == "READY_FOR_SCIENTIFIC_VALIDATION"
        )
        packet = math_engine.handoff(admitted["candidate"], admitted["audit"])
        with tempfile.TemporaryDirectory() as scratch:
            engine = self.api.ScientificEngine(
                Path(scratch) / "handoff.sqlite",
                GENERATOR,
                {"budget": 4, "steps": 3, "seeds": [0], "isolated": False},
            )
            record = engine.accept_handoff(packet, math_engine)
            self.assertEqual(
                self.api.digest(record), self.api.digest(packet["candidate"])
            )
            self.assertEqual(len(engine.intake_receipts), 1)
            forged = copy.deepcopy(packet)
            forged["handoff_hash"] = "0" * 64
            with self.assertRaises(ValueError):
                engine.accept_handoff(forged, math_engine)
            datasets = []
            for name, parts in self.original["data"].items():
                x = parts["train"]["x"] + parts["development"]["x"] + parts["test"]["x"]
                y = parts["train"]["y"] + parts["development"]["y"] + parts["test"]["y"]
                datasets.append(self.api.Dataset(name, x, y))
            with contextlib.redirect_stdout(io.StringIO()):
                engine.develop(datasets, candidates=[record])
                engine.freeze()
                result = engine.evaluate()
            self.assertNotEqual(result["status"], "INVALID_EVIDENCE", result)
            self.assertEqual(
                len(engine.journal.evidence()["manifest"]["intake_receipts"]), 1
            )
            self.assertEqual(
                result["gates"]["mathematical_foundation"]["status"], "UNVERIFIED"
            )

    def test_regression_pipeline_has_train_only_metric_normalization(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as scratch:
            rng = np.random.default_rng(77)
            data = []
            for i in range(2):
                x = rng.normal(size=(80, 3))
                y = x[:, 0] + 0.1 * rng.normal(size=80)
                data.append(self.api.Dataset("reg" + str(i), x, y, "regression", 1))
            engine = self.api.ScientificEngine(
                Path(scratch) / "regression.sqlite",
                GENERATOR,
                {"budget": 4, "steps": 3, "seeds": [0], "isolated": False},
            )
            with contextlib.redirect_stdout(io.StringIO()):
                engine.develop(data)
                engine.freeze()
                result = engine.evaluate()
            self.assertNotEqual(result["status"], "INVALID_EVIDENCE", result)
            body = engine.journal.evidence()
            name = next(iter(body["manifest"]["datasets"]))
            body["manifest"]["datasets"][name]["normalizer"] = 2.0
            body["data"][name]["meta"]["normalizer"] = 2.0
            self.invalid(body)


if __name__ == "__main__":
    unittest.main()
