"""Contract tests for the data intelligence engine."""

import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import data_engine as de  # noqa: E402


def load_module(relative, name):
    path = os.path.join(ROOT, relative)
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def synthetic(seed=0, n=400, d=6, kind="classification", noise=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    if kind == "classification":
        y = (x @ w + 0.1 * rng.normal(size=n) > 0).astype(int)
        flip = rng.random(n) < noise
        y[flip] = 1 - y[flip]
    else:
        y = x @ w + 0.5 * rng.normal(size=n) + noise * rng.normal(size=n) * np.std(x @ w)
    return x, y


def codes(card):
    return {w["code"] for w in card.warnings}


def decisions(card):
    """Everything the engine fitted or concluded (not test-partition descriptions)."""
    d = card.to_dict()
    for key in ("timings", "split", "array_sha256", "raw_sha256", "content_sha256", "rows_input", "name", "source"):
        d.pop(key, None)
    d["warnings"] = [w for w in d["warnings"] if w["code"] != "UNSEEN_AT_TRANSFORM"]
    return de.canonical(d)


class ParsingContract(unittest.TestCase):
    def test_csv_missing_tokens_bom_quotes_and_empty_header(self):
        text = '﻿a,b,,target\n1,"x, y",,0\nNA,?,,1\n" 3 ",z,,0\n'
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as fh:
            fh.write(text)
        try:
            t = de.read_csv(fh.name)
        finally:
            os.unlink(fh.name)
        self.assertEqual(t.columns, ("a", "b", "unnamed_2", "target"))
        self.assertEqual(t.rows[0], ("1", "x, y", None, "0"))
        self.assertEqual(t.rows[1], (None, None, None, "1"))
        self.assertEqual(t.rows[2][0], "3")
        self.assertEqual(len(t.raw_sha256), 64)

    def test_trailing_empty_header_with_short_rows(self):
        # Kaggle's breast-cancer CSV: the header ends with a comma, data rows do not.
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            fh.write('"id","a","target",\n1,2.5,M\n2,3.5,B,\n')
        try:
            t = de.read_csv(fh.name)
            self.assertEqual(t.columns, ("id", "a", "target", "unnamed_3"))
            self.assertEqual(t.rows[0], ("1", "2.5", "M", None))
            with open(fh.name, "w") as out:
                out.write("a,b,c\n1,2\n")
            with self.assertRaises(ValueError):
                de.read_csv(fh.name)
        finally:
            os.unlink(fh.name)

    def test_strict_tables(self):
        with self.assertRaises(ValueError):
            de.Table("t", ("a", "a"), [], "s", "")
        with self.assertRaises(ValueError):
            de.Table("t", ("a", "b"), [("1",)], "s", "")
        x, y = synthetic()
        with self.assertRaises(ValueError):
            de.DataEngine().analyze(de.from_arrays(x, y), "missing_column")
        with self.assertRaises(ValueError):
            de.DataEngine().analyze(de.from_arrays(x, y), "target", seed=1.5)


class SplitContract(unittest.TestCase):
    def test_disjoint_stratified_duplicates_and_groups(self):
        x, y = synthetic(1, n=500)
        x = np.vstack([x, x[:40]])  # 40 exact duplicate rows
        y = np.concatenate([y, y[:40]])
        groups = np.arange(len(y)) // 3
        cols = [f"f{j}" for j in range(x.shape[1])] + ["entity"]
        t = de.from_arrays(np.column_stack([x, groups]), y, cols)
        card = de.DataEngine().analyze(t, "target", groups="entity")
        self.assertNotEqual(card.verdict, "BLOCKED")
        ids = {k: set(map(int, v)) for k, v in card._row_ids.items()}
        self.assertFalse(ids["train"] & ids["development"] or ids["train"] & ids["test"] or ids["development"] & ids["test"])
        self.assertEqual(sum(map(len, ids.values())), len(y))
        rows = {}
        for part, members in ids.items():
            for i in members:
                rows.setdefault(t.rows[i][:-2], set()).add(part)  # features only
                rows.setdefault(("group", t.rows[i][-2]), set()).add(part)
        self.assertTrue(all(len(v) == 1 for v in rows.values()), "duplicate or group crossed partitions")
        sizes = card.record["split"]["sizes"]
        self.assertAlmostEqual(sizes["train"] / len(y), 0.6, delta=0.03)
        counts = card.record["split"]["class_counts"]
        for c in ("0", "1"):
            total = sum(counts[p][c] for p in counts)
            self.assertAlmostEqual(counts["train"][c] / total, 0.6, delta=0.05)

    def test_deterministic_and_seed_sensitive(self):
        x, y = synthetic(2)
        t = de.from_arrays(x, y)
        a = de.DataEngine().analyze(t, "target", seed=3)
        b = de.DataEngine().analyze(t, "target", seed=3)
        c = de.DataEngine().analyze(t, "target", seed=4)
        self.assertEqual(a.card_hash(), b.card_hash())
        self.assertNotEqual(a._row_ids["test"], c._row_ids["test"])

    def test_rare_class_moved_into_training(self):
        x, y = synthetic(3, n=300)
        y = y.astype(object)
        y[:6] = "rare"
        card = de.DataEngine().analyze(de.from_arrays(x, y), "target")
        self.assertGreater(card.record["split"]["class_counts"]["train"]["rare"], 0)

    def test_declared_split_is_respected_and_fits_on_declared_train(self):
        rng = np.random.default_rng(26)
        x, y = synthetic(26, n=300)
        cat = rng.choice(["a", "b"], size=300).astype(object)
        order = rng.permutation(300)
        declared = {"train": sorted(map(int, order[:180])), "development": sorted(map(int, order[180:240])),
                    "test": sorted(map(int, order[240:]))}
        for i in declared["test"][:10]:
            cat[i] = "only_in_test"
        t = de.from_arrays(np.column_stack([x, cat]), y, [f"f{j}" for j in range(6)] + ["cat"])
        card = de.DataEngine().analyze(t, "target", split=declared)
        self.assertNotEqual(card.verdict, "BLOCKED", card.summary())
        self.assertEqual(card.record["split"]["method"], "declared")
        self.assertEqual({k: sorted(map(int, v)) for k, v in card._row_ids.items()}, declared)
        self.assertNotIn("cat=only_in_test", card.record["features"])  # vocabulary from declared train only
        self.assertEqual(card.used_rows, tuple(range(300)))

    def test_invalid_declared_splits_block(self):
        x, y = synthetic(27, n=200)
        x = np.vstack([x, x[:1]])  # row 200 duplicates row 0
        y = np.concatenate([y, y[:1]])
        t = de.from_arrays(x, y)
        base = {"train": list(range(0, 120)), "development": list(range(120, 160)), "test": list(range(160, 201))}
        cases = {
            "overlap": {**base, "development": list(range(110, 160))},
            "missing": {**base, "test": list(range(160, 200))},
            "duplicate crossing": base,  # rows 0 (train) and 200 (test) are duplicates
        }
        for label, split in cases.items():
            card = de.DataEngine().analyze(t, "target", split=split)
            self.assertEqual(card.verdict, "BLOCKED", label)
            self.assertIn("SPLIT_DECLARED_INVALID", codes(card), label)
        only_zero = [i for i in range(201) if y[i] == 0]
        ones = [i for i in range(201) if y[i] == 1]
        card = de.DataEngine().analyze(t, "target", split={"train": only_zero[:-20], "development": only_zero[-20:],
                                                          "test": ones})
        self.assertIn("CLASS_ABSENT_FROM_TRAIN", codes(card))
        with self.assertRaises(ValueError):
            de.DataEngine().analyze(t, "target", split={"train": [1]})

    def test_temporal_split_moves_forward_in_time(self):
        x, y = synthetic(4, n=300)
        stamp = np.arange(300)[::-1]
        t = de.from_arrays(np.column_stack([x, stamp]), y, [f"f{j}" for j in range(6)] + ["t"])
        card = de.DataEngine().analyze(t, "target", time="t")
        times = {k: [stamp[int(i)] for i in v] for k, v in card._row_ids.items()}
        self.assertLess(max(times["train"]), min(times["development"]))
        self.assertLess(max(times["development"]), min(times["test"]))
        self.assertNotIn("t", card.record["features"])


class LeakageBoundary(unittest.TestCase):
    def test_test_and_development_features_cannot_change_decisions(self):
        rng = np.random.default_rng(5)
        x, y = synthetic(5, n=400)
        cat = rng.choice(["a", "b", "c"], size=400).astype(object)
        t = de.from_arrays(np.column_stack([x, cat]), y, [f"f{j}" for j in range(6)] + ["cat"])
        base = de.DataEngine().analyze(t, "target")
        for part in ("test", "development"):
            rows = [list(r) for r in t.rows]
            for i in map(int, base._row_ids[part]):
                for j in range(6):
                    rows[i][j] = repr(1e6 + rng.random())
                rows[i][6] = f"unseen_{i}"
            changed = de.Table("t", t.columns, [tuple(r) for r in rows], "perturbed", "")
            other = de.DataEngine().analyze(changed, "target")
            self.assertEqual(base._row_ids, other._row_ids)
            self.assertEqual(decisions(base), decisions(other), f"{part} rows influenced fitted decisions")
            self.assertTrue(np.array_equal(base.train[0], other.train[0]))

    def test_sealed_test_needs_purpose_and_ledger_refuses_reuse(self):
        x, y = synthetic(6)
        card = de.DataEngine().analyze(de.from_arrays(x, y), "target")
        self.assertIsNone(getattr(card, "test", None))
        with self.assertRaises(de.SealedTestError):
            card.unseal_test("")

        class Ledger:
            def __init__(self):
                self.seen = set()

            def holdout_access(self, fingerprint, purpose, study):
                if fingerprint in self.seen:
                    raise RuntimeError("holdout reuse refused")
                self.seen.add(fingerprint)

        ledger = Ledger()
        X, Y = card.unseal_test("final evaluation", ledger=ledger)
        self.assertEqual(len(X), card.record["split"]["sizes"]["test"])
        with self.assertRaises(RuntimeError):
            card.unseal_test("second look", ledger=ledger)
        self.assertEqual(card.unseal_count, 1)
        self.assertFalse(X.flags.writeable)


class ColumnIntelligence(unittest.TestCase):
    def analyze(self, extra, names, kind="classification", seed=7, **kw):
        x, y = synthetic(seed, n=400, kind=kind)
        cols = [f"f{j}" for j in range(x.shape[1])] + names
        data = np.column_stack([x] + extra) if extra else x
        return de.DataEngine().analyze(de.from_arrays(data, y, cols if extra else None), "target", **kw)

    def plan(self, card, name):
        return next(c for c in card.record["columns"] if c["name"] == name)

    def test_identifiers_text_constant_and_mostly_missing(self):
        rng = np.random.default_rng(8)
        dense = rng.permutation(400) + 1000
        uid = np.array([f"u{v:08x}" for v in rng.integers(0, 2**32, 400)], dtype=object)
        text = np.array([f"note number {i} by {rng.integers(99)}" for i in range(400)], dtype=object)
        const = np.full(400, 7.0)
        sparse = np.where(rng.random(400) < 0.8, np.nan, rng.normal(size=400))
        card = self.analyze([dense, uid, text, const, sparse], ["ref", "user", "note", "k", "sparse"])
        self.assertEqual(self.plan(card, "ref")["reason"], "IDENTIFIER")
        self.assertEqual(self.plan(card, "user")["reason"], "IDENTIFIER")
        self.assertEqual(self.plan(card, "note")["reason"], "FREE_TEXT")
        self.assertEqual(self.plan(card, "k")["reason"], "CONSTANT")
        self.assertEqual(self.plan(card, "sparse")["role"], "indicator_only")
        self.assertIn("sparse__missing", card.record["features"])

    def test_categorical_encoding_rare_missing_unseen_and_frequency(self):
        rng = np.random.default_rng(9)
        cat = rng.choice(["red", "green", "blue"], size=400).astype(object)
        cat[:3] = "violet"  # rare
        cat[3:20] = None
        wide = np.array([f"c{v}" for v in rng.integers(0, 60, 400)], dtype=object)
        card = self.analyze([cat, wide], ["colour", "wide"])
        colour = self.plan(card, "colour")
        self.assertEqual(colour["role"], "categorical")
        self.assertTrue(colour["other"] and colour["missing_level"])
        self.assertIn("colour=__missing__", card.record["features"])
        self.assertEqual(self.plan(card, "wide")["role"], "frequency")
        wide_missing = wide.copy()
        wide_missing[:30] = None
        card2 = self.analyze([cat, wide_missing], ["colour", "wide"])
        self.assertIn("wide__missing", card2.record["features"])
        self.assertTrue(any(w["code"] == "MISSING_VALUES" and w.get("column") == "wide" for w in card2.warnings))
        Xtr = card.train[0]
        block = [j for j, f in enumerate(card.record["features"]) if f.startswith("colour=")]
        self.assertTrue(np.allclose(Xtr[:, block].sum(1), 1.0))

    def test_disguised_missing_versus_genuine_zero_mass(self):
        rng = np.random.default_rng(10)
        chol = rng.normal(240, 40, 400)
        chol[rng.random(400) < 0.15] = 0.0  # impossible value used as "unknown"
        spend = np.where(rng.random(400) < 0.3, 0.0, rng.gamma(1.0, 30.0, 400))  # genuine zeros
        card = self.analyze([chol, spend], ["chol", "spend"])
        self.assertEqual(self.plan(card, "chol")["sentinels"], [0.0])
        self.assertEqual(self.plan(card, "spend")["sentinels"], [])
        self.assertIn("chol__missing", card.record["features"])

    def test_redundant_pair(self):
        rng = np.random.default_rng(11)
        level = rng.integers(0, 6, 400)
        names = np.array(["zero", "low", "mid", "high", "top", "max"], dtype=object)[level]
        card = self.analyze([level.astype(float), names], ["grade_num", "grade"])
        self.assertEqual(self.plan(card, "grade")["reason"], "REDUNDANT")
        self.assertEqual(self.plan(card, "grade_num")["role"], "numeric")

    def test_unparseable_numeric_and_unseen_counts(self):
        rng = np.random.default_rng(12)
        v = rng.normal(size=400).astype(object)
        v[5] = "n.a."
        card = self.analyze([v], ["v"])
        self.assertTrue(codes(card) & {"UNPARSEABLE_NUMERIC", "UNSEEN_AT_TRANSFORM"})
        self.assertEqual(self.plan(card, "v")["role"], "numeric")


class Diagnostics(unittest.TestCase):
    def goals(self, card):
        return {g for g, v in card.record["goals"].items() if v["active"]}

    def test_clean_data_raises_no_goals(self):
        for kind in ("classification", "regression"):
            x, y = synthetic(13, n=600, kind=kind)
            card = de.DataEngine().analyze(de.from_arrays(x, y), "target")
            self.assertEqual(card.verdict, "READY", card.summary())
            self.assertEqual(self.goals(card), set(), card.summary())

    def test_planted_defects_are_detected(self):
        rng = np.random.default_rng(14)
        x, y = synthetic(14, n=600, noise=0.25)
        self.assertIn("target_noise", self.goals(de.DataEngine().analyze(de.from_arrays(x, y), "target")))
        x, y = synthetic(15, n=600)
        x2 = np.column_stack([x, x[:, 0] + 0.01 * rng.normal(size=600)])
        self.assertIn("ill_conditioned", self.goals(de.DataEngine().analyze(de.from_arrays(x2, y), "target")))
        x3 = x.copy()
        rows = rng.choice(600, 30, replace=False)
        x3[rows, rng.integers(0, 6, 30)] = rng.choice([-1, 1], 30) * 12
        self.assertIn("outliers", self.goals(de.DataEngine().analyze(de.from_arrays(x3, y), "target")))
        keep = (y == 0) | (rng.random(600) < 0.2)
        self.assertIn("class_imbalance", self.goals(de.DataEngine().analyze(de.from_arrays(x[keep], y[keep]), "target")))
        x4 = np.column_stack([x, rng.normal(size=(600, 120))])
        self.assertIn("high_dimension", self.goals(de.DataEngine().analyze(de.from_arrays(x4, y), "target")))

    def test_leakage_blocks_and_allow_overrides(self):
        x, y = synthetic(16, n=500)
        leak = y + 0.05 * np.random.default_rng(0).normal(size=500)
        t = de.from_arrays(np.column_stack([x, leak]), y, [f"f{j}" for j in range(6)] + ["leak"])
        card = de.DataEngine().analyze(t, "target")
        self.assertEqual(card.verdict, "BLOCKED")
        self.assertEqual(card.blocking[0]["column"], "leak")
        with self.assertRaises(ValueError):
            card.train
        allowed = de.DataEngine().analyze(t, "target", allow=["leak"])
        self.assertNotEqual(allowed.verdict, "BLOCKED")
        dropped = de.DataEngine().analyze(t, "target", drop=["leak"])
        self.assertNotIn("LEAKAGE_SUSPECT", codes(dropped))

    def test_sum_leak_in_regression(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(500, 5))
        a, b = np.exp(x[:, 0]), np.exp(x[:, 1])
        t = de.from_arrays(np.column_stack([x[:, 2:], a, b]), a + b, ["u", "v", "w", "part_a", "part_b"])
        card = de.DataEngine().analyze(t, "target")
        self.assertEqual(card.verdict, "BLOCKED")
        leak = [w for w in card.blocking if w["code"] == "LEAKAGE_SUSPECT"]
        self.assertTrue(any(set(w.get("evidence", {}).get("columns", [w.get("column")])) >= {"part_a", "part_b"} for w in leak))

    def test_sorted_file_and_temporal_shift(self):
        x, y = synthetic(18, n=400)
        order = np.argsort(y, kind="stable")
        sorted_card = de.DataEngine().analyze(de.from_arrays(x[order], y[order]), "target")
        self.assertIn("SORTED_BY_TARGET", codes(sorted_card))
        rng = np.random.default_rng(19)
        t = np.arange(600)
        xs = rng.normal(size=(600, 4)) + (t[:, None] / 600.0) * 2.0
        ys = (xs[:, 0] - xs[:, 1] + 0.2 * rng.normal(size=600) > 0).astype(int)
        drift = de.DataEngine().analyze(de.from_arrays(np.column_stack([xs, t]), ys, ["a", "b", "c", "d", "t"]), "target", time="t")
        self.assertTrue(drift.record["goals"]["distribution_shift"]["active"])
        flat = rng.normal(size=(600, 4))
        yf = (flat[:, 0] > 0).astype(int)
        still = de.DataEngine().analyze(de.from_arrays(np.column_stack([flat, t]), yf, ["a", "b", "c", "d", "t"]), "target", time="t")
        self.assertFalse(still.record["goals"]["distribution_shift"]["active"])

    def test_blocking_conditions(self):
        x, y = synthetic(20, n=20)
        self.assertIn("TOO_FEW_ROWS", codes(de.DataEngine().analyze(de.from_arrays(x, y), "target")))
        x, y = synthetic(21, n=200)
        y = y.astype(object)
        y[:2] = "tiny"
        self.assertIn("CLASS_TOO_RARE", codes(de.DataEngine().analyze(de.from_arrays(x, y), "target")))
        self.assertIn("TARGET_CONSTANT", codes(de.DataEngine().analyze(de.from_arrays(x, np.zeros(200)), "target", kind="regression")))


class Exports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = load_module("generator_only/hypothesis_generator.py", "_de_test_generator")
        cls.sv = load_module("scientific_validation/scientific_validation.py", "_de_test_science")

    def test_generator_task_spec_and_partitions_fit(self):
        if self.gen is None:
            self.skipTest("generator not present")
        x, y = synthetic(22, n=400, noise=0.2)
        card = de.DataEngine().analyze(de.from_arrays(x, y), "target")
        spec = card.task_spec(self.gen)
        self.assertEqual(spec.features, len(card.record["features"]))
        self.assertIn("target_noise", spec.goals)
        train, dev = card.generator_partitions(self.gen)
        h = self.gen.HypothesisGenerator(seed=1).generate(spec, 1)[0]
        model = self.gen.GeneratedModel(h, steps=20).fit(train)
        self.assertEqual(len(model.predict(dev.x)), len(dev.y))

    def test_science_dataset(self):
        if self.sv is None:
            self.skipTest("scientific_validation not present")
        x, y = synthetic(23, n=300, kind="regression")
        card = de.DataEngine().analyze(de.from_arrays(x, y), "target")
        ds = card.science_dataset(self.sv, historical_exposure=False, source="synthetic test data")
        self.assertEqual(ds.x.shape, (300, len(card.record["features"])))
        self.assertEqual(ds.kind, "regression")

    def test_cohort_overlap(self):
        x, y = synthetic(24, n=200)
        a = de.from_arrays(x, y, name="a")
        b = de.from_arrays(x[50:150], y[50:150], name="b")
        c = de.from_arrays(x + 9, y, name="c")
        result = {(r["a"], r["b"]): r for r in de.cohort_overlap([a, b, c])}
        self.assertEqual(result[("a", "b")]["shared_rows"], 100)
        self.assertEqual(result[("a", "c")]["shared_rows"], 0)
        # Two unrelated tables sharing one column name are not comparable row by row.
        d = de.from_arrays(np.column_stack([x[:, :1], np.arange(200)]), y, ["x0", "other"], target="y_d", name="d")
        e = de.from_arrays(np.column_stack([x[:, :1], -np.arange(200), np.ones(200)]), y, ["x0", "z", "w"],
                           target="y_e", name="e")
        pair = de.cohort_overlap([d, e])[0]
        self.assertFalse(pair["comparable"])
        self.assertIsNone(pair["shared_rows"])

    def test_cli(self):
        x, y = synthetic(25, n=200)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "data.csv")
            with open(path, "w") as fh:
                fh.write(",".join([f"f{j}" for j in range(6)] + ["label"]) + "\n")
                for row, label in zip(x, y):
                    fh.write(",".join(repr(float(v)) for v in row) + f",{label}\n")
            out = os.path.join(d, "card.json")
            self.assertEqual(de.main(["analyze", path, "--target", "label", "--json", out]), 0)
            self.assertTrue(os.path.getsize(out) > 1000)


if __name__ == "__main__":
    unittest.main()
