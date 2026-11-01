"""Demonstration runner for the data intelligence engine.

1. Planted-defect benchmark: known defects injected into clean synthetic tasks,
   many seeds; measures detection rate per defect and false-alarm rate per flag.
2. Real datasets with documented issues. Development datasets were inspected
   while the rules were written; held-out datasets were fixed in advance, with
   the expectations below written before the engine was first run on them.
3. Leakage-boundary check on real data: test-partition (and development)
   feature values are replaced; every fitted decision must stay identical.

Usage:
  python demo_data_engine.py --dev-dir DIR --heldout-dir DIR [--seeds 20] [--output out.json]
CSV files come from Kaggle (see REPORT.md for slugs and SHA-256 digests).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_engine as de  # noqa: E402

DEV_DATASETS = {
    "titanic": {
        "file": "Titanic-Dataset.csv",
        "target": "Survived",
        "documented": [("IDENTIFIER_COLUMN", "PassengerId"), ("FREE_TEXT_COLUMN", "Name"), ("MOSTLY_MISSING", "Cabin")],
    },
    "telco_churn": {
        "file": "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "target": "Churn",
        "documented": [("IDENTIFIER_COLUMN", "customerID"), ("MISSING_VALUES", "TotalCharges"), ("EXACT_LINEAR_DEPENDENCE", None)],
    },
    "adult_census": {
        "file": "adult.csv",
        "target": "income",
        "documented": [("REDUNDANT_COLUMNS", "education"), ("MISSING_VALUES", "workclass"), ("MISSING_VALUES", "occupation"),
                       ("MISSING_VALUES", "native.country")],
        "goals": ["class_imbalance"],
    },
    "heart_failure": {
        "file": "heart.csv",
        "target": "HeartDisease",
        "documented": [("DISGUISED_MISSING", "Cholesterol")],
    },
}

# Written before the engine was run on these files. "documented" = issues stated
# in the dataset's public description or long-standing community analyses.
# "predicted_miss" = documented issues the frozen rules were expected NOT to catch.
HELDOUT_DATASETS = {
    "pima_diabetes": {
        "file": "diabetes.csv",
        "target": "Outcome",
        "documented": [("DISGUISED_MISSING", "Glucose"), ("DISGUISED_MISSING", "BloodPressure"),
                       ("DISGUISED_MISSING", "SkinThickness"), ("DISGUISED_MISSING", "Insulin"),
                       ("DISGUISED_MISSING", "BMI")],
        "must_not": [("DISGUISED_MISSING", "Pregnancies")],
        "predicted_miss": [("DISGUISED_MISSING", "Glucose"), ("DISGUISED_MISSING", "Insulin")],
        "kind": "classification",
    },
    "breast_cancer_kaggle": {
        "file": "data.csv",
        "target": "diagnosis",
        "documented": [("IDENTIFIER_COLUMN", "id"), ("CONSTANT_COLUMN", "unnamed_32")],
        "goals": ["ill_conditioned"],
        "kind": "classification",
    },
    "bike_sharing_day": {
        "file": "day.csv",
        "target": "cnt",
        "time": "dteday",
        "documented": [("IDENTIFIER_COLUMN", "instant"), ("LEAKAGE_SUSPECT", ("casual", "registered"))],
        "verdict": "BLOCKED",
        "rerun_drop": ["casual", "registered"],
        "rerun_goals": ["distribution_shift", "ill_conditioned"],
        "kind": "regression",
    },
}


# --------------------------------------------------------------------------
# 1. Planted-defect benchmark


def base_task(kind, seed, n=600, d=8):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    signal = x @ w + 0.5 * np.sin(2 * x[:, 0])
    if kind == "classification":
        y = (signal + 0.15 * rng.normal(size=n) > 0).astype(int)
    else:
        y = signal + 0.3 * rng.normal(size=n)
    return rng, x.astype(object), y, [f"x{j}" for j in range(d)], signal


def add(x, cols, column, name):
    return np.column_stack([x, np.asarray(column, dtype=object)]), cols + [name]


def plan(card, name):
    return next((c for c in card.record.get("columns", []) if c["name"] == name), {})


def has(card, code, column=None):
    for w in card.warnings:
        if w["code"] != code:
            continue
        if column is None or w.get("column") == column:
            return True
        if isinstance(column, tuple) and set(column) <= set(w.get("evidence", {}).get("columns", [])):
            return True
    return False


def goal(card, name):
    return bool(card.record.get("goals", {}).get(name, {}).get("active"))


def make_defect(name, kind, seed):
    rng, x, y, cols, signal = base_task(kind, seed)
    n = len(y)
    kw = {}
    if name == "label_noise":
        if kind == "classification":
            flip = rng.random(n) < 0.25
            y = np.where(flip, 1 - y, y)
        else:
            y = y + 1.5 * np.std(signal) * rng.normal(size=n)
        check = lambda c: goal(c, "target_noise")
    elif name == "outliers":
        rows = rng.choice(n, int(0.04 * n), replace=False)
        for r in rows:
            x[r, rng.integers(x.shape[1])] = float(rng.choice([-1, 1]) * rng.uniform(8, 12))
        check = lambda c: goal(c, "outliers")
    elif name == "collinear":
        x, cols = add(x, cols, x[:, 0].astype(float) + 0.01 * rng.normal(size=n), "x0_copy")
        check = lambda c: goal(c, "ill_conditioned")
    elif name == "class_imbalance":
        keep = (y == 0) | (rng.random(n) < 0.2)
        x, y = x[keep], y[keep]
        check = lambda c: goal(c, "class_imbalance")
    elif name == "high_dimension":
        x = np.column_stack([x, rng.normal(size=(n, 80)).astype(object)])
        cols = cols + [f"noise{j}" for j in range(80)]
        check = lambda c: goal(c, "high_dimension")
    elif name == "leakage":
        scale = 0.05 if kind == "classification" else 0.01 * np.std(y)
        x, cols = add(x, cols, y + scale * rng.normal(size=n), "leak")
        check = lambda c: c.verdict == "BLOCKED" and has(c, "LEAKAGE_SUSPECT", "leak")
    elif name == "sum_leak":
        a = np.exp(x[:, 0].astype(float))
        b = np.exp(x[:, 1].astype(float))
        x, cols = add(x, cols, a, "part_a")
        x, cols = add(x, cols, b, "part_b")
        y = a + b
        check = lambda c: c.verdict == "BLOCKED" and has(c, "LEAKAGE_SUSPECT", ("part_a", "part_b"))
    elif name == "identifier_numeric":
        x, cols = add(x, cols, rng.permutation(n) + 5000, "ref")
        check = lambda c: plan(c, "ref").get("reason") == "IDENTIFIER"
    elif name == "identifier_string":
        x, cols = add(x, cols, [f"AC{v:010d}" for v in rng.choice(10**9, n, replace=False)], "account")
        check = lambda c: plan(c, "account").get("reason") == "IDENTIFIER"
    elif name == "free_text":
        words = np.array(["late", "order", "broken", "refund", "great", "slow", "item", "box", "again", "never"])
        x, cols = add(x, cols, [" ".join(rng.choice(words, 6)) + f" #{i}" for i in range(n)], "comment")
        check = lambda c: plan(c, "comment").get("reason") == "FREE_TEXT"
    elif name == "constant":
        x, cols = add(x, cols, np.full(n, 3.0), "version")
        check = lambda c: plan(c, "version").get("reason") == "CONSTANT"
    elif name == "mostly_missing":
        x, cols = add(x, cols, np.where(rng.random(n) < 0.75, None, rng.normal(size=n)), "sparse")
        check = lambda c: plan(c, "sparse").get("role") == "indicator_only"
    elif name == "disguised_missing":
        lab = rng.normal(200, 30, n)
        lab[rng.random(n) < 0.15] = 0.0
        x, cols = add(x, cols, lab, "lab")
        check = lambda c: plan(c, "lab").get("sentinels") == [0.0]
    elif name == "missing_values":
        x[rng.random(n) < 0.2, 2] = None
        check = lambda c: plan(c, "x2").get("indicator") is True and "x2__missing" in c.record["features"]
    elif name == "redundant":
        level = rng.integers(0, 6, n)
        x, cols = add(x, cols, level.astype(float), "grade_num")
        x, cols = add(x, cols, np.array(["g0", "g1", "g2", "g3", "g4", "g5"], dtype=object)[level], "grade")
        check = lambda c: plan(c, "grade").get("reason") == "REDUNDANT"
    elif name == "duplicates":
        extra = rng.choice(n, n // 10, replace=False)
        x = np.vstack([x, x[extra]])
        y = np.concatenate([y, y[extra]])
        check = lambda c: has(c, "DUPLICATE_ROWS") and c.record["split"]["duplicate_rows"] >= n // 10
    elif name == "sorted_by_target":
        order = np.argsort(y, kind="stable")
        x, y = x[order], y[order]
        check = lambda c: has(c, "SORTED_BY_TARGET")
    elif name == "temporal_shift":
        t = np.arange(n)
        drift = (t / n) * 2.0
        for j in range(3):
            x[:, j] = x[:, j].astype(float) + drift
        x, cols = add(x, cols, t, "t")
        kw = {"time": "t"}
        check = lambda c: goal(c, "distribution_shift")
    else:
        raise ValueError(name)
    table = de.from_arrays(x, y, cols, name=f"{name}-{kind}-{seed}")
    return table, kw, check


DEFECTS = {
    "label_noise": ("classification", "regression"),
    "outliers": ("classification", "regression"),
    "collinear": ("classification", "regression"),
    "class_imbalance": ("classification",),
    "high_dimension": ("classification", "regression"),
    "leakage": ("classification", "regression"),
    "sum_leak": ("regression",),
    "identifier_numeric": ("classification", "regression"),
    "identifier_string": ("classification", "regression"),
    "free_text": ("classification", "regression"),
    "constant": ("classification", "regression"),
    "mostly_missing": ("classification", "regression"),
    "disguised_missing": ("classification", "regression"),
    "missing_values": ("classification", "regression"),
    "redundant": ("classification", "regression"),
    "duplicates": ("classification", "regression"),
    "sorted_by_target": ("classification", "regression"),
    "temporal_shift": ("classification", "regression"),
}


def flags(card):
    out = {f"goal:{g}" for g, v in card.record.get("goals", {}).items() if v["active"]}
    out |= {f"code:{w['code']}" for w in card.warnings if w["severity"] != "info"}
    if card.verdict == "BLOCKED":
        out.add("verdict:BLOCKED")
    return out


def planted_benchmark(seeds, offset=0):
    engine = de.DataEngine()
    detection = {}
    for name, kinds in DEFECTS.items():
        for kind in kinds:
            hits = 0
            for s in range(seeds):
                table, kw, check = make_defect(name, kind, 1000 + offset + s)
                card = engine.analyze(table, "target", **kw)
                hits += bool(check(card))
            detection[f"{name}/{kind}"] = {"detected": hits, "runs": seeds}
    false_alarms = {}
    for kind in ("classification", "regression"):
        counts = {}
        noise_metric = []
        bayes = []
        for s in range(seeds):
            _, x, y, cols, signal = base_task(kind, 5000 + offset + s)
            card = engine.analyze(de.from_arrays(x, y, cols), "target")
            for f in flags(card):
                counts[f] = counts.get(f, 0) + 1
            probes = card.record["diagnostics"]["probes"]
            noise_metric.append(probes["hard_rate"] if kind == "classification" else probes["unexplained"])
            if kind == "classification":
                bayes.append(float(np.mean((signal > 0) != (y == 1))))
            # Stationary temporal control: a time column without drift.
            xt, ct = add(x, cols, np.arange(len(y)), "t")
            card_t = engine.analyze(de.from_arrays(xt, y, ct), "target", time="t")
            if goal(card_t, "distribution_shift"):
                counts["goal:distribution_shift(temporal control)"] = counts.get("goal:distribution_shift(temporal control)", 0) + 1
        false_alarms[kind] = {
            "runs": seeds,
            "fired": dict(sorted(counts.items())),
            "noise_metric": {"min": min(noise_metric), "mean": float(np.mean(noise_metric)), "max": max(noise_metric)},
        }
        if bayes:
            false_alarms[kind]["label_disagreement_with_noise_free_rule"] = float(np.mean(bayes))
    noisy = []
    for s in range(seeds):
        table, kw, _ = make_defect("label_noise", "classification", 1000 + offset + s)
        noisy.append(engine.analyze(table, "target", **kw).record["diagnostics"]["probes"]["hard_rate"])
    false_alarms["classification"]["noise_metric_with_25pct_flips"] = {
        "min": min(noisy), "mean": float(np.mean(noisy)), "max": max(noisy)}
    return {"detection": detection, "clean_false_alarms": false_alarms, "seed_offset": offset}


# --------------------------------------------------------------------------
# 2. Real datasets


def found(card, item):
    code, column = item
    return has(card, code, column)


def run_real(spec, directory, name):
    path = os.path.join(directory, spec["file"])
    if not os.path.exists(path):
        return {"available": False, "path": path}
    start = time.perf_counter()
    try:
        table = de.read_csv(path, name)
        card = de.DataEngine().analyze(table, spec["target"], time=spec.get("time"))
    except Exception as exc:  # an engine crash is a result, not a skipped row
        return {"available": True, "error": f"{type(exc).__name__}: {exc}"}
    seconds = time.perf_counter() - start
    result = {
        "available": True,
        "raw_sha256": table.raw_sha256,
        "rows": len(table.rows),
        "columns": len(table.columns),
        "verdict": card.verdict,
        "kind": card.record.get("kind"),
        "seconds": round(seconds, 3),
        "card_sha256": card.card_hash(),
        "documented": [{"issue": list(i) if not isinstance(i[1], tuple) else [i[0], list(i[1])], "found": found(card, i)} for i in spec["documented"]],
        "goals_active": sorted(g for g, v in card.record.get("goals", {}).items() if v["active"]),
        "warnings": [
            {k: w[k] for k in ("code", "severity", "column") if k in w} for w in card.warnings
        ],
        "summary": card.summary(),
    }
    if "kind" in spec:
        result["kind_correct"] = card.record.get("kind") == spec["kind"]
    if "must_not" in spec:
        result["must_not"] = [{"issue": list(i), "found": found(card, i)} for i in spec["must_not"]]
    if "predicted_miss" in spec:
        result["predicted_miss"] = [list(i) for i in spec["predicted_miss"]]
    if "goals" in spec:
        result["expected_goals"] = {g: goal(card, g) for g in spec["goals"]}
    if "verdict" in spec:
        result["verdict_expected"] = spec["verdict"]
    if card.verdict != "BLOCKED":
        result["leakage_boundary"] = leakage_boundary(table, card, spec)
    if "rerun_drop" in spec:
        again = de.DataEngine().analyze(table, spec["target"], time=spec.get("time"), drop=spec["rerun_drop"])
        result["rerun"] = {
            "drop": spec["rerun_drop"],
            "verdict": again.verdict,
            "goals_active": sorted(g for g, v in again.record["goals"].items() if v["active"]),
            "expected_goals": {g: goal(again, g) for g in spec["rerun_goals"]},
            "shift": again.record["diagnostics"].get("shift"),
            "summary": again.summary(),
        }
        result["leakage_boundary"] = leakage_boundary(table, again, {**spec, "drop": spec["rerun_drop"]})
    return result


def decisions(card):
    d = card.to_dict()
    for key in ("timings", "split", "array_sha256", "raw_sha256", "content_sha256", "rows_input", "name", "source"):
        d.pop(key, None)
    d["warnings"] = [w for w in d["warnings"] if w["code"] != "UNSEEN_AT_TRANSFORM"]
    return de.canonical(d)


def leakage_boundary(table, card, spec):
    """Replace development or test feature cells; fitted decisions must not move.

    Only rows that are alone in their duplicate bucket are changed, and only
    columns the engine keeps, so the split itself is unaffected by design.
    """
    rng = np.random.default_rng(99)
    used = [i for i, r in enumerate(table.rows) if r[table.columns.index(spec["target"])] is not None]
    bucket = dict(zip(used, card._science["groups"]))
    sizes = {}
    for b in bucket.values():
        sizes[b] = sizes.get(b, 0) + 1
    kept = {c["name"]: c["role"] for c in card.record["columns"] if c["role"] != "dropped"}
    out = {}
    for part in ("test", "development"):
        rows = [list(r) for r in table.rows]
        changed = 0
        targets = [int(i) for i in card._row_ids[part] if sizes[bucket[int(i)]] == 1]
        for i in targets:
            for name, role in kept.items():
                j = table.columns.index(name)
                rows[i][j] = repr(1e6 + rng.random()) if role == "numeric" else f"__perturbed_{i}"
                changed += 1
        other_table = de.Table(table.name, table.columns, [tuple(r) for r in rows], table.source, table.raw_sha256)
        other = de.DataEngine().analyze(
            other_table, spec["target"], time=spec.get("time"), drop=spec.get("drop", ())
        )
        out[part] = {
            "rows_changed": len(targets),
            "cells_changed": changed,
            "split_identical": other._row_ids == card._row_ids,
            "decisions_identical": decisions(other) == decisions(card),
            "train_matrix_identical": bool(np.array_equal(other.train[0], card.train[0])),
        }
    return out


# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-dir", required=True)
    ap.add_argument("--heldout-dir", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--fresh-offset", type=int, default=7000, help="seed offset for the fresh-seed rerun")
    ap.add_argument("--output")
    args = ap.parse_args(argv)
    start = time.perf_counter()
    report = {
        "version": de.VERSION,
        "engine_sha256": de._sha_text(open(os.path.join(HERE, "data_engine.py"), encoding="utf-8").read()),
        "policy_sha256": de.Policy().digest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
    }
    t0 = time.perf_counter()
    report["planted"] = planted_benchmark(args.seeds)
    report["planted"]["seconds"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    report["planted_fresh_seeds"] = planted_benchmark(args.seeds, offset=args.fresh_offset)
    report["planted_fresh_seeds"]["seconds"] = round(time.perf_counter() - t0, 2)
    report["development_datasets"] = {k: run_real(v, args.dev_dir, k) for k, v in DEV_DATASETS.items()}
    report["builtin_datasets"] = {}
    for name in ("iris", "wine", "breast_cancer", "digits", "diabetes"):
        card = de.DataEngine().analyze(de.load_builtin(name), "target")
        report["builtin_datasets"][name] = {
            "verdict": card.verdict,
            "goals_active": sorted(g for g, v in card.record["goals"].items() if v["active"]),
            "warnings": [w["code"] for w in card.warnings if w["severity"] != "info"],
        }
    report["heldout_datasets"] = {k: run_real(v, args.heldout_dir, k) for k, v in HELDOUT_DATASETS.items()}
    report["seconds"] = round(time.perf_counter() - start, 2)
    text = json.dumps(report, indent=2, sort_keys=True, default=de._default)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
