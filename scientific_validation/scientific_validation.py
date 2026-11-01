"""Scientific validation for one frozen ML candidate, never automatic novelty proof.

One implementation file: development-only selection, atomic one-shot holdout,
tuned baselines/ablations, fixed corruptions, dataset-clustered inference and
ten separately scoped evidence gates. Four-file delivery includes the SQLite
experiment artifact. Existing generator source remains unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import importlib.util
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
import resource
import sqlite3
import statistics
import subprocess
import sys
import time
import warnings

import numpy as np
from scipy import stats
from threadpoolctl import threadpool_limits

VERSION = "scientific-validation-1.0"
GENERATOR_SHA256 = "8ce3010e741603c9627c85f37944586c4cde2155eeb24660adcfd58ca5931b3f"
GATES = (
    "novelty",
    "mathematical_foundation",
    "working_implementation",
    "accuracy_generalization",
    "robustness",
    "efficiency",
    "fair_benchmarking",
    "statistical_validation",
    "independent_reproducibility",
    "real_world_validation",
)
CONDITIONS = ("clean", "feature_noise", "feature_outliers", "covariate_corruption")
FAMILIES = ("linear", "kernel", "extra_trees", "boosting", "neighbors")
METHODS = ("candidate", "fixed_matched", "fixed_search", "first_only") + FAMILIES
CORRUPTION_RECIPE = {
    "noise_train_std": 0.15,
    "outlier_entry_fraction": 0.05,
    "outlier_train_std": 6.0,
    "shift_columns_fraction": 0.2,
    "shift_train_std": 0.5,
}
SIZE_FRACTIONS = (0.25, 0.5)
SIZE_RECIPE = {
    "fractions": [0.25, 0.5, 1.0],
    "seed": 314159,
    "sampling": "nested train-only class-covering subsets; original row order retained",
    "methods": ["candidate", "reference"],
    "full_size": "reuse planned full-data fits",
    "repeats": "all frozen fitting seeds",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SOURCE_SHA256 = file_hash(__file__)


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite strict numeric value required")
    return float(value)


def strict(value, allowed, required=()):
    if (
        not isinstance(value, dict)
        or set(value) - set(allowed)
        or set(required) - set(value)
    ):
        raise ValueError("unknown/missing fields")


def exact_rate(value):
    if not isinstance(value, str) or len(value) > 80:
        raise ValueError("exact rate string required")
    import re

    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?", value):
        raise ValueError("bounded decimal or rational rate required")
    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid rate") from exc
    if not 0 < rate <= 1:
        raise ValueError("learning rate outside (0,1]")
    return str(rate)


def load_generator(path=None):
    path = Path(
        path
        or os.environ.get("SCIENCE_GENERATOR")
        or Path(__file__).resolve().parent.parent
        / "generator_only"
        / "hypothesis_generator.py"
    ).resolve()
    if file_hash(path) != GENERATOR_SHA256:
        raise ValueError("unreviewed generator source")
    spec = importlib.util.spec_from_file_location("_scientific_generator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def array_hash(array):
    a = np.ascontiguousarray(array)
    if a.dtype.kind not in "iuf":
        raise ValueError("numeric arrays required")
    return digest(
        {
            "shape": list(a.shape),
            "dtype": a.dtype.str,
            "bytes": hashlib.sha256(a.tobytes()).hexdigest(),
        }
    )


@dataclass(frozen=True)
class Protocol:
    budget: int = 24
    seeds: tuple = (0, 1, 2, 3, 4)
    steps: int = 160
    generation_seed: int = 2026
    split_seed: int = 101
    development_seed: int = 17
    learning_rates: tuple = ("1/50", "2/25")
    alpha: float = 0.05
    primary_minimum: float = 0.005
    max_corruption_drop: float = 0.10
    max_corruption_disadvantage: float = 0.02
    max_fit_ratio: float = 5.0
    max_predict_ratio: float = 5.0
    max_rss_ratio: float = 2.0
    max_fit_growth: float = 12.0
    isolated: bool = True
    timeout_seconds: int = 120

    @classmethod
    def parse(cls, value=None):
        if isinstance(value, cls):
            value = asdict(value)
        value = {} if value is None else value
        strict(value, cls.__dataclass_fields__)
        body = {**asdict(cls()), **value}
        if (
            type(body["budget"]) is not int
            or not 4 <= body["budget"] <= 24
            or body["budget"] % 2
        ):
            raise ValueError("even development budget 4..24 required")
        for field in (
            "steps",
            "generation_seed",
            "split_seed",
            "development_seed",
            "timeout_seconds",
        ):
            if type(body[field]) is not int or not 0 <= body[field] < 2**32:
                raise ValueError("strict bounded integer config required")
        if not 1 <= body["steps"] <= 20000 or not 1 <= body["timeout_seconds"] <= 600:
            raise ValueError("step/time budget invalid")
        seeds = body["seeds"]
        if (
            not isinstance(seeds, (list, tuple))
            or not 1 <= len(seeds) <= 32
            or any(type(x) is not int or not 0 <= x < 2**32 for x in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("distinct uint32 seeds required")
        rates = body["learning_rates"]
        if not isinstance(rates, (list, tuple)) or len(rates) != 2:
            raise ValueError("two prespecified rates required")
        rates = tuple(exact_rate(x) for x in rates)
        if len(set(rates)) != 2:
            raise ValueError("distinct rate values required")
        for field in (
            "alpha",
            "primary_minimum",
            "max_corruption_drop",
            "max_corruption_disadvantage",
            "max_fit_ratio",
            "max_predict_ratio",
            "max_rss_ratio",
            "max_fit_growth",
        ):
            value = finite(body[field])
            if value <= 0:
                raise ValueError("positive numeric criteria required")
            body[field] = value
        if not 0 < body["alpha"] < 0.5:
            raise ValueError("alpha outside (0,.5)")
        if type(body["isolated"]) is not bool:
            raise ValueError("isolated must be Boolean")
        body["seeds"] = tuple(seeds)
        body["learning_rates"] = rates
        return cls(**body)


@dataclass
class Dataset:
    name: str
    x: object
    y: object
    kind: str = "classification"
    classes: int = 2
    groups: object = None
    historical_exposure: bool = True
    source: str = "caller-supplied; provenance not authenticated"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or len(self.name) > 80:
            raise ValueError("dataset name required")
        self.x = np.array(self.x, dtype=np.float64, copy=True)
        self.y = np.array(self.y, copy=True)
        if (
            self.x.ndim != 2
            or self.y.ndim != 1
            or len(self.x) != len(self.y)
            or len(self.x) < 20
            or not 1 <= self.x.shape[1] <= 4096
        ):
            raise ValueError("dataset dimensions invalid")
        if (
            not np.isfinite(self.x).all()
            or self.y.dtype.kind not in "iuf"
            or not np.isfinite(self.y).all()
        ):
            raise ValueError("finite numeric dataset required")
        if (
            self.kind not in ("classification", "regression")
            or type(self.classes) is not int
        ):
            raise ValueError("task type invalid")
        if self.kind == "classification":
            if (
                not 2 <= self.classes <= 256
                or self.y.dtype.kind not in "iu"
                or set(np.unique(self.y)) - set(range(self.classes))
            ):
                raise ValueError(
                    "integer-encoded labels and public class count required"
                )
            self.y = self.y.astype(np.int64)
        elif self.classes != 1:
            raise ValueError("regression has one output")
        else:
            self.y = self.y.astype(np.float64)
        if type(self.historical_exposure) is not bool:
            raise ValueError("historical exposure must be explicit Boolean")
        if (
            not isinstance(self.source, str)
            or not self.source
            or len(self.source) > 2000
        ):
            raise ValueError("source/provenance description required")
        if self.groups is not None and len(self.groups) != len(self.x):
            raise ValueError("group count mismatch")
        self.x.setflags(write=False)
        self.y.setflags(write=False)


def prepare_dataset(data, protocol, gen):
    parts = gen.split_data(data.x, data.y, seed=protocol.split_seed, groups=data.groups)
    result = {}
    metadata = {}
    for part in parts:
        indices = [int(x) for x in part.row_ids]
        ids = [data.name + ":" + str(i) for i in indices]
        group_ids = [
            (
                digest({"group": str(data.groups[i])})
                if data.groups is not None
                else gen._row_hash(data.x[i])
            )
            for i in indices
        ]
        value = {
            "role": part.role,
            "x": part.x.tolist(),
            "y": part.y.tolist(),
            "ids": ids,
        }
        result[part.role] = value
        metadata[part.role] = {
            "ids": ids,
            "row_hashes": [gen._row_hash(x) for x in part.x],
            "groups": group_ids,
            "size": len(part.y),
            "x_hash": array_hash(part.x),
            "y_hash": array_hash(part.y),
        }
    if data.kind == "classification" and set(parts[0].y) != set(range(data.classes)):
        raise ValueError("development training rows lack declared classes")
    final_y = np.r_[parts[0].y, parts[1].y]
    normalizer = (
        1.0 if data.kind == "classification" else max(float(np.var(final_y)), 1e-12)
    )
    result["meta"] = {
        "name": data.name,
        "kind": data.kind,
        "classes": data.classes,
        "features": data.x.shape[1],
        "rows": len(data.x),
        "source": data.source,
        "historical_exposure": data.historical_exposure,
        "content_hash": digest({"x": array_hash(data.x), "y": array_hash(data.y)}),
        "groups_supplied": data.groups is not None,
        "partitions": metadata,
        "normalizer": normalizer,
        "development_normalizer": (
            1.0
            if data.kind == "classification"
            else max(float(np.var(parts[0].y)), 1e-12)
        ),
    }
    return result


def check_cohort(parts):
    """Conservative observable overlap checks, not proof of independence."""
    content = set()
    rows = set()
    groups = set()
    for data in parts:
        meta = data["meta"]
        current = {
            row for part in meta["partitions"].values() for row in part["row_hashes"]
        }
        current_groups = (
            {
                (meta["source"], g)
                for part in meta["partitions"].values()
                for g in part["groups"]
            }
            if meta["groups_supplied"]
            else set()
        )
        if meta["content_hash"] in content or rows & current or groups & current_groups:
            raise ValueError(
                "cohort datasets overlap: aliased content, feature rows or same-source entity groups"
            )
        content.add(meta["content_hash"])
        rows.update(current)
        groups.update(current_groups)


def size_subset(train, kind, fraction):
    n = len(train["y"])
    target = max(2, int(n * fraction))
    rng = np.random.default_rng(SIZE_RECIPE["seed"])
    order = rng.permutation(n).tolist()
    if kind == "classification":
        first = [
            next(i for i in order if train["y"][i] == label)
            for label in sorted(set(train["y"]))
        ]
        order = first + [i for i in order if i not in set(first)]
        if target < len(first):
            raise ValueError("size study subset cannot contain all training classes")
    indices = sorted(order[:target])
    return {
        "role": "train",
        "x": [train["x"][i] for i in indices],
        "y": [train["y"][i] for i in indices],
        "ids": [train["ids"][i] for i in indices],
    }


def builtin_datasets(names):
    from sklearn import datasets

    loaders = {
        "cancer": datasets.load_breast_cancer,
        "wine": datasets.load_wine,
        "digits": datasets.load_digits,
        "iris": datasets.load_iris,
        "diabetes": datasets.load_diabetes,
    }
    if not isinstance(names, (list, tuple)) or len(set(names)) != len(names):
        raise ValueError("distinct dataset names required")
    result = []
    for name in names:
        if name not in loaders:
            raise ValueError("unknown built-in dataset")
        data = loaders[name]()
        kind = "regression" if name == "diabetes" else "classification"
        result.append(
            Dataset(
                name,
                data.data,
                data.target,
                kind,
                1 if kind == "regression" else len(np.unique(data.target)),
                historical_exposure=True,
                source="scikit-learn " + name + " public historical benchmark",
            )
        )
    return result


def baseline_grids(kind):
    if kind not in ("classification", "regression"):
        raise ValueError("unknown task")
    grids = {family: [] for family in FAMILIES}
    if kind == "classification":
        for c, scale, weight in product(
            (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
            ("standard", "robust"),
            (None, "balanced"),
        ):
            grids["linear"].append(
                {
                    "family": "linear",
                    "scale": scale,
                    "params": {"C": c, "class_weight": weight},
                }
            )
    else:
        for alpha, scale in product(
            (
                0.0001,
                0.001,
                0.01,
                0.1,
                1.0,
                10.0,
                100.0,
                1000.0,
                10000.0,
                100000.0,
                1e6,
                1e7,
            ),
            ("standard", "robust"),
        ):
            grids["linear"].append(
                {"family": "linear", "scale": scale, "params": {"alpha": alpha}}
            )
    for c, gamma, scale in product(
        (0.1, 1.0, 10.0, 100.0), ("scale", 0.01, 0.1), ("standard", "robust")
    ):
        grids["kernel"].append(
            {"family": "kernel", "scale": scale, "params": {"C": c, "gamma": gamma}}
        )
    for trees, features, leaf, depth in product(
        (150, 300), ("sqrt", 1.0), (1, 2, 4), (None, 16)
    ):
        grids["extra_trees"].append(
            {
                "family": "extra_trees",
                "scale": "none",
                "params": {
                    "n_estimators": trees,
                    "max_features": features,
                    "min_samples_leaf": leaf,
                    "max_depth": depth,
                },
            }
        )
    for rate, leaves, l2, iterations in product(
        (0.03, 0.1), (15, 31, 63), (0.0, 1.0), (100, 200)
    ):
        grids["boosting"].append(
            {
                "family": "boosting",
                "scale": "none",
                "params": {
                    "learning_rate": rate,
                    "max_leaf_nodes": leaves,
                    "l2_regularization": l2,
                    "max_iter": iterations,
                },
            }
        )
    for neighbors, weights, scale in product(
        (1, 3, 5, 9, 15, 31), ("uniform", "distance"), ("standard", "robust")
    ):
        grids["neighbors"].append(
            {
                "family": "neighbors",
                "scale": scale,
                "params": {"n_neighbors": neighbors, "weights": weights},
            }
        )
    return grids


def paired_statistics(values, alpha=0.05, family_size=1):
    if not isinstance(values, dict) or len(values) < 2 or len(values) > 20:
        raise ValueError("2..20 independent dataset clusters required")
    means = []
    count = 0
    for key, row in sorted(values.items()):
        if not isinstance(key, str) or not isinstance(row, list) or not row:
            raise ValueError("named nonempty paired clusters required")
        clean = [finite(x) for x in row]
        means.append(float(np.mean(clean)))
        count += len(clean)
    alpha = finite(alpha)
    if not 0 < alpha < 1 or type(family_size) is not int or family_size < 1:
        raise ValueError("statistical criteria invalid")
    if not all(math.isfinite(x) for x in means):
        raise ValueError("cluster means overflow")
    n = len(means)
    observed = abs(sum(means))
    total = 1 << n
    # Exact enumeration at dataset level. Seeds never increase n.
    exceed = sum(
        abs(sum(v * s for v, s in zip(means, signs))) >= observed - 1e-12
        for signs in product((-1, 1), repeat=n)
    )
    mean = float(np.mean(means))
    se = float(np.std(means, ddof=1) / math.sqrt(n))
    critical = float(stats.t.ppf(1 - alpha / (2 * family_size), n - 1))
    if not all(
        math.isfinite(x)
        for x in (
            observed,
            mean,
            se,
            critical,
            mean - critical * se,
            mean + critical * se,
        )
    ):
        raise ValueError("statistical calculation overflow")
    return {
        "unit": "independent dataset clusters; seeds averaged within dataset",
        "independent_units": n,
        "paired_runs": count,
        "dataset_means": dict(zip(sorted(values), means)),
        "mean_effect": mean,
        "p_value": exceed / total,
        "p_value_resolution": 2 / total,
        "simultaneous_ci": [mean - critical * se, mean + critical * se],
        "ci_method": "Bonferroni simultaneous Student t across dataset means; independent approximately normal cluster means assumed",
        "test_assumption": "paired dataset sign-flip requires exchangeability/symmetry under the null",
    }


def holm_adjust(pvalues):
    if not isinstance(pvalues, dict) or not pvalues:
        raise ValueError("named p-values required")
    ordered = sorted(((finite(p), name) for name, p in pvalues.items()))
    if any(not 0 <= p <= 1 for p, _ in ordered):
        raise ValueError("p-value out of range")
    out = {}
    previous = 0.0
    for i, (p, name) in enumerate(ordered):
        previous = max(previous, min(1.0, p * (len(ordered) - i)))
        out[name] = previous
    return out


def score_predictions(kind, truth, prediction, normalizer=1.0):
    from sklearn.metrics import balanced_accuracy_score, mean_squared_error

    y = np.asarray(truth)
    p = np.asarray(prediction)
    if (
        y.ndim != 1
        or p.shape != y.shape
        or not len(y)
        or p.dtype.kind not in "iuf"
        or not np.isfinite(p).all()
    ):
        raise ValueError("finite aligned predictions required")
    if kind == "classification":
        if p.dtype.kind not in "iu" or np.any(p < 0):
            raise ValueError(
                "classification predictions must use nonnegative encoded labels"
            )
        return float(balanced_accuracy_score(y, p))
    if kind != "regression" or finite(normalizer) <= 0:
        raise ValueError("metric mismatch")
    return -float(mean_squared_error(y, p)) / normalizer


def _suite():
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.pipeline import make_pipeline

    return locals()


def build_model(spec, kind, features, classes, seed, gen):
    suite = _suite()
    family = spec["family"]
    if family == "generated":
        h = gen.Hypothesis.from_dict(spec["record"])
        a = h.algorithm
        if a.kind != kind:
            raise ValueError("candidate task mismatch")
        from dataclasses import replace

        if spec["mode"] == "fixed":
            a = replace(a, controller=gen.Controller("half"))
        elif spec["mode"] == "first_only":
            a = replace(a, coupling="first_only")
        elif spec["mode"] != "adaptive":
            raise ValueError("unknown ablation")
        task = gen.TaskSpec(
            kind,
            features,
            classes,
            goals=h.task.goals,
            max_parameters=h.task.max_parameters,
            max_state_vectors=h.task.max_state_vectors,
        )
        adapted = gen.Hypothesis(a, task, "construct", (), h.seed, h.round_no)
        return gen.GeneratedModel(
            adapted,
            steps=spec["steps"],
            learning_rate=float(Fraction(exact_rate(spec["learning_rate"]))),
            seed=seed,
        )
    if family not in FAMILIES:
        raise ValueError("unknown estimator family")
    params = dict(spec["params"])
    classification = kind == "classification"
    names = {
        "linear": "LogisticRegression" if classification else "Ridge",
        "kernel": "SVC" if classification else "SVR",
        "extra_trees": (
            "ExtraTreesClassifier" if classification else "ExtraTreesRegressor"
        ),
        "boosting": (
            "HistGradientBoostingClassifier"
            if classification
            else "HistGradientBoostingRegressor"
        ),
        "neighbors": (
            "KNeighborsClassifier" if classification else "KNeighborsRegressor"
        ),
    }
    if family == "linear" and classification:
        params.update(max_iter=2000, random_state=seed)
    if family in ("extra_trees", "boosting"):
        params["random_state"] = seed
    if family in ("extra_trees", "neighbors"):
        params["n_jobs"] = 1
    if family == "kernel" and classification:
        params["random_state"] = seed
    model = suite[names[family]](**params)
    scale = spec["scale"]
    if scale not in ("none", "standard", "robust"):
        raise ValueError("unknown train-only scaler")
    if scale != "none":
        model = suite["make_pipeline"](
            suite["StandardScaler" if scale == "standard" else "RobustScaler"](), model
        )
    return model


def environment():
    import sklearn, scipy

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "threads": 1,
    }


def fit_predict(request):
    """Only labelled TRAIN arrays enter fitting; conditions contain X only."""
    gen = load_generator(request["generator_path"])
    _suite()
    train = request["train"]
    if train.get("role") != "train":
        raise ValueError("fitting requires train role")
    x = np.asarray(train["x"], dtype=float)
    y = np.asarray(train["y"])
    kind = request["kind"]
    spec = request["model"]
    if (
        x.ndim != 2
        or len(x) != len(y)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("training arrays invalid")
    model = build_model(
        spec, kind, x.shape[1], request["classes"], request["seed"], gen
    )
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        start = time.perf_counter()
        begin_ns = time.time_ns()
        if spec["family"] == "generated":
            model.fit(gen.Partition("train", x, y, tuple(train["ids"])))
        else:
            model.fit(x, y)
        fit_seconds = time.perf_counter() - start
        predictions = {}
        predict_times = []
        for condition in request["conditions"]:
            strict(condition, ("name", "x"), ("name", "x"))
            values = np.asarray(condition["x"], dtype=float)
            start = time.perf_counter()
            p = np.asarray(model.predict(values))
            elapsed = time.perf_counter() - start
            if (
                p.ndim != 1
                or len(p) != len(values)
                or p.dtype.kind not in "iuf"
                or not np.isfinite(p).all()
            ):
                raise ValueError("estimator produced invalid predictions")
            predictions[condition["name"]] = p.tolist()
            if condition["name"] == "clean":
                predict_times.append(elapsed)
                for _ in range(4):
                    start = time.perf_counter()
                    again = np.asarray(model.predict(values))
                    predict_times.append(time.perf_counter() - start)
                    if not np.array_equal(again, p):
                        raise ValueError(
                            "same fitted estimator predicts nondeterministically"
                        )
    return {
        "success": True,
        "predictions": predictions,
        "fit_seconds": fit_seconds,
        "predict_seconds": statistics.median(predict_times),
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "fit_begin_ns": begin_ns,
        "fit_end_ns": time.time_ns(),
        "fit_role": "train",
        "fit_row_hash": digest(train["ids"]),
        "warnings": [str(x.message)[:1000] for x in caught],
        "environment": environment(),
    }


def execute_fit(request, isolated, timeout):
    start = time.time_ns()
    try:
        if isolated:
            process = subprocess.run(
                [sys.executable, "-I", str(Path(__file__).resolve()), "_worker"],
                input=canonical(request),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if process.returncode:
                raise ValueError("worker failed: " + process.stderr[-1000:])
            result = json.loads(process.stdout)
            if not result.get("success"):
                raise ValueError(result.get("error", "worker failure"))
            result["resource_scope"] = (
                "isolated worker process; peak RSS includes interpreter, imports, data, fitting and inference"
            )
        else:
            result = fit_predict(request)
            result["resource_scope"] = (
                "in-process diagnostic; peak RSS is not a per-model measurement"
            )
        return result
    except (
        ValueError,
        RuntimeError,
        MemoryError,
        OSError,
        TypeError,
        KeyError,
        subprocess.TimeoutExpired,
    ) as exc:
        return {
            "success": False,
            "error": type(exc).__name__ + ": " + str(exc),
            "fit_begin_ns": start,
            "fit_end_ns": time.time_ns(),
        }


def resource_description(isolated):
    return (
        "fresh isolated worker per final fit; single native thread; peak process RSS includes startup/imports/data"
        if isolated
        else "in-process diagnostic; peak RSS is not a per-model measurement"
    )


def validate_fit_measurements(row, manifest, events):
    if not isinstance(row, dict) or type(row["success"]) is not bool:
        raise ValueError("invalid fit result")
    begin = row["fit_begin_ns"]
    end = row["fit_end_ns"]
    if (
        type(begin) is not int
        or type(end) is not int
        or not events[1]["at_ns"] <= begin <= end <= events[2]["at_ns"]
    ):
        raise ValueError("fit occurred outside consumed test window")
    if not row["success"]:
        if not isinstance(row.get("error"), str) or not row["error"]:
            raise ValueError("failed fit lacks diagnostic")
        return
    if row["environment"] != manifest["environment"]:
        raise ValueError("fit environment changed")
    isolated = manifest["protocol"]["isolated"]
    scope = (
        "isolated worker process; peak RSS includes interpreter, imports, data, fitting and inference"
        if isolated
        else resource_description(False)
    )
    if row["resource_scope"] != scope:
        raise ValueError("resource measurement scope mismatch")
    for field in ("fit_seconds", "predict_seconds"):
        if finite(row[field]) <= 0:
            raise ValueError("nonpositive resource measurement")
    if type(row["peak_rss_kib"]) is not int or row["peak_rss_kib"] <= 0:
        raise ValueError("invalid peak process RSS")


def perturbations(train_x, test_x, seed):
    train = np.asarray(train_x, dtype=float)
    test = np.asarray(test_x, dtype=float)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    rng = np.random.default_rng(seed)
    noise = test + 0.15 * scale * rng.normal(size=test.shape)
    outliers = test.copy()
    mask = rng.random(test.shape) < 0.05
    signs = rng.choice((-1.0, 1.0), size=test.shape)
    replacement = mean + 6 * scale * signs
    outliers[mask] = replacement[mask]
    shifted = test.copy()
    columns = rng.permutation(test.shape[1])[: max(1, math.ceil(0.2 * test.shape[1]))]
    shifted[:, columns] += 0.5 * scale[columns]
    return [
        {"name": name, "x": value.tolist()}
        for name, value in zip(CONDITIONS, (test, noise, outliers, shifted))
    ]


def specification_id(spec):
    return digest(spec)


def select_development(development, datasets):
    """Reconstruct deterministic choices from every attempted development trial."""
    trials = development["trials"]
    choices = {}
    for family in ("candidate", "fixed_search"):
        ids = sorted({t["config_id"] for t in trials if t["family"] == family})
        scores = {}
        for config in ids:
            rows = [
                t for t in trials if t["family"] == family and t["config_id"] == config
            ]
            if (
                len(rows) == len(datasets)
                and {r["dataset"] for r in rows} == set(datasets)
                and all(r["success"] for r in rows)
            ):
                scores[config] = float(np.mean([r["score"] for r in rows]))
        if not scores:
            raise ValueError("no valid complete development candidate")
        choices[family] = min(scores, key=lambda c: (-scores[c], c))
    baselines = {}
    reference = {}
    for dataset in datasets:
        baselines[dataset] = {}
        family_scores = {}
        for family in FAMILIES:
            rows = [
                t
                for t in trials
                if t["dataset"] == dataset and t["family"] == family and t["success"]
            ]
            if not rows:
                raise ValueError("baseline family has no valid development fit")
            winner = min(rows, key=lambda r: (-r["score"], r["config_id"]))
            baselines[dataset][family] = winner["config_id"]
            family_scores[family] = winner["score"]
        reference[dataset] = min(family_scores, key=lambda f: (-family_scores[f], f))
    return {**choices, "baselines": baselines, "reference": reference}


class Journal:
    """One SQLite artifact. Atomic consumption persists before test arrays read."""

    def __init__(self, path, create=True):
        self.path = Path(path).resolve()
        if create:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            with self.connect() as db:
                db.executescript(
                    "CREATE TABLE state(id INTEGER PRIMARY KEY CHECK(id=1),phase TEXT,manifest TEXT,development TEXT,evidence TEXT,uses INTEGER);"
                    "CREATE TABLE vault(name TEXT PRIMARY KEY,train TEXT,development TEXT,test TEXT,meta TEXT);"
                    "CREATE TABLE events(seq INTEGER PRIMARY KEY,kind TEXT,at_ns INTEGER,manifest_hash TEXT,previous_hash TEXT,event_hash TEXT);"
                    "CREATE TABLE review(name TEXT PRIMARY KEY,payload TEXT);"
                )
                db.execute("INSERT INTO state VALUES(1,'NEW',NULL,NULL,NULL,0)")
        elif not self.path.exists():
            raise ValueError("journal missing")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def phase(self):
        with self.connect() as db:
            return db.execute("SELECT phase FROM state WHERE id=1").fetchone()[0]

    def _event(self, db, kind, manifest_hash):
        previous = db.execute(
            "SELECT seq,event_hash FROM events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        seq = 1 if previous is None else previous[0] + 1
        before = "" if previous is None else previous[1]
        body = {
            "seq": seq,
            "kind": kind,
            "at_ns": time.time_ns(),
            "manifest_hash": manifest_hash,
            "previous_hash": before,
        }
        db.execute(
            "INSERT INTO events VALUES(?,?,?,?,?,?)",
            (seq, kind, body["at_ns"], manifest_hash, before, digest(body)),
        )

    def freeze(self, manifest, development):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT phase FROM state WHERE id=1").fetchone()[0] not in (
                "NEW",
                "DEVELOPED",
            ):
                raise ValueError("cannot refreeze study")
            db.execute(
                "UPDATE state SET phase='FROZEN',manifest=?,development=? WHERE id=1",
                (canonical(manifest), canonical(development)),
            )
            self._event(db, "FREEZE", digest(manifest))

    def consume(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            phase, manifest, uses = db.execute(
                "SELECT phase,manifest,uses FROM state WHERE id=1"
            ).fetchone()
            if phase != "FROZEN" or uses != 0:
                raise ValueError("holdout is not frozen or already consumed")
            db.execute("UPDATE state SET phase='CONSUMED',uses=1 WHERE id=1")
            self._event(db, "CONSUME", digest(json.loads(manifest)))

    def events(self):
        with self.connect() as db:
            return [
                dict(
                    zip(
                        (
                            "seq",
                            "kind",
                            "at_ns",
                            "manifest_hash",
                            "previous_hash",
                            "event_hash",
                        ),
                        row,
                    )
                )
                for row in db.execute("SELECT * FROM events ORDER BY seq")
            ]

    def complete(self, evidence):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if (
                db.execute("SELECT phase FROM state WHERE id=1").fetchone()[0]
                != "CONSUMED"
            ):
                raise ValueError("test phase is not consumed")
            self._event(db, "COMPLETE", digest(evidence["manifest"]))
            evidence["events"] = [
                dict(
                    zip(
                        (
                            "seq",
                            "kind",
                            "at_ns",
                            "manifest_hash",
                            "previous_hash",
                            "event_hash",
                        ),
                        row,
                    )
                )
                for row in db.execute("SELECT * FROM events ORDER BY seq")
            ]
            evidence["artifact_hash"] = digest(
                {k: v for k, v in evidence.items() if k != "artifact_hash"}
            )
            db.execute(
                "UPDATE state SET phase='COMPLETE',evidence=? WHERE id=1",
                (canonical(evidence),),
            )

    def abort(self, reason):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            phase, manifest = db.execute(
                "SELECT phase,manifest FROM state WHERE id=1"
            ).fetchone()
            if phase != "CONSUMED":
                raise ValueError("only a consumed experiment can fail terminally")
            failure = {
                "status": "FAILED_AFTER_CONSUME",
                "reason": str(reason)[:4000],
                "holdout_uses": 1,
            }
            self._event(db, "FAILED", digest(json.loads(manifest)))
            db.execute(
                "UPDATE state SET phase='FAILED',evidence=? WHERE id=1",
                (canonical(failure),),
            )

    def evidence(self):
        with self.connect() as db:
            phase, body = db.execute(
                "SELECT phase,evidence FROM state WHERE id=1"
            ).fetchone()
            if phase != "COMPLETE":
                raise ValueError("no completed evidence")
            return json.loads(body)


def invalid_gates(reason):
    return {
        g: {"status": "UNVERIFIED", "reason": reason, "scope": "no validated evidence"}
        for g in GATES
    }


def validate_evidence(evidence, generator_path=None):
    # Full evidence reconstruction is supplied below; this guard also keeps
    # malformed external objects from inventing scientific approvals.
    try:
        if not isinstance(evidence, dict) or evidence.get("schema_version") != VERSION:
            raise ValueError("unsupported/missing evidence schema")
        return _validate_complete_evidence(evidence, generator_path)
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OverflowError,
        RecursionError,
        AttributeError,
        OSError,
    ) as exc:
        return {
            "accepted": False,
            "status": "INVALID_EVIDENCE",
            "gates": invalid_gates(str(exc)),
            "errors": [str(exc)],
        }


def _validate_complete_evidence(evidence, generator_path=None):
    strict(
        evidence,
        (
            "schema_version",
            "manifest",
            "manifest_hash",
            "development",
            "data",
            "runs",
            "replications",
            "size_study",
            "holdout_uses",
            "external_evidence",
            "scientific_gates_before",
            "events",
            "artifact_hash",
        ),
        (
            "manifest",
            "development",
            "data",
            "runs",
            "replications",
            "size_study",
            "events",
            "artifact_hash",
            "holdout_uses",
            "scientific_gates_before",
            "external_evidence",
        ),
    )
    if evidence["artifact_hash"] != digest(
        {k: v for k, v in evidence.items() if k != "artifact_hash"}
    ):
        raise ValueError("artifact hash mismatch")
    manifest = evidence["manifest"]
    protocol = Protocol.parse(manifest["protocol"])
    manifest_fields = (
        "schema_version",
        "protocol",
        "kind",
        "datasets",
        "candidate_id",
        "methods",
        "baselines",
        "reference",
        "development_hash",
        "sources",
        "environment",
        "conditions",
        "corruption_recipe",
        "size_recipe",
        "comparison_family",
        "metric",
        "replication_recipe",
        "resource_protocol",
        "external_review_policy",
        "intake_receipts",
    )
    strict(manifest, manifest_fields, manifest_fields)
    if manifest["schema_version"] != VERSION or manifest["kind"] not in (
        "classification",
        "regression",
    ):
        raise ValueError("unsupported manifest task/schema")
    if canonical(manifest["corruption_recipe"]) != canonical(
        CORRUPTION_RECIPE
    ) or canonical(manifest["size_recipe"]) != canonical(SIZE_RECIPE):
        raise ValueError("unsupported declared test recipe")
    if manifest["metric"] != (
        "balanced_accuracy"
        if manifest["kind"] == "classification"
        else "negative_mse_divided_by_frozen_training_variance"
    ):
        raise ValueError("unsupported metric")
    if manifest["resource_protocol"] != resource_description(protocol.isolated):
        raise ValueError("resource protocol/scope mismatch")
    env = manifest["environment"]
    env_fields = ("python", "numpy", "sklearn", "scipy", "platform", "threads")
    strict(env, env_fields, env_fields)
    if (
        type(env["threads"]) is not int
        or env["threads"] != 1
        or any(
            not isinstance(env[k], str) or not env[k]
            for k in env_fields
            if k != "threads"
        )
    ):
        raise ValueError("environment fields require strict declared types")
    if (
        evidence["scientific_gates_before"] != {g: "UNVERIFIED" for g in GATES}
        or evidence["external_evidence"] != {}
    ):
        raise ValueError("unreviewed scientific approvals injected")
    if evidence["manifest_hash"] != digest(manifest):
        raise ValueError("frozen manifest hash mismatch")
    if manifest["sources"]["generator"] != GENERATOR_SHA256:
        raise ValueError("generator source mismatch")
    if manifest["sources"]["scientific_engine"] != SOURCE_SHA256:
        raise ValueError("scientific engine source mismatch")
    if manifest["conditions"] != list(CONDITIONS) or set(manifest["methods"]) != set(
        METHODS
    ) - set(FAMILIES):
        raise ValueError("experiment design changed")
    comparisons = (
        ["reference"] + list(FAMILIES) + ["fixed_matched", "fixed_search", "first_only"]
    )
    if manifest["comparison_family"] != comparisons:
        raise ValueError("multiplicity family changed")
    names = sorted(manifest["datasets"])
    kind = manifest["kind"]
    if not 2 <= len(names) <= 20 or set(evidence["data"]) != set(names):
        raise ValueError("dataset coverage mismatch")
    if evidence["holdout_uses"] != 1 or type(evidence["holdout_uses"]) is not int:
        raise ValueError("one-shot protocol violated")
    events = evidence["events"]
    if not isinstance(events, list) or [x["kind"] for x in events] != [
        "FREEZE",
        "CONSUME",
        "COMPLETE",
    ]:
        raise ValueError("freeze/consume/complete event history invalid")
    previous = ""
    before = 0
    for seq, event in enumerate(events, 1):
        strict(
            event,
            ("seq", "kind", "at_ns", "manifest_hash", "previous_hash", "event_hash"),
            ("seq", "kind", "at_ns", "manifest_hash", "previous_hash", "event_hash"),
        )
        if (
            type(event["seq"]) is not int
            or event["seq"] != seq
            or type(event["at_ns"]) is not int
            or event["at_ns"] <= before
            or event["manifest_hash"] != evidence["manifest_hash"]
            or event["previous_hash"] != previous
        ):
            raise ValueError("event order/binding invalid")
        if event["event_hash"] != digest(
            {k: v for k, v in event.items() if k != "event_hash"}
        ):
            raise ValueError("event chain altered")
        previous = event["event_hash"]
        before = event["at_ns"]
    # Validate every data role and duplicate/group separation from raw arrays.
    gen = load_generator(generator_path)
    for name in names:
        meta = manifest["datasets"][name]
        data = evidence["data"][name]
        if (
            data["meta"] != meta
            or meta["name"] != name
            or meta["kind"] != kind
            or type(meta["historical_exposure"]) is not bool
            or type(meta["groups_supplied"]) is not bool
        ):
            raise ValueError("dataset metadata mismatch")
        if (
            type(meta["rows"]) is not int
            or not 20 <= meta["rows"] <= 10_000_000
            or type(meta["features"]) is not int
            or not 1 <= meta["features"] <= 4096
            or type(meta["classes"]) is not int
            or not (
                2 <= meta["classes"] <= 256
                if kind == "classification"
                else meta["classes"] == 1
            )
        ):
            raise ValueError("dataset dimensions invalid")
        original_x = [None] * meta["rows"]
        original_y = [None] * meta["rows"]
        ids = set()
        row_hashes = set()
        groups = set()
        for role in ("train", "development", "test"):
            part = data[role]
            decl = meta["partitions"][role]
            if type(decl["size"]) is not int or decl["size"] < 1:
                raise ValueError("partition size must be a positive strict integer")
            if part["role"] != role or part["ids"] != decl["ids"]:
                raise ValueError("partition role/ID mismatch")
            x = np.asarray(part["x"], dtype=float)
            y = np.asarray(part["y"])
            if (
                x.ndim != 2
                or x.shape[1] != meta["features"]
                or y.ndim != 1
                or not len(y)
                or len(x) != len(y)
                or len(x) != decl["size"]
                or len(x) != len(part["ids"])
                or not np.isfinite(x).all()
                or not np.isfinite(y).all()
            ):
                raise ValueError("partition arrays invalid")
            if kind == "classification" and (
                y.dtype.kind not in "iu"
                or np.any(y < 0)
                or np.any(y >= meta["classes"])
            ):
                raise ValueError("dataset labels outside public class schema")
            for identifier, row, target in zip(part["ids"], part["x"], part["y"]):
                suffix = identifier.removeprefix(name + ":")
                index = int(suffix)
                if (
                    identifier != name + ":" + str(index)
                    or not 0 <= index < meta["rows"]
                    or original_x[index] is not None
                ):
                    raise ValueError("original row identity invalid")
                original_x[index] = row
                original_y[index] = target
            if array_hash(x) != decl["x_hash"] or array_hash(y) != decl["y_hash"]:
                raise ValueError("data content hash changed")
            hashes = [gen._row_hash(row) for row in x]
            if hashes != decl["row_hashes"] or len(decl["groups"]) != len(x):
                raise ValueError("row/group provenance changed")
            if (
                len(set(part["ids"])) != len(part["ids"])
                or ids & set(part["ids"])
                or row_hashes & set(hashes)
                or groups & set(decl["groups"])
            ):
                raise ValueError("row or group leakage across roles")
            ids.update(part["ids"])
            row_hashes.update(hashes)
            groups.update(decl["groups"])
        if len(ids) != meta["rows"]:
            raise ValueError("dataset partition coverage invalid")
        if meta["content_hash"] != digest(
            {
                "x": array_hash(np.asarray(original_x, dtype=float)),
                "y": array_hash(np.asarray(original_y)),
            }
        ):
            raise ValueError("original dataset content binding invalid")
        final_y = np.array(data["train"]["y"] + data["development"]["y"])
        normalizer = (
            1.0 if kind == "classification" else max(float(np.var(final_y)), 1e-12)
        )
        dev_normalizer = (
            1.0
            if kind == "classification"
            else max(float(np.var(np.asarray(data["train"]["y"]))), 1e-12)
        )
        if (
            finite(meta["normalizer"]) != normalizer
            or finite(meta["development_normalizer"]) != dev_normalizer
        ):
            raise ValueError("metric normalization used heldout statistics")
    check_cohort(list(evidence["data"].values()))
    development = evidence["development"]
    if (
        Protocol.parse(development["protocol"]) != protocol
        or development["intake_receipts"] != manifest["intake_receipts"]
    ):
        raise ValueError("development protocol/intake binding changed")
    if manifest["development_hash"] != digest(development):
        raise ValueError("development evidence changed after freeze")
    if (
        type(development["end_ns"]) is not int
        or development["end_ns"] >= events[0]["at_ns"]
    ):
        raise ValueError("development finished after freeze")
    configs = development["configurations"]
    family_ids = development["family_configurations"]
    if set(family_ids) != set(("candidate", "fixed_search") + FAMILIES):
        raise ValueError("tuning family missing")
    for key, spec in configs.items():
        if key != specification_id(spec):
            raise ValueError("config identity changed")
        if spec["family"] == "generated":
            h = gen.Hypothesis.from_dict(spec["record"])
            if (
                h.algorithm.kind != kind
                or spec["steps"] != protocol.steps
                or exact_rate(spec["learning_rate"]) not in protocol.learning_rates
            ):
                raise ValueError("candidate config differs from preregistration")
    expected_ids = {f: [] for f in ("candidate", "fixed_search") + FAMILIES}
    for family, mode in (("candidate", "adaptive"), ("fixed_search", "fixed")):
        for record, rate in product(
            development["candidate_records"], protocol.learning_rates
        ):
            spec = {
                "family": "generated",
                "record": record,
                "learning_rate": rate,
                "steps": protocol.steps,
                "mode": mode,
            }
            expected_ids[family].append(specification_id(spec))
    for family, grid in baseline_grids(kind).items():
        expected_ids[family] = [specification_id(x) for x in grid[: protocol.budget]]
    if (
        not isinstance(development["candidate_records"], list)
        or not 1 <= len(development["candidate_records"]) <= protocol.budget // 2
        or len({digest(c) for c in development["candidate_records"]})
        != len(development["candidate_records"])
    ):
        raise ValueError("candidate pool has duplicates/exceeds frozen budget")
    for packet in manifest["intake_receipts"]:
        strict(
            packet,
            ("candidate", "math_admission", "scientific_status", "handoff_hash"),
            ("candidate", "math_admission", "scientific_status", "handoff_hash"),
        )
        if (
            packet["candidate"] not in development["candidate_records"]
            or packet["scientific_status"] != "NOT_EVALUATED"
            or packet["handoff_hash"]
            != digest(
                {"candidate": packet["candidate"], "receipt": packet["math_admission"]}
            )
        ):
            raise ValueError("upstream handoff content changed")
    if family_ids != expected_ids or set(configs) != set(
        x for ids in expected_ids.values() for x in ids
    ):
        raise ValueError("declared tuning grids changed")
    trial_seen = set()
    for trial in development["trials"]:
        key = (trial["dataset"], trial["family"], trial["config_id"])
        if key in trial_seen:
            raise ValueError("duplicate development experiment")
        trial_seen.add(key)
        if (
            trial["dataset"] not in names
            or trial["config_id"] not in family_ids[trial["family"]]
        ):
            raise ValueError("unexpected development config")
        data = evidence["data"][trial["dataset"]]
        meta = manifest["datasets"][trial["dataset"]]
        if (
            trial["fit_role"] != "train"
            or trial["fit_row_hash"] != digest(data["train"]["ids"])
            or trial["validation_row_hash"] != digest(data["development"]["ids"])
        ):
            raise ValueError("development role leak")
        if (
            type(trial["success"]) is not bool
            or type(trial["end_ns"]) is not int
            or trial["end_ns"] > development["end_ns"]
            or trial["end_ns"] >= events[0]["at_ns"]
        ):
            raise ValueError("development timing/status invalid")
        if trial["success"]:
            if finite(trial["fit_seconds"]) <= 0:
                raise ValueError("invalid development fit time")
            if kind == "classification" and any(
                type(x) is not int or not 0 <= x < meta["classes"]
                for x in trial["predictions"]
            ):
                raise ValueError("invalid predicted class")
            score = score_predictions(
                kind,
                data["development"]["y"],
                trial["predictions"],
                meta["development_normalizer"],
            )
            if (
                trial["prediction_hash"] != digest(trial["predictions"])
                or abs(finite(trial["score"]) - score) > 1e-12
            ):
                raise ValueError("forged development score")
    expected_trials = {
        (d, f, c) for d in names for f, ids in family_ids.items() for c in ids
    }
    if trial_seen != expected_trials:
        raise ValueError("incomplete development search")
    choices = select_development(development, names)
    if (
        choices != development["choices"]
        or choices["candidate"] != manifest["candidate_id"]
        or choices["reference"] != manifest["reference"]
    ):
        raise ValueError("post-hoc candidate/reference selection")
    if canonical(manifest["methods"]["candidate"]) != canonical(
        configs[choices["candidate"]]
    ) or canonical(manifest["methods"]["fixed_search"]) != canonical(
        configs[choices["fixed_search"]]
    ):
        raise ValueError("selected model changed")
    for method, mode in (("fixed_matched", "fixed"), ("first_only", "first_only")):
        if canonical(manifest["methods"][method]) != canonical(
            {
                **manifest["methods"]["candidate"],
                "mode": mode,
            }
        ):
            raise ValueError("ablation changed more than intended mechanism")
    for name in names:
        if set(manifest["baselines"][name]) != set(FAMILIES):
            raise ValueError("baseline coverage incomplete")
        for family in FAMILIES:
            if canonical(manifest["baselines"][name][family]) != canonical(
                configs[choices["baselines"][name][family]]
            ):
                raise ValueError("baseline not development winner")
    runs = {}
    condition_cache = {}
    failures = []
    for row in evidence["runs"]:
        key = (row["dataset"], row["seed"], row["method"])
        if key in runs:
            raise ValueError("duplicate heldout experiment")
        if (
            type(row["seed"]) is not int
            or row["dataset"] not in names
            or row["seed"] not in protocol.seeds
            or row["method"] not in METHODS
        ):
            raise ValueError("unexpected heldout cell")
        runs[key] = row
        data = evidence["data"][row["dataset"]]
        meta = manifest["datasets"][row["dataset"]]
        spec = (
            manifest["methods"][row["method"]]
            if row["method"] in manifest["methods"]
            else manifest["baselines"][row["dataset"]][row["method"]]
        )
        if row["config_id"] != specification_id(spec) or row["test_row_hash"] != digest(
            data["test"]["ids"]
        ):
            raise ValueError("heldout model/data identity mismatch")
        if type(row["success"]) is not bool:
            raise ValueError("Boolean experiment success required")
        validate_fit_measurements(row, manifest, events)
        if not row["success"]:
            failures.append(key)
            continue
        if row["fit_role"] != "train" or row["fit_row_hash"] != digest(
            data["train"]["ids"] + data["development"]["ids"]
        ):
            raise ValueError("test labels entered fitting")
        if set(row["predictions"]) != set(CONDITIONS) or set(row["scores"]) != set(
            CONDITIONS
        ):
            raise ValueError("missing test corruption/score")
        if row["environment"] != manifest["environment"]:
            raise ValueError("unmatched resource environment")
        for field in ("fit_seconds", "predict_seconds"):
            if finite(row[field]) <= 0:
                raise ValueError("nonpositive resource measurement")
        if type(row["peak_rss_kib"]) is not int or row["peak_rss_kib"] <= 0:
            raise ValueError("invalid peak process RSS")
        corruption_key = (row["dataset"], row["seed"])
        if corruption_key not in condition_cache:
            seed = int(digest([row["dataset"], row["seed"], "corruptions-v1"])[:8], 16)
            condition_arrays = perturbations(
                data["train"]["x"] + data["development"]["x"], data["test"]["x"], seed
            )
            condition_cache[corruption_key] = {
                x["name"]: array_hash(np.array(x["x"], dtype=float))
                for x in condition_arrays
            }
        expected_hashes = condition_cache[corruption_key]
        if row["condition_input_hashes"] != expected_hashes:
            raise ValueError("test stress inputs differ between methods")
        for condition in CONDITIONS:
            p = row["predictions"][condition]
            if kind == "classification" and any(
                type(x) is not int or not 0 <= x < meta["classes"] for x in p
            ):
                raise ValueError("invalid predicted class")
            if row["prediction_hashes"][condition] != digest(p):
                raise ValueError("prediction hash mismatch")
            score = score_predictions(kind, data["test"]["y"], p, meta["normalizer"])
            if abs(finite(row["scores"][condition]) - score) > 1e-12:
                raise ValueError("heldout score inconsistent with raw predictions")
    if set(runs) != {(d, s, m) for d in names for s in protocol.seeds for m in METHODS}:
        raise ValueError("missing heldout experiment cells")
    gates = invalid_gates("external independent evidence not supplied")

    def gate(name, status, reason, scope):
        gates[name] = {"status": status, "reason": reason, "scope": scope}

    replications = evidence["replications"]
    reproduced = True
    if len(replications) != len(names) or {x["dataset"] for x in replications} != set(
        names
    ):
        raise ValueError("missing/duplicate planned deterministic refit")
    for row in replications:
        original = runs[row["dataset"], protocol.seeds[0], "candidate"]
        fit = row["fit"]
        if (
            type(row["seed"]) is not int
            or row["seed"] != protocol.seeds[0]
            or row["config_id"] != original["config_id"]
            or row["same_machine"] is not True
        ):
            raise ValueError("deterministic refit binding invalid")
        validate_fit_measurements(fit, manifest, events)
        if type(row["success"]) is not bool or row["success"] != fit["success"]:
            raise ValueError("deterministic refit success forged")
        match = (
            original["success"]
            and fit["success"]
            and canonical(fit["predictions"]) == canonical(original["predictions"])
        )
        if (
            type(row["prediction_hash_match"]) is not bool
            or row["prediction_hash_match"] != match
        ):
            raise ValueError("deterministic refit match forged")
        if fit["success"] and (
            fit["fit_role"] != "train"
            or fit["fit_row_hash"] != original["fit_row_hash"]
        ):
            raise ValueError("deterministic refit training identity invalid")
        reproduced &= bool(match)
    sizes = {}
    size_failures = []
    for row in evidence["size_study"]:
        key = (row["dataset"], row["seed"], row["method"], row["fraction"])
        if (
            row["dataset"] not in names
            or type(row["seed"]) is not int
            or row["seed"] not in protocol.seeds
            or row["method"] not in ("candidate", "reference")
            or type(row["fraction"]) is not float
            or row["fraction"] not in SIZE_FRACTIONS
            or key in sizes
        ):
            raise ValueError("unexpected/duplicate size-study cell")
        sizes[key] = row
        name = row["dataset"]
        data = evidence["data"][name]
        meta = manifest["datasets"][name]
        train = {
            field: data["train"][field] + data["development"][field]
            for field in ("x", "y", "ids")
        }
        train["role"] = "train"
        subset = size_subset(train, kind, row["fraction"])
        method = (
            "candidate" if row["method"] == "candidate" else manifest["reference"][name]
        )
        original = runs[name, row["seed"], method]
        if (
            type(row["training_rows"]) is not int
            or row["config_id"] != original["config_id"]
            or row["training_rows"] != len(subset["ids"])
        ):
            raise ValueError("size-study configuration changed")
        validate_fit_measurements(row, manifest, events)
        if not row["success"]:
            size_failures.append(key)
            continue
        if (
            row["fit_role"] != "train"
            or row["fit_row_hash"] != digest(subset["ids"])
            or set(row["predictions"]) != {"clean"}
        ):
            raise ValueError("size-study roles changed")
        p = row["predictions"]["clean"]
        if len(p) != len(data["test"]["ids"]) or not np.isfinite(np.asarray(p)).all():
            raise ValueError("size-study predictions invalid")
        if kind == "classification" and any(
            type(x) is not int or not 0 <= x < meta["classes"] for x in p
        ):
            raise ValueError("size-study class schema invalid")
    if set(sizes) != {
        (d, s, m, f)
        for d in names
        for s in protocol.seeds
        for m in ("candidate", "reference")
        for f in SIZE_FRACTIONS
    }:
        raise ValueError("missing size-study cells")
    candidate_failed = any(m == "candidate" for _, _, m in failures)
    gate(
        "working_implementation",
        "PASS" if not candidate_failed and reproduced else "FAIL",
        (
            "complete candidate predictions plus deterministic same-environment refits"
            if reproduced
            else "failed/missing candidate refits"
        ),
        "specified experiment cells, pinned source and environment only",
    )
    fair = (
        protocol.budget >= 24
        and all(len(family_ids[f]) >= len(family_ids["candidate"]) for f in FAMILIES)
        and len(family_ids["fixed_search"]) == len(family_ids["candidate"])
        and not failures
        and all(x["success"] for x in development["trials"])
    )
    gate(
        "fair_benchmarking",
        "PASS" if fair else "FAIL",
        "five development-tuned baseline families, matched fixed-controller structure/rate search and frozen references",
        "declared finite tuning grids; no claim of globally optimal tuning",
    )
    statistics_out = {}
    robustness = {}
    efficiency = {}
    if not failures:
        for comparison in comparisons:
            delta = {}
            for dataset in names:
                method = (
                    manifest["reference"][dataset]
                    if comparison == "reference"
                    else comparison
                )
                delta[dataset] = [
                    runs[dataset, s, "candidate"]["scores"]["clean"]
                    - runs[dataset, s, method]["scores"]["clean"]
                    for s in protocol.seeds
                ]
            statistics_out[comparison] = paired_statistics(
                delta, protocol.alpha, len(comparisons)
            )
        corrected = holm_adjust(
            {name: x["p_value"] for name, x in statistics_out.items()}
        )
        for name, adjusted in corrected.items():
            statistics_out[name]["holm_p"] = adjusted
        target_names = ("reference", "fixed_search")
        statistical = all(
            statistics_out[n]["holm_p"] <= protocol.alpha
            and statistics_out[n]["mean_effect"] > 0
            for n in target_names
        )
        meaningful = all(
            statistics_out[n]["mean_effect"] >= protocol.primary_minimum
            and statistics_out[n]["simultaneous_ci"][0] > 0
            for n in target_names
        )
        gate(
            "statistical_validation",
            "PASS" if statistical else "FAIL",
            "dataset-clustered exact sign-flip with Holm correction across the complete frozen family",
            "symmetric/exchangeable dataset-null assumption; seeds are not independent datasets",
        )
        unseen = all(
            not manifest["datasets"][name]["historical_exposure"] for name in names
        )
        gate(
            "accuracy_generalization",
            (
                "PASS"
                if statistical and meaningful and unseen
                else "FAIL" if not meaningful else "UNVERIFIED"
            ),
            (
                "required positive effect/interval/significance and genuinely unexposed datasets"
                if unseen
                else "historically exposed datasets; retrospective benchmark only"
            ),
            "preregistered heldout cohort; not universal future generalization",
        )
        robust_pass = True
        resource_pass = protocol.isolated and not size_failures
        for name in names:
            reference = manifest["reference"][name]
            clean = float(
                np.mean(
                    [
                        runs[name, s, "candidate"]["scores"]["clean"]
                        for s in protocol.seeds
                    ]
                )
            )
            drops = {
                c: clean
                - float(
                    np.mean(
                        [
                            runs[name, s, "candidate"]["scores"][c]
                            for s in protocol.seeds
                        ]
                    )
                )
                for c in CONDITIONS[1:]
            }
            relative = {
                c: float(
                    np.mean(
                        [
                            runs[name, s, "candidate"]["scores"][c]
                            - runs[name, s, reference]["scores"][c]
                            for s in protocol.seeds
                        ]
                    )
                )
                for c in CONDITIONS[1:]
            }
            robustness[name] = {"drops": drops, "relative_effects": relative}
            robust_pass &= (
                max(drops.values()) <= protocol.max_corruption_drop
                and min(relative.values()) >= -protocol.max_corruption_disadvantage
            )
            ratios = {
                field: statistics.median(
                    runs[name, s, "candidate"][field] / runs[name, s, reference][field]
                    for s in protocol.seeds
                )
                for field in ("fit_seconds", "predict_seconds", "peak_rss_kib")
            }
            size_results = []
            for fraction in SIZE_FRACTIONS:
                if any(
                    not sizes[name, s, m, fraction]["success"]
                    for s in protocol.seeds
                    for m in ("candidate", "reference")
                ):
                    size_results.append({"fraction": fraction, "status": "FAILED"})
                    continue
                cell = {
                    "fraction": fraction,
                    "training_rows": sizes[
                        name, protocol.seeds[0], "candidate", fraction
                    ]["training_rows"],
                }
                for field in ("fit_seconds", "predict_seconds", "peak_rss_kib"):
                    cell[field + "_ratio"] = statistics.median(
                        sizes[name, s, "candidate", fraction][field]
                        / sizes[name, s, "reference", fraction][field]
                        for s in protocol.seeds
                    )
                resource_pass &= (
                    cell["fit_seconds_ratio"] <= protocol.max_fit_ratio
                    and cell["predict_seconds_ratio"] <= protocol.max_predict_ratio
                    and cell["peak_rss_kib_ratio"] <= protocol.max_rss_ratio
                )
                size_results.append(cell)
            growth = None
            if not size_failures:
                growth = statistics.median(
                    runs[name, s, "candidate"]["fit_seconds"]
                    / sizes[name, s, "candidate", SIZE_FRACTIONS[0]]["fit_seconds"]
                    for s in protocol.seeds
                )
                resource_pass &= growth <= protocol.max_fit_growth
            efficiency[name] = {
                "full_size_ratios": ratios,
                "size_results": size_results,
                "candidate_fit_growth_4x_rows": growth,
            }
            resource_pass &= (
                ratios["fit_seconds"] <= protocol.max_fit_ratio
                and ratios["predict_seconds"] <= protocol.max_predict_ratio
                and ratios["peak_rss_kib"] <= protocol.max_rss_ratio
            )
        gate(
            "robustness",
            "PASS" if robust_pass else "FAIL",
            "noise/outlier/feature-corruption drop and baseline-relative criteria",
            "specified synthetic corruptions of real heldout data; real domain shift remains unverified",
        )
        gate(
            "efficiency",
            "PASS" if resource_pass else "FAIL",
            (
                "prespecified fit/predict/RSS ratios at three training sizes and fitting growth criterion"
                if resource_pass
                else "resource ratio/growth criterion not met, failed study cells or resources not isolated"
            ),
            "measured finite training sizes; isolated process peak RSS includes interpreter/import/data; no asymptotic scalability claim",
        )
    else:
        for name in (
            "accuracy_generalization",
            "robustness",
            "efficiency",
            "statistical_validation",
        ):
            gate(
                name,
                "FAIL",
                "failed planned evaluation cells; missing results are not imputed",
                "declared complete experiment matrix",
            )
    gate(
        "novelty",
        "UNVERIFIED",
        "no independently established prior-art comparison",
        "structural uniqueness and benchmark wins do not establish novelty",
    )
    gate(
        "mathematical_foundation",
        "UNVERIFIED",
        "full proof applicability is not established by empirical runs",
        "conditional math-stage receipts require separate scrutiny",
    )
    gate(
        "independent_reproducibility",
        "UNVERIFIED",
        "same-machine deterministic refits do not establish independent replication",
        "external provenance and independent results required",
    )
    gate(
        "real_world_validation",
        "UNVERIFIED",
        "offline public-data experiments are not deployment/domain validation",
        "external application evidence required",
    )
    accepted = all(g["status"] == "PASS" for g in gates.values())
    return {
        "accepted": accepted,
        "status": (
            "ACCEPTED"
            if accepted
            else (
                "NOT_DEMONSTRATED"
                if any(g["status"] == "FAIL" for g in gates.values())
                else "REQUIRES_EXTERNAL_REVIEW"
            )
        ),
        "gates": gates,
        "statistics": statistics_out,
        "robustness": robustness,
        "efficiency": efficiency,
        "complete_cells": len(runs),
        "size_study_cells": len(sizes),
        "fit_failures": failures,
        "size_study_failures": size_failures,
        "errors": [],
        "evidence_hash": evidence["artifact_hash"],
        "manifest_hash": evidence["manifest_hash"],
    }


class ScientificEngine:
    def __init__(self, path, generator_path=None, protocol=None, create=True):
        self.generator_path = Path(
            generator_path
            or os.environ.get("SCIENCE_GENERATOR")
            or Path(__file__).resolve().parent.parent
            / "generator_only"
            / "hypothesis_generator.py"
        ).resolve()
        self.gen = load_generator(self.generator_path)
        self.protocol = Protocol.parse(protocol)
        self.journal = Journal(path, create=create)
        self.intake_receipts = []
        if not create:
            with self.journal.connect() as db:
                row = db.execute(
                    "SELECT manifest,development FROM state WHERE id=1"
                ).fetchone()
            stored = (
                json.loads(row[0])["protocol"]
                if row[0]
                else json.loads(row[1])["protocol"] if row[1] else None
            )
            if stored is not None:
                restored = Protocol.parse(stored)
                if protocol is not None and restored != self.protocol:
                    raise ValueError("cannot change persisted study protocol")
                self.protocol = restored

    def _development_parts(self):
        if self.journal.phase() not in ("NEW", "DEVELOPING", "DEVELOPED"):
            raise ValueError("development is closed")
        with self.journal.connect() as db:
            rows = db.execute(
                "SELECT name,train,development,meta FROM vault ORDER BY name"
            ).fetchall()
        return {
            name: {
                "train": json.loads(train),
                "development": json.loads(dev),
                "meta": json.loads(meta),
            }
            for name, train, dev, meta in rows
        }

    def develop(self, datasets, candidates=None):
        if self.journal.phase() != "NEW":
            raise ValueError("development cannot be rerun in this study")
        if (
            not isinstance(datasets, list)
            or not 2 <= len(datasets) <= 20
            or len({d.name for d in datasets}) != len(datasets)
        ):
            raise ValueError("2..20 distinctly named datasets required")
        if len({d.kind for d in datasets}) != 1:
            raise ValueError("one task kind per scientific cohort")
        kind = datasets[0].kind
        protocol = self.protocol
        prepared = [prepare_dataset(d, protocol, self.gen) for d in datasets]
        check_cohort(prepared)
        with self.journal.connect() as db:
            for part in prepared:
                db.execute(
                    "INSERT INTO vault VALUES(?,?,?,?,?)",
                    (
                        part["meta"]["name"],
                        canonical(part["train"]),
                        canonical(part["development"]),
                        canonical(part["test"]),
                        canonical(part["meta"]),
                    ),
                )
            db.execute("UPDATE state SET phase='DEVELOPING' WHERE id=1")
        if candidates is None:
            task = self.gen.TaskSpec(
                kind,
                8,
                3 if kind == "classification" else 1,
                goals=("outliers", "ill_conditioned"),
            )
            candidates = [
                h.to_dict()
                for h in self.gen.HypothesisGenerator(
                    protocol.generation_seed
                ).generate(task, protocol.budget // 2)
            ]
        else:
            if (
                not isinstance(candidates, list)
                or not 1 <= len(candidates) <= protocol.budget // 2
            ):
                raise ValueError("bounded candidate records required")
            candidates = [
                self.gen.Hypothesis.from_dict(c).to_dict() for c in candidates
            ]
        if any(c["algorithm"]["kind"] != kind for c in candidates):
            raise ValueError("candidate cohort task mismatch")
        configs = {}
        family_ids = {f: [] for f in ("candidate", "fixed_search") + FAMILIES}
        for family, mode in (("candidate", "adaptive"), ("fixed_search", "fixed")):
            for candidate, rate in product(candidates, protocol.learning_rates):
                spec = {
                    "family": "generated",
                    "record": candidate,
                    "learning_rate": rate,
                    "steps": protocol.steps,
                    "mode": mode,
                }
                key = specification_id(spec)
                configs[key] = spec
                family_ids[family].append(key)
        for family, grid in baseline_grids(kind).items():
            for spec in grid[: protocol.budget]:
                key = specification_id(spec)
                configs[key] = spec
                family_ids[family].append(key)
        development = {
            "protocol": asdict(protocol),
            "intake_receipts": copy.deepcopy(self.intake_receipts),
            "trials": [],
            "configurations": configs,
            "family_configurations": family_ids,
            "candidate_records": candidates,
            "selection_rule": "maximum mean development score; lexical config ID resolves ties; no heldout access",
        }
        for name, data in self._development_parts().items():
            validation = data["development"]
            train = data["train"]
            meta = data["meta"]
            for family, ids in family_ids.items():
                for key in ids:
                    request = {
                        "generator_path": str(self.generator_path),
                        "train": train,
                        "conditions": [{"name": "development", "x": validation["x"]}],
                        "kind": kind,
                        "classes": meta["classes"],
                        "model": configs[key],
                        "seed": protocol.development_seed,
                    }
                    # Development predictions need no test timing repetitions.
                    request["conditions"][0]["name"] = "clean"
                    result = execute_fit(request, False, protocol.timeout_seconds)
                    trial = {
                        "dataset": name,
                        "family": family,
                        "config_id": key,
                        "success": result["success"],
                        "fit_row_hash": digest(train["ids"]),
                        "validation_row_hash": digest(validation["ids"]),
                        "fit_role": "train",
                        "end_ns": result["fit_end_ns"],
                    }
                    if result["success"]:
                        prediction = result["predictions"]["clean"]
                        trial.update(
                            predictions=prediction,
                            prediction_hash=digest(prediction),
                            score=score_predictions(
                                kind,
                                validation["y"],
                                prediction,
                                meta["development_normalizer"],
                            ),
                            fit_seconds=result["fit_seconds"],
                            warnings=result["warnings"],
                        )
                    else:
                        trial.update(score=None, error=result["error"])
                    development["trials"].append(trial)
            print(
                canonical(
                    {
                        "phase": "development",
                        "dataset": name,
                        "completed_trials": len(development["trials"]),
                    }
                ),
                flush=True,
            )
        names = sorted(d.name for d in datasets)
        development["choices"] = select_development(development, names)
        development["end_ns"] = time.time_ns()
        with self.journal.connect() as db:
            db.execute(
                "UPDATE state SET phase='DEVELOPED',development=? WHERE id=1",
                (canonical(development),),
            )
        return development["choices"]

    def accept_handoff(self, packet, math_engine):
        """Verify a live mathematical handoff; it remains scoped evidence only."""
        if self.journal.phase() != "NEW":
            raise ValueError("candidate intake is closed")
        validated = math_engine.handoff(packet["candidate"], packet["math_admission"])
        if validated["handoff_hash"] != packet["handoff_hash"]:
            raise ValueError("mathematical handoff mismatch")
        self.intake_receipts.append(copy.deepcopy(validated))
        return self.gen.Hypothesis.from_dict(packet["candidate"]).to_dict()

    def freeze(self):
        if self.journal.phase() != "DEVELOPED":
            raise ValueError("development must finish before freeze")
        parts = self._development_parts()
        with self.journal.connect() as db:
            development = json.loads(
                db.execute("SELECT development FROM state WHERE id=1").fetchone()[0]
            )
        if Protocol.parse(development["protocol"]) != self.protocol:
            raise ValueError("development protocol changed before freeze")
        choices = development["choices"]
        configs = development["configurations"]
        selected = copy.deepcopy(configs[choices["candidate"]])
        fixed_matched = {**copy.deepcopy(selected), "mode": "fixed"}
        first_only = {**copy.deepcopy(selected), "mode": "first_only"}
        methods = {
            "candidate": selected,
            "fixed_matched": fixed_matched,
            "fixed_search": copy.deepcopy(configs[choices["fixed_search"]]),
            "first_only": first_only,
        }
        manifest = {
            "schema_version": VERSION,
            "protocol": asdict(self.protocol),
            "kind": next(iter(parts.values()))["meta"]["kind"],
            "datasets": {name: part["meta"] for name, part in parts.items()},
            "candidate_id": choices["candidate"],
            "methods": methods,
            "baselines": {
                name: {
                    family: configs[key]
                    for family, key in choices["baselines"][name].items()
                }
                for name in parts
            },
            "reference": choices["reference"],
            "development_hash": digest(development),
            "sources": {
                "generator": GENERATOR_SHA256,
                "scientific_engine": SOURCE_SHA256,
            },
            "environment": environment(),
            "conditions": list(CONDITIONS),
            "corruption_recipe": copy.deepcopy(CORRUPTION_RECIPE),
            "size_recipe": copy.deepcopy(SIZE_RECIPE),
            "comparison_family": ["reference"]
            + list(FAMILIES)
            + ["fixed_matched", "fixed_search", "first_only"],
            "metric": (
                "balanced_accuracy"
                if next(iter(parts.values()))["meta"]["kind"] == "classification"
                else "negative_mse_divided_by_frozen_training_variance"
            ),
            "replication_recipe": "refit candidate at first seed on same machine; no claim of independent replication",
            "resource_protocol": resource_description(self.protocol.isolated),
            "intake_receipts": copy.deepcopy(development["intake_receipts"]),
            "external_review_policy": "novelty, full mathematical validity, independence and deployment cannot be inferred from this benchmark",
        }
        self.journal.freeze(manifest, development)
        return manifest

    def evaluate(self):
        try:
            return self._evaluate()
        except Exception as exc:
            if self.journal.phase() == "CONSUMED":
                self.journal.abort(type(exc).__name__ + ": " + str(exc))
            raise

    def _evaluate(self):
        with self.journal.connect() as db:
            row = db.execute(
                "SELECT manifest,development FROM state WHERE id=1"
            ).fetchone()
        if row[0] is None:
            raise ValueError("no frozen experiment")
        manifest = json.loads(row[0])
        development = json.loads(row[1])
        if manifest["sources"] != {
            "generator": file_hash(self.generator_path),
            "scientific_engine": file_hash(__file__),
        }:
            raise ValueError("source changed after preregistration")
        if manifest["environment"] != environment():
            raise ValueError("execution environment changed after freeze")
        self.journal.consume()  # commits before read_heldout or predictions
        runs = []
        data_evidence = {}
        replications = []
        size_study = []
        protocol = Protocol.parse(manifest["protocol"])
        for name, meta in manifest["datasets"].items():
            parts = self.read_heldout(name)
            train = {
                "role": "train",
                "x": parts["train"]["x"] + parts["development"]["x"],
                "y": parts["train"]["y"] + parts["development"]["y"],
                "ids": parts["train"]["ids"] + parts["development"]["ids"],
            }
            test = parts["test"]
            data_evidence[name] = parts
            for seed in protocol.seeds:
                corruption_seed = int(digest([name, seed, "corruptions-v1"])[:8], 16)
                conditions = perturbations(train["x"], test["x"], corruption_seed)
                inputs = {
                    c["name"]: array_hash(np.array(c["x"], dtype=float))
                    for c in conditions
                }
                for method in METHODS:
                    spec = (
                        manifest["methods"][method]
                        if method in manifest["methods"]
                        else manifest["baselines"][name][method]
                    )
                    request = {
                        "generator_path": str(self.generator_path),
                        "train": train,
                        "conditions": conditions,
                        "kind": manifest["kind"],
                        "classes": meta["classes"],
                        "model": spec,
                        "seed": seed,
                    }
                    fit = execute_fit(
                        request, protocol.isolated, protocol.timeout_seconds
                    )
                    run = {
                        "dataset": name,
                        "seed": seed,
                        "method": method,
                        "config_id": specification_id(spec),
                        "test_row_hash": digest(test["ids"]),
                        "condition_input_hashes": inputs,
                        **fit,
                    }
                    if fit["success"]:
                        run["scores"] = {
                            condition: score_predictions(
                                meta["kind"], test["y"], p, meta["normalizer"]
                            )
                            for condition, p in fit["predictions"].items()
                        }
                        run["prediction_hashes"] = {
                            condition: digest(p)
                            for condition, p in fit["predictions"].items()
                        }
                    runs.append(run)
                    if method == "candidate" and seed == protocol.seeds[0]:
                        repeat = execute_fit(
                            request, protocol.isolated, protocol.timeout_seconds
                        )
                        replications.append(
                            {
                                "dataset": name,
                                "seed": seed,
                                "config_id": run["config_id"],
                                "same_machine": True,
                                "success": repeat["success"],
                                "fit": repeat,
                                "prediction_hash_match": fit["success"]
                                and repeat["success"]
                                and digest(fit["predictions"])
                                == digest(repeat["predictions"]),
                            }
                        )
                print(
                    canonical(
                        {
                            "phase": "heldout",
                            "dataset": name,
                            "seed": seed,
                            "completed_fits": len(runs),
                        }
                    ),
                    flush=True,
                )
            for fraction in SIZE_FRACTIONS:
                subset = size_subset(train, manifest["kind"], fraction)
                for seed, method in product(protocol.seeds, ("candidate", "reference")):
                    spec = (
                        manifest["methods"]["candidate"]
                        if method == "candidate"
                        else manifest["baselines"][name][manifest["reference"][name]]
                    )
                    request = {
                        "generator_path": str(self.generator_path),
                        "train": subset,
                        "conditions": [{"name": "clean", "x": test["x"]}],
                        "kind": manifest["kind"],
                        "classes": meta["classes"],
                        "model": spec,
                        "seed": seed,
                    }
                    fit = execute_fit(
                        request, protocol.isolated, protocol.timeout_seconds
                    )
                    size_study.append(
                        {
                            "dataset": name,
                            "seed": seed,
                            "method": method,
                            "fraction": fraction,
                            "training_rows": len(subset["ids"]),
                            "config_id": specification_id(spec),
                            **fit,
                        }
                    )
            print(
                canonical(
                    {
                        "phase": "size_study",
                        "dataset": name,
                        "completed_fits": len(size_study),
                    }
                ),
                flush=True,
            )
        evidence = {
            "schema_version": VERSION,
            "manifest": manifest,
            "manifest_hash": digest(manifest),
            "development": development,
            "data": data_evidence,
            "runs": runs,
            "replications": replications,
            "size_study": size_study,
            "holdout_uses": 1,
            "external_evidence": {},
            "scientific_gates_before": {g: "UNVERIFIED" for g in GATES},
        }
        self.journal.complete(evidence)
        return validate_evidence(self.journal.evidence(), self.generator_path)

    def read_heldout(self, name):
        if self.journal.phase() != "CONSUMED":
            raise ValueError("heldout arrays unavailable before atomic consume")
        with self.journal.connect() as db:
            row = db.execute(
                "SELECT train,development,test,meta FROM vault WHERE name=?", (name,)
            ).fetchone()
        if row is None:
            raise ValueError("unknown heldout dataset")
        return dict(zip(("train", "development", "test", "meta"), map(json.loads, row)))


if __name__ == "__main__":

    def main():
        if len(sys.argv) == 2 and sys.argv[1] == "_worker":
            try:
                request = json.load(sys.stdin)
                result = fit_predict(request)
            except Exception as exc:
                result = {
                    "success": False,
                    "error": type(exc).__name__ + ": " + str(exc),
                }
            print(canonical(result))
            return
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--generator", type=Path, help="reviewed generator source; SHA-256 checked"
        )
        commands = parser.add_subparsers(dest="command", required=True)
        for command in ("demo", "develop"):
            p = commands.add_parser(command)
            p.add_argument("artifact", type=Path)
            p.add_argument(
                "--datasets",
                nargs="+",
                choices=("cancer", "wine", "digits", "iris", "diabetes"),
                default=["cancer", "wine", "digits", "iris"],
            )
            p.add_argument("--budget", type=int, default=24)
            p.add_argument("--steps", type=int, default=160)
            p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        for command in ("freeze", "evaluate", "inspect"):
            p = commands.add_parser(command)
            p.add_argument("artifact", type=Path)
        args = parser.parse_args()
        if args.command in ("demo", "develop"):
            engine = ScientificEngine(
                args.artifact,
                args.generator,
                {"budget": args.budget, "steps": args.steps, "seeds": args.seeds},
            )
            engine.develop(builtin_datasets(args.datasets))
            if args.command == "develop":
                print(
                    canonical(
                        {
                            "phase": engine.journal.phase(),
                            "artifact": str(engine.journal.path),
                        }
                    )
                )
                return
            manifest = engine.freeze()
            print(
                canonical(
                    {
                        "phase": "FROZEN",
                        "manifest_hash": digest(manifest),
                        "candidate": manifest["candidate_id"],
                    }
                ),
                flush=True,
            )
            result = engine.evaluate()
        else:
            engine = ScientificEngine(args.artifact, args.generator, create=False)
            if args.command == "freeze":
                manifest = engine.freeze()
                print(canonical({"phase": "FROZEN", "manifest_hash": digest(manifest)}))
                return
            if args.command == "evaluate":
                result = engine.evaluate()
            elif engine.journal.phase() == "COMPLETE":
                result = validate_evidence(
                    engine.journal.evidence(), engine.generator_path
                )
            else:
                with engine.journal.connect() as db:
                    row = db.execute(
                        "SELECT phase,uses,evidence FROM state WHERE id=1"
                    ).fetchone()
                print(
                    canonical(
                        {
                            "phase": row[0],
                            "holdout_uses": row[1],
                            "failure": json.loads(row[2]) if row[2] else None,
                        }
                    )
                )
                return
        print(json.dumps(result, indent=2, allow_nan=False))

    main()
