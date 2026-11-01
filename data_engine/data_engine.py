"""Data intelligence engine for the ML-algorithm research pipeline.

Turns a raw table into leakage-safe train/development/test partitions, an
auditable data card, and a generator TaskSpec whose research goals are derived
from measured evidence instead of being typed in by hand.

Leakage boundary
----------------
* Every learned transformation (column types, category vocabularies, imputation
  values, disguised-missing sentinels, redundancy and leakage checks) and every
  diagnostic is computed from training rows only. Development and test rows are
  only transformed.
* Labels outside the training partition are used for one purpose only:
  proportional stratification of the split, as in ordinary stratified sampling.
  The target's value set (to decide classification vs. regression and the class
  list) is read from all rows; feature-label associations never are.
* Duplicate detection and group handling for the split read feature values of
  all rows, never labels, so exact duplicates and declared entities never cross
  partitions.
* The test partition is sealed. Reading it needs a stated purpose and, when a
  holdout ledger (the recorder) is attached, a ledger entry that refuses reuse.

Dependencies: Python standard library and NumPy. scikit-learn is used only by
``load_builtin`` to read its bundled datasets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import sys
import time as _time
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np

VERSION = "data-engine-1.0"

MISSING_TOKENS = frozenset(
    {"", "na", "n/a", "nan", "null", "none", "?", "missing", "#n/a", "-nan"}
)
# Values commonly used to encode "unknown" inside numeric columns.
SENTINELS = (0.0, -1.0, -9.0, -99.0, -999.0, -9999.0, 99.0, 999.0, 9999.0)
# Research goals understood by generator_only/hypothesis_generator.py.
GOALS = (
    "outliers",
    "target_noise",
    "ill_conditioned",
    "class_imbalance",
    "high_dimension",
    "distribution_shift",
)
SEVERITIES = ("info", "warning", "block")


def _id_like_name(name):
    """id, ID, user_id, customerID, PassengerId, uuid ... (declared identifiers)."""
    lowered = name.strip()
    return bool(re.search(r"(^|[_\W])(id|uuid|guid|identifier)$", lowered, re.IGNORECASE)
                or re.search(r"[a-z](ID|Id)$", lowered))


@dataclass(frozen=True)
class Policy:
    """Every threshold the engine uses. Bound into each card by hash."""

    split: tuple = (0.6, 0.2, 0.2)
    max_classes_auto: int = 20  # integer targets with <= this many values are classes
    min_rows: int = 30
    min_class_rows: int = 5
    max_missing: float = 0.5  # above this a column keeps only its missingness indicator
    numeric_parse_rate: float = 0.98
    id_unique_ratio: float = 0.95
    id_span_ratio: float = 2.0
    max_onehot: int = 25
    min_level_rows: int = 5
    sentinel_min_rate: float = 0.01
    sentinel_max_rate: float = 0.5  # a value carrying most of a column is a real state
    sentinel_gap: float = 0.5  # isolation from the other values, in units of their IQR
    leak_accuracy: float = 0.99
    leak_r2: float = 0.99
    leak_rank: float = 0.995
    leak_linear_r2: float = 0.999  # regression target reconstructed linearly from features
    outlier_fence: float = 3.0
    outlier_row_rate: float = 0.02
    outlier_max_mode: float = 0.5  # columns dominated by one value have a degenerate IQR
    condition_limit: float = 1000.0
    collinear_corr: float = 0.98
    imbalance_ratio: float = 2.5
    high_dim_features: int = 50
    high_dim_ratio: float = 0.2
    noise_hard_rate: float = 0.08
    noise_unexplained: float = 0.4
    shift_auc: float = 0.65
    sorted_target_rho: float = 0.5
    probe_rows: int = 3000
    folds: int = 5
    rff_features: int = 256
    knn_k: int = 15
    max_redundancy_columns: int = 60
    max_bijection_levels: int = 50

    def digest(self) -> str:
        return _sha_text(canonical(asdict(self)))


# --------------------------------------------------------------------------
# Small utilities


def _default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def canonical(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_default
    )


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def array_hash(array) -> str:
    a = np.ascontiguousarray(array)
    head = canonical({"dtype": str(a.dtype), "shape": list(a.shape)})
    return hashlib.sha256(head.encode() + a.tobytes()).hexdigest()


def _clean(value):
    """Normalize one raw cell to a stripped string, or None when missing."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return repr(value) if math.isfinite(value) else None
    text = str(value).strip()
    return None if text.lower() in MISSING_TOKENS else text


_BAD = object()


def _parse_numbers(values):
    """Strings -> float array (NaN for missing); also counts unparseable cells."""
    cache = {}
    out = np.empty(len(values), dtype=np.float64)
    bad = 0
    for i, v in enumerate(values):
        if v is None:
            out[i] = np.nan
            continue
        r = cache.get(v)
        if r is None:
            try:
                f = float(v)
                r = f if math.isfinite(f) else _BAD
            except ValueError:
                r = _BAD
            cache[v] = r
        if r is _BAD:
            bad += 1
            out[i] = np.nan
        else:
            out[i] = r
    return out, bad


def _ranks(a):
    a = np.asarray(a, dtype=np.float64)
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    return (cum - (counts - 1) / 2.0)[inverse]


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / denom if denom > 0 else 0.0


def _spearman(a, b):
    return _pearson(_ranks(a), _ranks(b))


def _auc(scores, positive):
    positive = np.asarray(positive, dtype=bool)
    n1 = int(positive.sum())
    n0 = len(positive) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    r = _ranks(scores)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _balanced_accuracy(truth, pred):
    recalls = [float(np.mean(pred[truth == c] == c)) for c in np.unique(truth)]
    return float(np.mean(recalls))


def _r2(truth, pred):
    ss = float(np.sum((truth - truth.mean()) ** 2))
    return 1.0 - float(np.sum((truth - pred) ** 2)) / ss if ss > 0 else 0.0


def _skew(v):
    v = v - v.mean()
    s = math.sqrt(float(np.mean(v * v)))
    return float(np.mean(v**3)) / s**3 if s > 0 else 0.0


# --------------------------------------------------------------------------
# Tables and loaders


@dataclass
class Table:
    """A raw table of normalized string cells (None = missing)."""

    name: str
    columns: tuple
    rows: list
    source: str
    raw_sha256: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("table name required")
        cols = tuple(str(c).strip() for c in self.columns)
        if not cols or any(not c for c in cols):
            raise ValueError("every column needs a non-empty name")
        if len(set(cols)) != len(cols):
            dup = sorted({c for c in cols if cols.count(c) > 1})
            raise ValueError(f"duplicate column names: {dup}")
        self.columns = cols
        width = len(cols)
        for i, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(f"row {i + 1} has {len(row)} cells, expected {width}")

    def content_hash(self) -> str:
        """Order-sensitive hash of normalized content (independent of file bytes)."""
        h = hashlib.sha256(canonical(list(self.columns)).encode())
        for row in self.rows:
            h.update(canonical(list(row)).encode())
        return h.hexdigest()


def dropwhile_empty(header):
    """Header without its run of trailing empty names."""
    end = len(header)
    while end and not header[end - 1].strip():
        end -= 1
    return header[:end]


def read_csv(path, name=None, delimiter=","):
    """Read a CSV file with a header row into a Table (strict: no ragged rows)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    text = raw.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("empty CSV file") from None
    # A header ending in empty names (a trailing delimiter) may have data rows
    # without those trailing cells; pad them as missing. Any other ragged row is
    # an error.
    trailing_empty = len(header) - len(list(dropwhile_empty(header)))
    header = [h.strip() or f"unnamed_{j}" for j, h in enumerate(header)]
    width = len(header)
    rows = []
    for number, r in enumerate(reader, start=2):
        if not r:
            continue
        if width - trailing_empty <= len(r) < width:
            r = list(r) + [""] * (width - len(r))
        elif len(r) != width:
            raise ValueError(f"line {number} has {len(r)} cells, expected {width}")
        rows.append(tuple(_clean(c) for c in r))
    label = name or str(path).replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return Table(label, tuple(header), rows, f"csv:{path}", hashlib.sha256(raw).hexdigest())


def from_arrays(x, y=None, columns=None, target="target", name="arrays", source=None):
    """Build a Table from a feature matrix and optional target vector."""
    x = np.asarray(x, dtype=object)
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional")
    columns = list(columns) if columns is not None else [f"x{j}" for j in range(x.shape[1])]
    if len(columns) != x.shape[1]:
        raise ValueError("column name count mismatch")
    rows = [list(map(_clean, r)) for r in x]
    if y is not None:
        y = np.asarray(y, dtype=object)
        if y.ndim != 1 or len(y) != len(rows):
            raise ValueError("y must be one-dimensional and aligned with x")
        columns.append(target)
        for row, value in zip(rows, y):
            row.append(_clean(value))
    rows = [tuple(r) for r in rows]
    table = Table(name, tuple(columns), rows, source or "in-memory arrays", "")
    table.raw_sha256 = table.content_hash()
    return table


def load_builtin(name):
    """scikit-learn's bundled datasets (no network): iris, wine, breast_cancer, digits, diabetes."""
    from sklearn import __version__ as sk_version
    from sklearn import datasets

    loaders = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer,
        "digits": datasets.load_digits,
        "diabetes": datasets.load_diabetes,
    }
    if name not in loaders:
        raise ValueError(f"unknown builtin dataset {name!r}; choose from {sorted(loaders)}")
    bunch = loaders[name]()
    columns = [str(c) for c in getattr(bunch, "feature_names", [])] or None
    return from_arrays(
        bunch.data, bunch.target, columns, "target", name, f"sklearn {sk_version} bundled {name}"
    )


def cohort_overlap(tables, columns=None):
    """Pairwise shared-row check across tables (identical datasets, leaked rows).

    Rows are compared on the columns the two tables share (or ``columns``).
    """
    out = []
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            a, b = tables[i], tables[j]
            shared = [c for c in (columns or a.columns) if c in a.columns and c in b.columns]
            if not shared:
                out.append({"a": a.name, "b": b.name, "shared_columns": 0, "shared_rows": 0})
                continue
            ia = [a.columns.index(c) for c in shared]
            ib = [b.columns.index(c) for c in shared]
            ha = {_sha_text(canonical([r[k] for k in ia])) for r in a.rows}
            hb = {_sha_text(canonical([r[k] for k in ib])) for r in b.rows}
            inter = len(ha & hb)
            out.append(
                {
                    "a": a.name,
                    "b": b.name,
                    "shared_columns": len(shared),
                    "shared_rows": inter,
                    "jaccard": inter / max(1, len(ha | hb)),
                    "identical_content": a.content_hash() == b.content_hash(),
                }
            )
    return out


# --------------------------------------------------------------------------
# Results


class SealedTestError(RuntimeError):
    pass


@dataclass
class DataCard:
    """The engine's output: decisions, evidence, partitions and exports."""

    name: str
    verdict: str
    record: dict
    warnings: list
    _train: tuple | None = field(default=None, repr=False)
    _development: tuple | None = field(default=None, repr=False)
    _test: tuple | None = field(default=None, repr=False)
    _row_ids: dict | None = field(default=None, repr=False)
    _science: dict | None = field(default=None, repr=False)
    _unseal_log: list = field(default_factory=list, repr=False)

    # -- inspection ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "verdict": self.verdict,
            **self.record,
            "warnings": self.warnings,
        }

    def card_hash(self) -> str:
        """Deterministic hash of decisions and evidence (timings excluded)."""
        d = self.to_dict()
        d.pop("timings", None)
        return _sha_text(canonical(d))

    @property
    def blocking(self):
        return [w for w in self.warnings if w["severity"] == "block"]

    def summary(self) -> str:
        r = self.record
        lines = [f"{self.name}: {self.verdict}"]
        if "kind" in r:
            lines.append(
                f"  task={r['kind']} classes={r.get('classes')} rows used={r['rows_used']}"
                f"/{r['rows_input']} features={len(r.get('features', []))}"
            )
        if "split" in r:
            s = r["split"]["sizes"]
            lines.append(f"  split train/dev/test = {s['train']}/{s['development']}/{s['test']}")
        if "goals" in r:
            active = [g for g, v in r["goals"].items() if v["active"]]
            lines.append(f"  goals: {', '.join(active) or 'none'}")
        for w in self.warnings:
            if w["severity"] != "info":
                where = f" [{w['column']}]" if w.get("column") else ""
                lines.append(f"  {w['severity'].upper()} {w['code']}{where}: {w['message']}")
        return "\n".join(lines)

    # -- partitions ---------------------------------------------------------
    def _require_ready(self):
        if self.verdict == "BLOCKED":
            reasons = "; ".join(f"{w['code']}: {w['message']}" for w in self.blocking)
            raise ValueError(f"data card is BLOCKED ({reasons})")

    @property
    def train(self):
        self._require_ready()
        return self._train

    @property
    def development(self):
        self._require_ready()
        return self._development

    def unseal_test(self, purpose, ledger=None, study=None):
        """Release the sealed test partition once per stated purpose.

        With a ledger (recorder.Recorder or anything with ``holdout_access``),
        the ledger decides; it refuses a second use of the same test partition.
        """
        self._require_ready()
        if not isinstance(purpose, str) or len(purpose.strip()) < 3:
            raise SealedTestError("a stated purpose is required to unseal the test partition")
        fingerprint = self.record["split"]["fingerprints"]["test"]
        if ledger is not None:
            ledger.holdout_access(fingerprint, purpose=purpose, study=study or self.name)
        self._unseal_log.append({"purpose": purpose, "fingerprint": fingerprint})
        return self._test

    @property
    def unseal_count(self):
        return len(self._unseal_log)

    # -- exports ------------------------------------------------------------
    def task_spec(self, generator_module):
        """Generator TaskSpec with data-derived goals (training evidence only)."""
        self._require_ready()
        goals = tuple(g for g, v in self.record["goals"].items() if v["active"])
        kind = self.record["kind"]
        classes = len(self.record["classes"]) if kind == "classification" else 1
        return generator_module.TaskSpec(
            kind=kind, features=len(self.record["features"]), classes=classes, goals=goals
        )

    def generator_partitions(self, generator_module):
        """(train, development) as the generator's role-tagged Partition objects."""
        self._require_ready()
        p = generator_module.Partition
        return (
            p("train", self._train[0], self._train[1], self._row_ids["train"]),
            p("development", self._development[0], self._development[1], self._row_ids["development"]),
        )

    def science_dataset(self, science_module, name=None, historical_exposure=True, source=None):
        """All used rows as a scientific_validation.Dataset.

        scientific_validation applies its own split. Encoders here were fitted on
        this engine's training partition, so label-free encoder statistics
        (category vocabulary, medians) may include rows that land in the
        validator's test split. No target value enters any encoder. Duplicate
        buckets are passed as group IDs so duplicates stay together there too.
        """
        self._require_ready()
        s = self._science
        kind = self.record["kind"]
        classes = len(self.record["classes"]) if kind == "classification" else 1
        return science_module.Dataset(
            name or self.name,
            s["x"],
            s["y"],
            kind=kind,
            classes=classes,
            groups=s["groups"],
            historical_exposure=historical_exposure,
            source=source or self.record["source"],
        )


# --------------------------------------------------------------------------
# Engine


class DataEngine:
    def __init__(self, policy: Policy | None = None):
        self.policy = policy or Policy()

    def analyze(
        self,
        table: Table,
        target: str,
        *,
        kind=None,
        groups=None,
        time=None,
        drop=(),
        allow=(),
        seed=0,
    ) -> DataCard:
        """Analyze ``table`` predicting column ``target``.

        groups: column of real entity IDs (rows of one entity never cross partitions).
        time:   column giving temporal order; switches to a forward-in-time split.
        drop:   columns to exclude by declaration (e.g. known post-outcome fields).
        allow:  columns exempted from the leakage block after human review.
        """
        if not isinstance(table, Table):
            raise TypeError("Table required (use read_csv, from_arrays or load_builtin)")
        if kind not in (None, "classification", "regression"):
            raise ValueError("kind must be classification, regression or None")
        if type(seed) is not int:
            raise ValueError("integer seed required")
        return _Analysis(self.policy, table, target, kind, groups, time, drop, allow, seed).run()


class _Analysis:
    def __init__(self, policy, table, target, kind, groups, time, drop, allow, seed):
        self.p = policy
        self.t = table
        self.kind = kind
        self.seed = seed
        self.warnings = []
        self.timings = {}
        self.columns_plan = []
        names = table.columns
        for label, col in (("target", target), ("groups", groups), ("time", time)):
            if col is not None and col not in names:
                raise ValueError(f"{label} column {col!r} not in table")
        for col in tuple(drop) + tuple(allow):
            if col not in names:
                raise ValueError(f"column {col!r} not in table")
        if target in (groups, time) or target in drop:
            raise ValueError("target cannot also be a group, time or dropped column")
        self.target = target
        self.groups_col = groups
        self.time_col = time
        self.drop = tuple(drop)
        self.allow = frozenset(allow)
        self.feature_cols = [
            c for c in names if c not in (target, groups, time) and c not in self.drop
        ]

    # -- helpers ------------------------------------------------------------
    def warn(self, code, severity, message, column=None, **evidence):
        assert severity in SEVERITIES
        entry = {"code": code, "severity": severity, "message": message}
        if column is not None:
            entry["column"] = column
        if evidence:
            entry["evidence"] = evidence
        self.warnings.append(entry)

    def col(self, name, rows=None):
        k = self.t.columns.index(name)
        idx = self.used if rows is None else rows
        return [self.t.rows[i][k] for i in idx]

    def _tick(self, label, start):
        self.timings[label] = round(_time.perf_counter() - start, 6)
        return _time.perf_counter()

    # -- pipeline -----------------------------------------------------------
    def run(self):
        t0 = start = _time.perf_counter()
        self._target()
        start = self._tick("target", start)
        if not self._blocked():
            self._split()
            start = self._tick("split", start)
        if not self._blocked():
            self._fit_columns()
            self._redundancy()
            self._encode()
            start = self._tick("schema_and_encoding", start)
            self._leakage()
            start = self._tick("leakage", start)
            self._diagnostics()
            start = self._tick("diagnostics", start)
        self.timings["total"] = round(_time.perf_counter() - t0, 6)
        return self._card()

    def _blocked(self):
        return any(w["severity"] == "block" for w in self.warnings)

    # -- target -------------------------------------------------------------
    def _target(self):
        p = self.p
        k = self.t.columns.index(self.target)
        raw = [r[k] for r in self.t.rows]
        self.used = np.array([i for i, v in enumerate(raw) if v is not None], dtype=np.int64)
        dropped = len(raw) - len(self.used)
        if dropped:
            self.warn(
                "ROWS_DROPPED_MISSING_TARGET",
                "warning",
                f"{dropped} rows have no target value and were excluded",
                self.target,
                rows=dropped,
            )
        values = [raw[i] for i in self.used]
        numbers, bad = _parse_numbers(values)
        finite = numbers[np.isfinite(numbers)]
        integral = bool(len(finite)) and bool(np.all(finite == np.round(finite)))
        distinct = sorted(set(values))
        if self.kind is None:
            if bad or (integral and len(distinct) <= p.max_classes_auto):
                self.kind = "classification"
            else:
                self.kind = "regression"
            self.kind_source = "inferred from target value set"
        else:
            self.kind_source = "declared"
        if len(self.used) < p.min_rows:
            self.warn("TOO_FEW_ROWS", "block", f"{len(self.used)} usable rows; at least {p.min_rows} required")
        if self.kind == "classification":
            if bad == 0:
                distinct = sorted(distinct, key=float)
            self.classes = distinct
            code = {v: i for i, v in enumerate(distinct)}
            self.y_all = np.array([code[v] for v in values], dtype=np.int64)
            counts = np.bincount(self.y_all, minlength=len(distinct))
            if len(distinct) < 2:
                self.warn("TARGET_CONSTANT", "block", "the target has a single value", self.target)
            elif len(distinct) > 256:
                self.warn("TOO_MANY_CLASSES", "block", f"{len(distinct)} classes; at most 256 supported", self.target)
            rare = [distinct[i] for i, c in enumerate(counts) if c < p.min_class_rows]
            if rare and len(distinct) <= 256:
                self.warn(
                    "CLASS_TOO_RARE",
                    "block",
                    f"classes {rare[:10]} have fewer than {p.min_class_rows} rows",
                    self.target,
                    classes=rare[:50],
                )
        else:
            self.classes = None
            if bad:
                self.warn("TARGET_NOT_NUMERIC", "block", f"{bad} target values are not numbers", self.target)
                self.y_all = np.zeros(len(values))
            else:
                self.y_all = numbers
                if float(np.std(numbers)) == 0.0:
                    self.warn("TARGET_CONSTANT", "block", "the target has zero variance", self.target)
        if not self.feature_cols:
            self.warn("NO_FEATURES", "block", "no feature columns remain after declarations")

    # -- split --------------------------------------------------------------
    def _split(self):
        p = self.p
        n = len(self.used)
        # Duplicate keys from feature values only. Identifier-like columns (unique
        # per row and named like an ID, non-numeric, or a dense integer sequence)
        # are left out so a fresh ID cannot hide a re-recorded observation.
        key_cols = self._split_key_columns()
        kidx = [self.t.columns.index(c) for c in key_cols]
        if kidx:
            keys = [canonical([self.t.rows[i][k] for k in kidx]) for i in self.used]
        else:
            keys = [str(i) for i in range(n)]
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        first = {}
        for i, key in enumerate(keys):
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
        dup_rows = n - len(first)
        missing_groups = 0
        if self.groups_col is not None:
            seen = {}
            for i, g in enumerate(self.col(self.groups_col)):
                if g is None:
                    missing_groups += 1
                    continue
                if g in seen:
                    parent[find(i)] = find(seen[g])
                else:
                    seen[g] = i
            if missing_groups:
                self.warn(
                    "MISSING_GROUP_IDS",
                    "warning",
                    f"{missing_groups} rows have no group ID and were treated as independent",
                    self.groups_col,
                )
        buckets = {}
        for i in range(n):
            buckets.setdefault(find(i), []).append(i)
        buckets = list(buckets.values())
        self.bucket_of = np.empty(n, dtype=np.int64)
        for b, members in enumerate(buckets):
            self.bucket_of[members] = b
        if dup_rows:
            self.warn(
                "DUPLICATE_ROWS",
                "info",
                f"{dup_rows} rows duplicate another row's features; duplicates are kept in one partition",
                rows=dup_rows,
            )
        if len(buckets) < 3:
            self.warn("SPLIT_INFEASIBLE", "block", "fewer than three independent row groups")
            return
        assign = np.full(len(buckets), -1, dtype=np.int64)
        fractions = np.array(p.split, dtype=np.float64)
        if self.time_col is not None:
            self.split_method = "temporal"
            times = self._times()
            order = sorted(range(len(buckets)), key=lambda b: (min(times[i] for i in buckets[b]), b))
            filled = 0
            bounds = np.cumsum(fractions)[:2] * n
            for b in order:
                assign[b] = 0 if filled < bounds[0] else (1 if filled < bounds[1] else 2)
                filled += len(buckets[b])
        else:
            self.split_method = "stratified" if self.kind == "classification" else "target-quantile stratified"
            if self.kind == "classification":
                strata = [int(np.bincount(self.y_all[m]).argmax()) for m in buckets]
            else:
                edges = np.quantile(self.y_all, [0.2, 0.4, 0.6, 0.8])
                strata = [int(np.searchsorted(edges, float(np.mean(self.y_all[m])))) for m in buckets]
            by_stratum = {}
            for b, s in enumerate(strata):
                by_stratum.setdefault(s, []).append(b)
            for s in sorted(by_stratum):
                group = by_stratum[s]
                random.Random(f"{self.seed}:{s}").shuffle(group)
                group.sort(key=lambda b: -len(buckets[b]))  # stable: large buckets first
                total = sum(len(buckets[b]) for b in group)
                filled = np.zeros(3)
                for b in group:
                    target = fractions * total
                    choice = int(np.argmax(target - filled))
                    assign[b] = choice
                    filled[choice] += len(buckets[b])
        row_part = assign[self.bucket_of]
        if self.kind == "classification":
            for c in range(len(self.classes)):
                if not np.any((row_part == 0) & (self.y_all == c)):
                    if self.split_method == "temporal":
                        self.warn(
                            "CLASS_ABSENT_FROM_TRAIN",
                            "block",
                            f"class {self.classes[c]!r} does not occur in the temporal training window",
                            self.target,
                        )
                        continue
                    holders = [b for b in range(len(buckets)) if assign[b] != 0 and np.any(self.y_all[buckets[b]] == c)]
                    move = min(holders, key=lambda b: (len(buckets[b]), b))
                    assign[move] = 0
                    row_part = assign[self.bucket_of]
        for part, label in enumerate(("train", "development", "test")):
            if not np.any(row_part == part):
                self.warn("SPLIT_INFEASIBLE", "block", f"the {label} partition would be empty")
        self.part = row_part
        self.pos = {label: np.flatnonzero(row_part == k) for k, label in enumerate(("train", "development", "test"))}
        self.rows_of = {label: self.used[self.pos[label]] for label in self.pos}
        # Integrity: no row, duplicate key or group crosses partitions.
        for a in range(3):
            for b in range(a + 1, 3):
                ka = {keys[i] for i in np.flatnonzero(row_part == a)}
                kb = {keys[i] for i in np.flatnonzero(row_part == b)}
                if ka & kb:
                    raise AssertionError("duplicate feature rows crossed partitions")
        if self.groups_col is not None:
            gcol = self.col(self.groups_col)
            homes = {}
            for i, g in enumerate(gcol):
                if g is not None and homes.setdefault(g, row_part[i]) != row_part[i]:
                    raise AssertionError("a group crossed partitions")
        self.dup_rows = dup_rows
        self.n_buckets = len(buckets)

    def _split_key_columns(self):
        n = len(self.used)
        keep = []
        for c in self.feature_cols:
            values = self.col(c)
            if None not in values and len(set(values)) == n:
                nums, bad = _parse_numbers(values)
                dense = (
                    bad == 0
                    and bool(np.all(nums == np.round(nums)))
                    and (nums.max() - nums.min() + 1) / n <= self.p.id_span_ratio
                )
                if _id_like_name(c) or bad > 0 or dense:
                    continue
            keep.append(c)
        return keep

    def _times(self):
        values = self.col(self.time_col)
        numbers, bad = _parse_numbers(values)
        if bad == 0 and not np.any(np.isnan(numbers)):
            return numbers.tolist()
        out = []
        for v in values:
            if v is None:
                raise ValueError(f"time column {self.time_col!r} has missing values")
            try:
                out.append(datetime.fromisoformat(v).timestamp())
            except ValueError:
                raise ValueError(f"time value {v!r} is neither a number nor an ISO date") from None
        return out

    # -- column typing (training rows only) ---------------------------------
    def _fit_columns(self):
        p = self.p
        train_rows = self.rows_of["train"]
        n_train = len(train_rows)
        for name in self.feature_cols:
            values = self.col(name, train_rows)
            present = [v for v in values if v is not None]
            n_missing = n_train - len(present)
            missing_rate = n_missing / n_train
            distinct = set(present)
            plan = {"name": name, "missing_rate": round(missing_rate, 6), "distinct": len(distinct)}
            self.columns_plan.append(plan)
            if len(distinct) <= 1:
                plan.update(role="dropped", reason="CONSTANT")
                self.warn("CONSTANT_COLUMN", "info", "constant in training rows; dropped", name)
                continue
            numbers, bad = _parse_numbers(present)
            numeric = (len(present) - bad) / len(present) >= p.numeric_parse_rate
            unique_ratio = len(distinct) / len(present)
            plan["unique_ratio"] = round(unique_ratio, 6)
            if _id_like_name(name) and unique_ratio >= p.id_unique_ratio:
                plan.update(role="dropped", reason="IDENTIFIER")
                self.warn("IDENTIFIER_COLUMN", "warning", "named like an identifier and unique per row; dropped", name)
                continue
            if numeric:
                if bad:
                    self.warn(
                        "UNPARSEABLE_NUMERIC",
                        "warning",
                        f"{bad} training cells are not numbers and are treated as missing",
                        name,
                        cells=bad,
                        examples=sorted({v for v in present if _parse_numbers([v])[1]})[:5],
                    )
                finite = numbers[np.isfinite(numbers)]
                if self._is_numeric_id(finite, values, train_rows):
                    plan.update(role="dropped", reason="IDENTIFIER")
                    self.warn("IDENTIFIER_COLUMN", "warning", "row identifier (unique, dense integer sequence); dropped", name)
                    continue
            elif unique_ratio >= p.id_unique_ratio:
                tokens = float(np.mean([len(v.split()) for v in present]))
                reason = "FREE_TEXT" if tokens >= 2.0 else "IDENTIFIER"
                plan.update(role="dropped", reason=reason, mean_tokens=round(tokens, 3))
                self.warn(
                    f"{reason}_COLUMN",
                    "warning",
                    "free text (unique multi-word values); dropped" if reason == "FREE_TEXT"
                    else "identifier (unique per row); dropped",
                    name,
                )
                continue
            if missing_rate > p.max_missing:
                plan.update(role="indicator_only")
                self.warn(
                    "MOSTLY_MISSING",
                    "warning",
                    f"{missing_rate:.1%} missing in training rows; only a missingness indicator is kept",
                    name,
                    missing_rate=round(missing_rate, 6),
                )
                continue
            if numeric:
                self._plan_numeric(plan, name, values)
            else:
                self._plan_categorical(plan, name, present, n_missing, n_train)

    def _is_numeric_id(self, finite, values, train_rows):
        p = self.p
        if len(finite) < 10 or not np.all(finite == np.round(finite)):
            return False
        distinct = np.unique(finite)
        if len(distinct) / len(finite) < 0.99:
            return False
        span = (distinct[-1] - distinct[0] + 1) / len(distinct)
        if span <= p.id_span_ratio:
            return True
        positions = np.array([r for r, v in zip(train_rows, values) if v is not None], dtype=np.float64)
        nums, _ = _parse_numbers([v for v in values if v is not None])
        ok = np.isfinite(nums)
        return abs(_spearman(nums[ok], positions[ok])) >= 0.99

    def _plan_numeric(self, plan, name, values):
        p = self.p
        numbers, _ = _parse_numbers(values)
        finite = numbers[np.isfinite(numbers)]
        sentinels = []
        for s in SENTINELS:
            mask = finite == s
            rate = float(mask.mean()) if len(finite) else 0.0
            if not p.sentinel_min_rate <= rate <= p.sentinel_max_rate:
                continue
            other = finite[~mask]
            if len(other) < 20 or len(np.unique(other)) < 10:
                continue
            q1, q3 = np.percentile(other, [25, 75])
            iqr = q3 - q1
            if iqr <= 0:
                continue
            # An isolated point mass: outside the other values' range, separated by a gap.
            gap = other.min() - s if s < other.min() else (s - other.max() if s > other.max() else 0.0)
            if gap >= p.sentinel_gap * iqr:
                sentinels.append(s)
                self.warn(
                    "DISGUISED_MISSING",
                    "warning",
                    f"value {s:g} fills {rate:.1%} of training rows but is isolated from the other "
                    f"values [{other.min():g}, {other.max():g}] (gap {gap / iqr:.2f} IQR); treated as missing",
                    name,
                    value=s,
                    rate=round(rate, 6),
                    gap_iqr=round(float(gap / iqr), 6),
                    other_range=[float(other.min()), float(other.max())],
                )
        if sentinels:
            numbers[np.isin(numbers, sentinels)] = np.nan
            finite = numbers[np.isfinite(numbers)]
        missing = int(np.sum(~np.isfinite(numbers)))
        median = float(np.median(finite))
        plan.update(
            role="numeric",
            median=median,
            sentinels=sentinels,
            indicator=missing > 0,
            distinct_finite=int(len(np.unique(finite))),
        )
        if missing:
            self.warn(
                "MISSING_VALUES",
                "info",
                f"{missing} training values missing; median-imputed with a missingness indicator",
                name,
                missing=missing,
            )

    def _plan_categorical(self, plan, name, present, n_missing, n_train):
        p = self.p
        counts = {}
        for v in present:
            counts[v] = counts.get(v, 0) + 1
        if len(counts) > p.max_onehot:
            plan.update(
                role="frequency",
                freq={k: c / n_train for k, c in sorted(counts.items())},
                indicator=n_missing > 0,
            )
            self.warn(
                "HIGH_CARDINALITY",
                "info",
                f"{len(counts)} categories; encoded as training frequency",
                name,
                categories=len(counts),
            )
            if n_missing:
                self.warn("MISSING_VALUES", "info", f"{n_missing} training values missing; frequency 0 with a missingness indicator", name, missing=n_missing)
            return
        kept = sorted(k for k, c in counts.items() if c >= p.min_level_rows)
        rare = sorted(k for k, c in counts.items() if c < p.min_level_rows)
        if not kept:
            kept, rare = sorted(counts), []
        plan.update(
            role="categorical",
            levels=kept,
            other=bool(rare),
            missing_level=n_missing > 0,
            binary=len(kept) == 2 and not rare and n_missing == 0,
        )
        if rare:
            self.warn(
                "RARE_LEVELS",
                "info",
                f"{len(rare)} categories with < {p.min_level_rows} training rows pooled as __other__",
                name,
                pooled=len(rare),
            )
        if n_missing:
            self.warn("MISSING_VALUES", "info", f"{n_missing} training values missing; encoded as a __missing__ level", name, missing=n_missing)

    def _redundancy(self):
        """Exact one-to-one column pairs in training rows (same information twice)."""
        p = self.p
        train_rows = self.rows_of["train"]
        cands = [
            c
            for c in self.columns_plan
            if c["role"] in ("categorical", "numeric")
            and c["distinct"] <= p.max_bijection_levels
            and c["missing_rate"] == 0.0
        ][: p.max_redundancy_columns]
        values = {c["name"]: self.col(c["name"], train_rows) for c in cands}
        dropped = set()
        for i, a in enumerate(cands):
            for b in cands[i + 1 :]:
                if a["name"] in dropped or b["name"] in dropped or a["distinct"] != b["distinct"]:
                    continue
                pairs = set(zip(values[a["name"]], values[b["name"]]))
                if len(pairs) == a["distinct"]:
                    width = lambda c: 1 if c["role"] == "numeric" else len(c["levels"]) + int(c["other"]) + int(c["missing_level"])
                    loser, keeper = (a, b) if width(a) > width(b) else (b, a)
                    loser.update(role="dropped", reason="REDUNDANT", duplicate_of=keeper["name"])
                    dropped.add(loser["name"])
                    self.warn(
                        "REDUNDANT_COLUMNS",
                        "warning",
                        f"one-to-one with {keeper['name']!r} in training rows; dropped",
                        loser["name"],
                        partner=keeper["name"],
                    )

    # -- encoding -----------------------------------------------------------
    def _encode(self):
        features, sources, blocks = [], [], []
        columns = []
        unseen = {}
        for plan in self.columns_plan:
            role = plan["role"]
            if role == "dropped":
                continue
            name = plan["name"]
            values = self.col(name)
            in_fit = self.part == 0
            if role == "indicator_only":
                block = np.array([[0.0 if v is not None else 1.0] for v in values])
                names = [f"{name}__missing"]
            elif role == "numeric":
                numbers, bad = _parse_numbers(values)
                unparsed = ~np.isfinite(numbers) & np.array([v is not None for v in values])
                bad_out = int(np.sum(unparsed & ~in_fit))
                if plan["sentinels"]:
                    numbers[np.isin(numbers, plan["sentinels"])] = np.nan
                miss = ~np.isfinite(numbers)
                if bad_out:
                    unseen[name] = unseen.get(name, 0) + bad_out
                filled = np.where(miss, plan["median"], numbers)
                cols = [filled]
                names = [name]
                if plan["indicator"]:
                    cols.append(miss.astype(np.float64))
                    names.append(f"{name}__missing")
                block = np.column_stack(cols)
            elif role == "frequency":
                freq = plan["freq"]
                block = np.array([[freq.get(v, 0.0)] for v in values])
                names = [f"{name}__freq"]
                if plan["indicator"]:
                    block = np.column_stack([block, [0.0 if v is not None else 1.0 for v in values]])
                    names.append(f"{name}__missing")
                out = sum(1 for v, f in zip(values, in_fit) if not f and v is not None and v not in freq)
                if out:
                    unseen[name] = unseen.get(name, 0) + out
            else:  # categorical
                levels = plan["levels"]
                index = {v: j for j, v in enumerate(levels)}
                extra = []
                if plan["other"]:
                    extra.append("__other__")
                if plan["missing_level"]:
                    extra.append("__missing__")
                width = len(levels) + len(extra)
                block = np.zeros((len(values), width))
                out = 0
                for r, v in enumerate(values):
                    if v is None:
                        if plan["missing_level"]:
                            block[r, width - 1] = 1.0
                        elif plan["other"]:
                            block[r, len(levels)] = 1.0
                        continue
                    j = index.get(v)
                    if j is None:
                        if not in_fit[r]:
                            out += 1
                        if plan["other"]:
                            block[r, len(levels)] = 1.0
                    else:
                        block[r, j] = 1.0
                if out:
                    unseen[name] = unseen.get(name, 0) + out
                names = [f"{name}={v}" for v in levels + extra]
                if plan["binary"]:
                    block = block[:, 1:2]
                    names = names[1:2]
            start = len(features)
            features.extend(names)
            sources.extend([name] * len(names))
            blocks.append({"source": name, "role": role, "start": start, "width": len(names)})
            columns.append(block)
            plan["features"] = names
        if not features:
            self.warn("NO_FEATURES", "block", "every feature column was dropped")
            self.X = np.zeros((len(self.used), 0))
        else:
            self.X = np.column_stack(columns).astype(np.float64)
        self.features, self.sources, self.blocks = features, sources, blocks
        self.unseen = unseen
        if unseen:
            self.warn(
                "UNSEEN_AT_TRANSFORM",
                "info",
                "development/test cells with categories or number formats not seen in training "
                "were mapped to __other__, frequency 0 or the training median",
                cells=dict(sorted(unseen.items())),
            )

    # -- leakage (training rows only) ---------------------------------------
    def _folds(self, positions, labels):
        """Fold IDs at duplicate-bucket level, stratified for classification."""
        k = self.p.folds
        buckets = self.bucket_of[positions]
        uniq = np.unique(buckets)
        rng = random.Random(f"folds:{self.seed}")
        if self.kind == "classification":
            first_label = {}
            for b, lab in zip(buckets, labels):
                first_label.setdefault(int(b), int(lab))
            groups = {}
            for b in uniq:
                groups.setdefault(first_label[int(b)], []).append(int(b))
        else:
            groups = {0: [int(b) for b in uniq]}
        fold_of = {}
        offset = 0
        for g in sorted(groups):
            members = groups[g]
            rng.shuffle(members)
            for j, b in enumerate(members):
                fold_of[b] = (j + offset) % k
            offset += len(members)
        return np.array([fold_of[int(b)] for b in buckets], dtype=np.int64)

    def _leakage(self):
        p = self.p
        pos = self.pos["train"]
        y = self.y_all[pos]
        folds = self._folds(pos, y)
        rows = self.rows_of["train"]
        self.leakage = {}
        for plan in self.columns_plan:
            if plan["role"] == "dropped":
                continue
            name = plan["name"]
            values = self.col(name, rows)
            if plan["role"] == "indicator_only":
                values = ["1" if v is None else "0" for v in values]
            if plan["role"] == "numeric" and plan["distinct_finite"] > p.max_bijection_levels:
                x, _ = _parse_numbers(values)
                if plan["sentinels"]:
                    x[np.isin(x, plan["sentinels"])] = np.nan
                score = self._univariate_numeric(x, y, folds)
            else:
                score = self._univariate_categorical(values, y, folds)
            self.leakage[name] = score
            if self.kind == "classification":
                majority = float(np.max(np.bincount(y)) / len(y))
                leak = score["cv_accuracy"] >= p.leak_accuracy and score["cv_accuracy"] > majority + 0.05
            else:
                leak = score.get("cv_r2", 0.0) >= p.leak_r2 or abs(score.get("spearman", 0.0)) >= p.leak_rank
            if leak:
                if name in self.allow:
                    self.warn("LEAKAGE_ALLOWED", "warning", "near-perfect single-column predictor kept by explicit allowance", name, **score)
                else:
                    self.warn(
                        "LEAKAGE_SUSPECT",
                        "block",
                        "this single column predicts the target almost perfectly in training "
                        "cross-validation; check it is not derived from the outcome, then drop it "
                        "or pass it in allow=",
                        name,
                        **score,
                    )

    def _univariate_categorical(self, values, y, folds):
        pred = np.empty(len(y), dtype=np.float64)
        for f in range(self.p.folds):
            te = folds == f
            tr = ~te
            if not te.any() or not tr.any():
                continue
            table = {}
            for v, t in zip((values[i] for i in np.flatnonzero(tr)), y[tr]):
                table.setdefault(v, []).append(t)
            if self.kind == "classification":
                default = int(np.bincount(y[tr]).argmax())
                lookup = {v: int(np.bincount(np.array(ts, dtype=np.int64)).argmax()) for v, ts in table.items()}
            else:
                default = float(np.mean(y[tr]))
                lookup = {v: float(np.mean(ts)) for v, ts in table.items()}
            for i in np.flatnonzero(te):
                pred[i] = lookup.get(values[i], default)
        if self.kind == "classification":
            return {"cv_accuracy": round(float(np.mean(pred == y)), 6)}
        return {"cv_r2": round(_r2(y, pred), 6)}

    def _univariate_numeric(self, x, y, folds):
        if self.kind == "classification":
            return self._univariate_knn(x, y, folds)
        pred = np.empty(len(y), dtype=np.float64)
        for f in range(self.p.folds):
            te = folds == f
            tr = ~te
            if not te.any() or not tr.any():
                continue
            xt = x[tr]
            ok = np.isfinite(xt)
            edges = np.unique(np.quantile(xt[ok], np.linspace(0, 1, 21)[1:-1])) if ok.any() else np.array([])

            def bins(v):
                b = np.searchsorted(edges, v, side="right")
                return np.where(np.isfinite(v), b, -1)

            btr, bte = bins(xt), bins(x[te])
            if self.kind == "classification":
                default = int(np.bincount(y[tr]).argmax())
                lookup = {int(b): int(np.bincount(y[tr][btr == b]).argmax()) for b in np.unique(btr)}
            else:
                default = float(np.mean(y[tr]))
                lookup = {int(b): float(np.mean(y[tr][btr == b])) for b in np.unique(btr)}
            pred[te] = [lookup.get(int(b), default) for b in bte]
        if self.kind == "classification":
            return {"cv_accuracy": round(float(np.mean(pred == y)), 6)}
        ok = np.isfinite(x)
        rho = _spearman(x[ok], y[ok]) if ok.sum() > 2 else 0.0
        return {"cv_r2": round(_r2(y, pred), 6), "spearman": round(rho, 6)}

    def _univariate_knn(self, x, y, folds, k=5):
        """Out-of-fold 1-D k-nearest-neighbour accuracy of one numeric column.

        Adapts to wherever the column separates the classes (equal-frequency
        bins can straddle a class boundary). Missing values form their own group.
        """
        pred = np.empty(len(y), dtype=np.int64)
        for f in range(self.p.folds):
            te = np.flatnonzero(folds == f)
            tr = np.flatnonzero(folds != f)
            if not len(te) or not len(tr):
                continue
            default = int(np.bincount(y[tr]).argmax())
            ok_tr = tr[np.isfinite(x[tr])]
            miss_tr = tr[~np.isfinite(x[tr])]
            miss_label = int(np.bincount(y[miss_tr]).argmax()) if len(miss_tr) else default
            order = ok_tr[np.argsort(x[ok_tr], kind="mergesort")]
            xs, ys = x[order], y[order]
            kk = max(1, min(k, len(xs)))
            for i in te:
                v = x[i]
                if not np.isfinite(v) or not len(xs):
                    pred[i] = miss_label if not np.isfinite(v) else default
                    continue
                at = int(np.searchsorted(xs, v))
                lo, hi = max(0, at - kk), min(len(xs), at + kk)
                window = np.arange(lo, hi)
                near = window[np.argsort(np.abs(xs[window] - v), kind="mergesort")[:kk]]
                pred[i] = int(np.bincount(ys[near]).argmax())
        return {"cv_accuracy": round(float(np.mean(pred == y)), 6)}

    # -- diagnostics (training rows only) -----------------------------------
    def _diagnostics(self):
        p = self.p
        pos = self.pos["train"]
        X = self.X[pos]
        y = self.y_all[pos]
        n, d = X.shape
        diag = {"train_rows": int(n), "encoded_features": int(d)}
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        live = std > 0
        Z = (X[:, live] - mean[live]) / std[live]
        live_idx = np.flatnonzero(live)
        diag["zero_variance_features"] = int(d - live.sum())

        # Conditioning: drop one level per one-hot block (the dummy trap is not ill-conditioning).
        skip = set()
        for b in self.blocks:
            if b["role"] == "categorical" and b["width"] >= 2:
                skip.add(b["start"])
        keep = [j for j, f in enumerate(live_idx) if f not in skip]
        rng = np.random.default_rng(self.seed)
        sub = rng.choice(n, size=min(n, p.probe_rows), replace=False) if n > p.probe_rows else np.arange(n)
        dependencies = []
        if keep:
            _, s, vt = np.linalg.svd(Z[np.ix_(sub, keep)], full_matrices=False)
            tol = 1e-10 * s[0]
            kappa = float(s[0] / s[-1]) if s[-1] > tol else math.inf
            names = [self.features[live_idx[j]] for j in keep]
            for v in vt[s <= tol][:10]:
                big = np.flatnonzero(np.abs(v) >= 0.05 * np.abs(v).max())
                dependencies.append([names[j] for j in big[np.argsort(-np.abs(v[big]))]][:8])
        else:
            kappa = 1.0
        diag["condition_number"] = kappa if math.isfinite(kappa) else "inf"
        diag["exact_dependencies"] = dependencies
        if dependencies:
            self.warn(
                "EXACT_LINEAR_DEPENDENCE",
                "info",
                f"{len(dependencies)} exact linear dependencies among encoded training features "
                "(often one category implied by another column)",
                groups=dependencies,
            )
        numeric_cols = [b for b in self.blocks if b["role"] == "numeric"]
        pairs = []
        if len(numeric_cols) >= 2:
            idx = [b["start"] for b in numeric_cols]
            M = X[:, idx]
            sd = M.std(axis=0)
            ok = sd > 0
            C = np.corrcoef(M[:, ok], rowvar=False) if ok.sum() >= 2 else np.eye(1)
            names = [b["source"] for b, o in zip(numeric_cols, ok) if o]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    if abs(C[i, j]) >= p.collinear_corr:
                        pairs.append([names[i], names[j], round(float(C[i, j]), 6)])
        diag["collinear_pairs"] = pairs[:50]

        # Outliers and tails on continuous numeric columns.
        far_rows = np.zeros(n, dtype=bool)
        per_col = {}
        heavy = []
        dominated = []
        rows = self.rows_of["train"]
        for plan in self.columns_plan:
            if plan["role"] != "numeric" or plan["distinct_finite"] < 10:
                continue
            x, _ = _parse_numbers(self.col(plan["name"], rows))
            if plan["sentinels"]:
                x[np.isin(x, plan["sentinels"])] = np.nan
            ok = np.isfinite(x)
            q1, q3 = np.percentile(x[ok], [25, 75])
            iqr = q3 - q1
            sk = _skew(x[ok])
            if abs(sk) >= 2.0:
                heavy.append(plan["name"])
            _, mode_count = np.unique(x[ok], return_counts=True)
            if iqr <= 0 or mode_count.max() / ok.sum() >= p.outlier_max_mode:
                dominated.append(plan["name"])
                continue
            far = ok & ((x < q1 - p.outlier_fence * iqr) | (x > q3 + p.outlier_fence * iqr))
            if far.any():
                per_col[plan["name"]] = round(float(far.mean()), 6)
            far_rows |= far
        diag["outlier_row_rate"] = round(float(far_rows.mean()), 6)
        diag["outlier_columns"] = per_col
        diag["heavy_tailed_columns"] = heavy
        diag["outlier_rule_skipped_columns"] = dominated

        if self.kind == "classification":
            counts = np.bincount(y, minlength=len(self.classes))
            present = counts[counts > 0]
            diag["class_counts_train"] = {str(c): int(k) for c, k in zip(self.classes, counts)}
            diag["imbalance_ratio"] = round(float(present.max() / present.min()), 6)
            dup_conflict = 0
            buckets = self.bucket_of[pos]
            for b in np.unique(buckets):
                members = y[buckets == b]
                if len(members) > 1 and len(np.unique(members)) > 1:
                    dup_conflict += len(members)
            diag["conflicting_duplicate_rows"] = int(dup_conflict)
            if dup_conflict:
                self.warn(
                    "CONFLICTING_DUPLICATES",
                    "warning",
                    f"{dup_conflict} training rows share identical features but have different labels",
                    rows=int(dup_conflict),
                )

        positions = self.used[pos].astype(np.float64)
        rho = _spearman(positions, y) if n > 2 else 0.0
        diag["file_order_target_spearman"] = round(rho, 6)
        if abs(rho) >= p.sorted_target_rho:
            self.warn(
                "SORTED_BY_TARGET",
                "warning",
                f"file order correlates with the target (Spearman {rho:.2f}); sequential "
                "splits of this file would be invalid (this engine shuffles within strata)",
                spearman=round(rho, 6),
            )

        if Z.shape[1]:
            probes = self._probes(Z[sub], y[sub], pos[sub])
        else:
            probes = {}
        diag["probes"] = probes
        if self.kind == "regression" and probes.get("linear_r2", 0.0) >= p.leak_linear_r2:
            # The target is (almost) an exact linear function of the features: a
            # multivariate leak such as total = part_a + part_b.
            w = _ridge_predict(Z, y[:, None].astype(np.float64), np.eye(Z.shape[1]), alpha=1e-6)[:, 0]
            w = w - _ridge_predict(Z, y[:, None].astype(np.float64), np.zeros((1, Z.shape[1])), alpha=1e-6)[0, 0]
            order = np.argsort(-np.abs(w))
            top = [self.features[live_idx[j]] for j in order if abs(w[j]) >= 0.05 * abs(w[order[0]])][:6]
            sources = sorted({self.sources[self.features.index(f)] for f in top})
            if set(sources) <= self.allow:
                self.warn("LEAKAGE_ALLOWED", "warning", "target linearly reconstructable; columns explicitly allowed", columns=sources)
            else:
                self.warn(
                    "LEAKAGE_SUSPECT",
                    "block",
                    f"the target is reconstructed by a linear model with out-of-fold R^2 "
                    f"{probes['linear_r2']:.4f} in training rows (e.g. total = part_a + part_b); "
                    "check these columns are not computed from the outcome",
                    columns=sources,
                    cv_r2=probes["linear_r2"],
                    features=top,
                )

        if self.time_col is not None and Z.shape[1]:
            diag["shift"] = self._shift(Z, pos)
        self.diag = diag
        self._goals()

    def _probes(self, Z, y, positions):
        """Train-only cross-validated probes: linear ridge, random Fourier ridge, kNN.

        Descriptive diagnostics of the training partition; not benchmark results.
        """
        p = self.p
        folds = self._folds(positions, y)
        n, d = Z.shape
        rng = np.random.default_rng(self.seed + 1)
        m = min(n, 500)
        sample = Z[rng.choice(n, size=m, replace=False)]
        sq = (sample**2).sum(1)
        dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * sample @ sample.T, 0))
        sigma = float(np.median(dist[np.triu_indices(m, 1)])) or 1.0
        W = rng.normal(0.0, 1.0 / sigma, size=(d, p.rff_features))
        phase = rng.uniform(0, 2 * np.pi, size=p.rff_features)
        classes = np.unique(y) if self.kind == "classification" else None
        preds = {k: np.zeros(n) for k in ("linear", "fourier", "knn")}
        for f in range(p.folds):
            te = folds == f
            tr = ~te
            if te.sum() == 0 or tr.sum() < 5:
                continue
            Ztr, Zte = Z[tr], Z[te]
            if self.kind == "classification":
                T = (y[tr][:, None] == classes[None, :]).astype(np.float64)
            else:
                T = y[tr][:, None].astype(np.float64)
            for label, A_tr, A_te in (
                ("linear", Ztr, Zte),
                (
                    "fourier",
                    np.sqrt(2.0 / p.rff_features) * np.cos(Ztr @ W + phase),
                    np.sqrt(2.0 / p.rff_features) * np.cos(Zte @ W + phase),
                ),
            ):
                out = _ridge_predict(A_tr, T, A_te, alpha=1.0)
                preds[label][te] = classes[out.argmax(1)] if classes is not None else out[:, 0]
            preds["knn"][te] = _knn_predict(Ztr, y[tr], Zte, p.knn_k, classes)
        result = {}
        if self.kind == "classification":
            wrong = np.ones(n, dtype=bool)
            for k, v in preds.items():
                result[f"{k}_balanced_accuracy"] = round(_balanced_accuracy(y, v), 6)
                wrong &= v != y
            result["hard_rate"] = round(float(wrong.mean()), 6)
            best = max(result[f"{k}_balanced_accuracy"] for k in preds)
            result["nonlinear_gain"] = round(
                max(result["fourier_balanced_accuracy"], result["knn_balanced_accuracy"])
                - result["linear_balanced_accuracy"],
                6,
            )
            result["best"] = best
        else:
            for k, v in preds.items():
                result[f"{k}_r2"] = round(_r2(y, v), 6)
            best = max(result[f"{k}_r2"] for k in preds)
            result["unexplained"] = round(1.0 - max(best, 0.0), 6)
            result["nonlinear_gain"] = round(max(result["fourier_r2"], result["knn_r2"]) - result["linear_r2"], 6)
            result["best"] = best
        result["rows"] = int(n)
        return result

    def _shift(self, Z, pos):
        """Adversarial validation inside the training window: early half vs late half."""
        times = np.array(self._times_all(), dtype=np.float64)[pos]
        late = times > np.median(times)
        if late.all() or not late.any():
            return {"auc": 0.5, "note": "no temporal variation in training window"}
        folds = self._folds(pos, late.astype(np.int64)) if self.kind == "classification" else self._folds(pos, None)
        n = len(pos)
        score = np.zeros(n)
        for f in range(self.p.folds):
            te = folds == f
            tr = ~te
            if not te.any() or tr.sum() < 5:
                continue
            score[te] = _knn_predict(Z[tr], late[tr].astype(np.float64), Z[te], self.p.knn_k, None)
        auc = _auc(score, late)
        y = self.y_all[pos]
        result = {"auc": round(auc, 6)}
        if self.kind == "classification":
            early_mix = np.bincount(y[~late], minlength=len(self.classes)) / max(1, (~late).sum())
            late_mix = np.bincount(y[late], minlength=len(self.classes)) / max(1, late.sum())
            result["label_mix_tv_distance"] = round(float(0.5 * np.abs(early_mix - late_mix).sum()), 6)
        else:
            result["target_mean_change_sd"] = round(
                float((y[late].mean() - y[~late].mean()) / (y.std() or 1.0)), 6
            )
        return result

    def _times_all(self):
        if not hasattr(self, "_times_cache"):
            self._times_cache = self._times()
        return self._times_cache

    def _goals(self):
        p = self.p
        d = self.diag
        probes = d.get("probes", {})
        kappa = d["condition_number"]
        kappa_value = math.inf if kappa == "inf" else kappa
        goals = {}

        def goal(name, active, value, threshold, rule):
            goals[name] = {"active": bool(active), "value": value, "threshold": threshold, "rule": rule}

        goal("outliers", d["outlier_row_rate"] >= p.outlier_row_rate, d["outlier_row_rate"], p.outlier_row_rate,
             "share of training rows with a value beyond Tukey's 3-IQR far-out fence")
        if self.kind == "classification":
            goal("target_noise", probes.get("hard_rate", 0.0) >= p.noise_hard_rate, probes.get("hard_rate"),
                 p.noise_hard_rate,
                 "share of training rows misclassified out-of-fold by all three probes "
                 "(label errors and class overlap are not distinguishable)")
            goal("class_imbalance", d["imbalance_ratio"] >= p.imbalance_ratio, d["imbalance_ratio"],
                 p.imbalance_ratio, "largest / smallest training class count")
        else:
            goal("target_noise", probes.get("unexplained", 0.0) >= p.noise_unexplained, probes.get("unexplained"),
                 p.noise_unexplained, "1 - best out-of-fold probe R^2 in training rows")
            goal("class_imbalance", False, None, None, "not applicable to regression")
        goal("ill_conditioned", kappa_value >= p.condition_limit or bool(d["collinear_pairs"]),
             d["condition_number"], p.condition_limit,
             f"standardized training condition number (one level dropped per one-hot block) or a numeric "
             f"pair with |r| >= {p.collinear_corr}")
        ratio = d["encoded_features"] / max(1, d["train_rows"])
        goal("high_dimension", d["encoded_features"] >= p.high_dim_features or ratio >= p.high_dim_ratio,
             {"features": d["encoded_features"], "features_per_train_row": round(ratio, 6)},
             {"features": p.high_dim_features, "features_per_train_row": p.high_dim_ratio},
             "encoded feature count or features per training row")
        if "shift" in d:
            goal("distribution_shift", d["shift"]["auc"] >= p.shift_auc, d["shift"]["auc"], p.shift_auc,
                 "adversarial AUC separating early from late training rows (temporal split only)")
        else:
            goal("distribution_shift", False, None, p.shift_auc,
                 "not assessable without a declared time column")
        self.goals = goals

    # -- card ---------------------------------------------------------------
    def _card(self):
        p = self.p
        record = {
            "version": VERSION,
            "policy_sha256": p.digest(),
            "policy": asdict(p),
            "source": self.t.source,
            "raw_sha256": self.t.raw_sha256,
            "content_sha256": self.t.content_hash(),
            "target": self.target,
            "rows_input": len(self.t.rows),
            "rows_used": int(len(self.used)),
            "seed": self.seed,
            "declared": {
                "groups": self.groups_col,
                "time": self.time_col,
                "drop": list(self.drop),
                "allow": sorted(self.allow),
            },
            "timings": self.timings,
        }
        if self.kind is not None:
            record["kind"] = self.kind
            record["kind_source"] = getattr(self, "kind_source", "declared")
            record["classes"] = self.classes
        blocked = self._blocked()
        if hasattr(self, "part"):
            sizes = {k: int(len(v)) for k, v in self.pos.items()}
            fingerprints = {}
            for label, rows in self.rows_of.items():
                hashes = sorted(_sha_text(canonical(list(self.t.rows[i]))) for i in rows)
                fingerprints[label] = _sha_text(canonical(hashes))
            split = {
                "method": self.split_method,
                "fractions": list(p.split),
                "sizes": sizes,
                "duplicate_rows": int(self.dup_rows),
                "independent_buckets": int(self.n_buckets),
                "fingerprints": fingerprints,
            }
            if self.kind == "classification":
                split["class_counts"] = {
                    label: {str(c): int(k) for c, k in zip(self.classes, np.bincount(self.y_all[self.pos[label]], minlength=len(self.classes)))}
                    for label in self.pos
                }
            record["split"] = split
        if hasattr(self, "features"):
            record["columns"] = self.columns_plan
            record["features"] = self.features
            record["feature_sources"] = self.sources
        if hasattr(self, "leakage"):
            record["leakage_scores"] = self.leakage
        if hasattr(self, "diag"):
            record["diagnostics"] = self.diag
            record["goals"] = self.goals
        if blocked:
            verdict = "BLOCKED"
        elif any(w["severity"] == "warning" for w in self.warnings):
            verdict = "READY_WITH_WARNINGS"
        else:
            verdict = "READY"
        card = DataCard(self.t.name, verdict, record, self.warnings)
        if hasattr(self, "X") and not blocked:
            arrays = {}
            ids = {}
            for label in ("train", "development", "test"):
                pos = self.pos[label]
                X = np.ascontiguousarray(self.X[pos])
                y = np.ascontiguousarray(self.y_all[pos])
                X.setflags(write=False)
                y.setflags(write=False)
                arrays[label] = (X, y)
                ids[label] = tuple(str(int(i)) for i in self.used[pos])
            record["array_sha256"] = {
                label: {"x": array_hash(a[0]), "y": array_hash(a[1])} for label, a in arrays.items()
            }
            card._train = arrays["train"]
            card._development = arrays["development"]
            card._test = arrays["test"]
            card._row_ids = ids
            card._science = {
                "x": self.X,
                "y": self.y_all,
                "groups": [f"b{int(b)}" for b in self.bucket_of],
            }
        return card


def _ridge_predict(A_tr, T, A_te, alpha=1.0):
    mu = A_tr.mean(axis=0)
    t_mu = T.mean(axis=0)
    A = A_tr - mu
    n, d = A.shape
    if d <= 1500:
        w = np.linalg.solve(A.T @ A + alpha * np.eye(d), A.T @ (T - t_mu))
    else:
        w = A.T @ np.linalg.solve(A @ A.T + alpha * np.eye(n), T - t_mu)
    return (A_te - mu) @ w + t_mu


def _knn_predict(Z_tr, y_tr, Z_te, k, classes):
    k = max(1, min(k, len(Z_tr)))
    out = np.empty(len(Z_te))
    sq_tr = (Z_tr**2).sum(1)
    for start in range(0, len(Z_te), 512):
        Q = Z_te[start : start + 512]
        dist = (Q**2).sum(1)[:, None] + sq_tr[None, :] - 2 * Q @ Z_tr.T
        nn = np.argpartition(dist, k - 1, axis=1)[:, :k]
        votes = y_tr[nn]
        if classes is None:
            out[start : start + len(Q)] = votes.mean(axis=1)
        else:
            counts = (votes[:, :, None] == classes[None, None, :]).sum(1)
            out[start : start + len(Q)] = classes[counts.argmax(1)]
    return out


# --------------------------------------------------------------------------
# CLI


def main(argv=None):
    ap = argparse.ArgumentParser(description="Data intelligence engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyze a CSV file or builtin dataset")
    a.add_argument("source", help="CSV path, or builtin:<name>")
    a.add_argument("--target", required=False, default="target")
    a.add_argument("--kind", choices=["classification", "regression"])
    a.add_argument("--groups")
    a.add_argument("--time")
    a.add_argument("--drop", default="")
    a.add_argument("--allow", default="")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--json", help="write the full data card to this path")
    args = ap.parse_args(argv)
    if args.source.startswith("builtin:"):
        table = load_builtin(args.source.split(":", 1)[1])
    else:
        table = read_csv(args.source)
    card = DataEngine().analyze(
        table,
        args.target,
        kind=args.kind,
        groups=args.groups,
        time=args.time,
        drop=[c for c in args.drop.split(",") if c],
        allow=[c for c in args.allow.split(",") if c],
        seed=args.seed,
    )
    print(card.summary())
    print(f"  card sha256: {card.card_hash()}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(card.to_dict(), fh, indent=2, sort_keys=True, default=_default)
    return 0 if card.verdict != "BLOCKED" else 2


if __name__ == "__main__":
    sys.exit(main())
