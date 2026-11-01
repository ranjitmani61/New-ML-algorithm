"""Experiment recorder: an append-only, hash-chained evidence journal.

What it records
---------------
Runs (with environment, package versions, git state and source-file hashes),
metrics, content-addressed artifacts, candidate entities, per-stage decisions,
preregistrations and every access to a sealed test set.

What it guarantees
------------------
* Append-only: SQLite triggers reject UPDATE/DELETE on events and blobs.
* Tamper-evident: every event commits to the previous one (SHA-256 chain).
  ``verify()`` recomputes the chain, every blob digest and the semantic rules
  below. Publishing ``head()`` somewhere else (a commit message, a paper)
  additionally detects truncation or wholesale rewriting after that point.
* Holdout discipline: a test-set fingerprint can be accessed once. A second
  access is refused and the refusal itself is recorded; a declared reuse needs
  a written justification and taints every later result.
* No selection on the test set: leaderboards rank development metrics only;
  test metrics are flagged unless their entity was preregistered first.
* Concurrency: writers in separate processes serialize through BEGIN IMMEDIATE;
  sequence numbers stay gap-free.

What it does not guarantee
--------------------------
Hashes are not signatures. Whoever controls the file can rebuild a different,
internally consistent journal; only an externally published head detects that.
The recorder trusts the caller's description of what a metric or dataset is.

Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import math
import os
import platform
import socket
import sqlite3
import subprocess
import sys
import time
import traceback
import uuid
import zlib
from datetime import datetime, timezone

VERSION = "recorder-1.0"
SPLITS = ("train", "development", "test", "none")
MAX_EVENT_BYTES = 1_000_000
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(
    seq INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    run_id TEXT,
    kind TEXT NOT NULL,
    body TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS blobs(
    sha256 TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    codec TEXT NOT NULL,
    data BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS events_run_kind ON events(run_id, kind, seq);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
    BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
    BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS blobs_no_update BEFORE UPDATE ON blobs
    BEGIN SELECT RAISE(ABORT, 'blobs are append-only'); END;
CREATE TRIGGER IF NOT EXISTS blobs_no_delete BEFORE DELETE ON blobs
    BEGIN SELECT RAISE(ABORT, 'blobs are append-only'); END;
"""


class RecorderError(RuntimeError):
    pass


class HoldoutReuseError(RecorderError):
    """A sealed test set was requested a second time without declared reuse."""


class SelectionOnTestError(RecorderError):
    """Ranking or selecting candidates by test-set metrics."""


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _genesis(journal_id: str) -> str:
    return sha256_text(f"recorder-genesis:{journal_id}")


def _event_hash(seq, ts, run_id, kind, body_hash, prev_hash) -> str:
    return sha256_text(
        canonical({"seq": seq, "ts": ts, "run": run_id, "kind": kind, "body": body_hash, "prev": prev_hash})
    )


def _name(value, what, limit=200):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{what} must be a non-empty string of at most {limit} characters")
    return value


def _number(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{what} must be a finite number")
    return float(value)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment(sources=(), repo=None) -> dict:
    """Interpreter, platform, package versions, git state and source hashes."""
    from importlib import metadata

    packages = {}
    for dist in ("numpy", "scipy", "scikit-learn", "pandas", "torch"):
        try:
            packages[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    env = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "argv": [str(a) for a in sys.argv[:16]],
        "packages": packages,
        "sources": {str(p): file_sha256(p) for p in sources},
    }
    if repo is not None:
        try:
            run = lambda *args: subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10, check=True
            ).stdout.strip()
            env["git"] = {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
        except (OSError, subprocess.SubprocessError):
            env["git"] = {"commit": None, "dirty": None}
    return env


def _peak_rss_mb():
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 3)
    except (ImportError, AttributeError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Recorder:
    """Open (or create) a journal at ``path``."""

    def __init__(self, path, create=True, durable=False, sources=(), repo=None):
        self.path = str(path)
        exists = os.path.exists(self.path)
        if not exists and not create:
            raise RecorderError(f"no journal at {self.path}")
        self._db = sqlite3.connect(self.path, timeout=60.0, isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA busy_timeout=60000")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(f"PRAGMA synchronous={'FULL' if durable else 'NORMAL'}")
        self._depth = 0
        self.sources = tuple(sources)
        self.repo = repo
        tables = {r[0] for r in self._db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if exists and tables and "events" not in tables:
            raise RecorderError(f"{self.path} is an SQLite file but not a recorder journal")
        with self._tx():
            for statement in _SCHEMA.split(";\n"):
                if statement.strip():
                    self._db.execute(statement)
            row = self._db.execute("SELECT value FROM meta WHERE key='journal_id'").fetchone()
            if row is None:
                self.journal_id = uuid.uuid4().hex
                self._db.execute("INSERT INTO meta VALUES('journal_id', ?)", (self.journal_id,))
                self._db.execute("INSERT INTO meta VALUES('version', ?)", (VERSION,))
                self._append("journal.created", {"journal_id": self.journal_id, "version": VERSION}, None)
            else:
                self.journal_id = row[0]

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- transactions ---------------------------------------------------------
    @contextlib.contextmanager
    def _tx(self):
        if self._depth == 0:
            self._db.execute("BEGIN IMMEDIATE")
        self._depth += 1
        try:
            yield
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self._db.execute("ROLLBACK")
            raise
        else:
            self._depth -= 1
            if self._depth == 0:
                self._db.execute("COMMIT")

    @contextlib.contextmanager
    def batch(self):
        """Group many appends into one transaction (much faster)."""
        with self._tx():
            yield self

    def _append(self, kind, body, run_id):
        text = canonical(body)
        if len(text.encode("utf-8")) > MAX_EVENT_BYTES:
            raise ValueError("event body too large; store large payloads as artifacts")
        last = self._db.execute("SELECT seq, hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        seq = 1 if last is None else last[0] + 1
        prev = _genesis(self.journal_id) if last is None else last[1]
        ts = _now()
        body_hash = sha256_text(text)
        digest = _event_hash(seq, ts, run_id, kind, body_hash, prev)
        self._db.execute(
            "INSERT INTO events(seq, ts, run_id, kind, body, body_hash, prev_hash, hash) VALUES(?,?,?,?,?,?,?,?)",
            (seq, ts, run_id, kind, text, body_hash, prev, digest),
        )
        return seq

    def append(self, kind, body, run_id=None):
        """Low-level append of a custom event kind (namespaced, e.g. 'user.x')."""
        _name(kind, "event kind", 80)
        if "." not in kind or kind.split(".", 1)[0] in ("journal", "run", "holdout"):
            raise ValueError("custom event kinds need a namespace other than journal/run/holdout")
        with self._tx():
            return self._append(kind, body, run_id)

    # -- blobs ----------------------------------------------------------------
    def put_blob(self, data) -> str:
        if isinstance(data, str):
            data = data.encode("utf-8")
        if not isinstance(data, (bytes, bytearray)):
            data = canonical(data).encode("utf-8")
        data = bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        packed = zlib.compress(data, 6)
        codec, stored = ("zlib", packed) if len(packed) < len(data) else ("raw", data)
        with self._tx():
            self._db.execute(
                "INSERT OR IGNORE INTO blobs(sha256, size, codec, data) VALUES(?,?,?,?)",
                (digest, len(data), codec, sqlite3.Binary(stored)),
            )
        return digest

    def get_blob(self, digest) -> bytes:
        row = self._db.execute("SELECT codec, data FROM blobs WHERE sha256=?", (digest,)).fetchone()
        if row is None:
            raise KeyError(digest)
        try:
            data = zlib.decompress(row[1]) if row[0] == "zlib" else bytes(row[1])
        except zlib.error:
            raise RecorderError(f"blob {digest} is corrupted") from None
        if hashlib.sha256(data).hexdigest() != digest:
            raise RecorderError(f"blob {digest} is corrupted")
        return data

    # -- runs -----------------------------------------------------------------
    @contextlib.contextmanager
    def run(self, name, stage, params=None, parent=None, inputs=()):
        """Record a unit of work. Exceptions mark the run FAILED and propagate."""
        _name(name, "run name")
        _name(stage, "stage", 80)
        run_id = uuid.uuid4().hex
        body = {
            "run_id": run_id,
            "name": name,
            "stage": stage,
            "params": params or {},
            "parent": parent,
            "inputs": list(inputs),
            "env": environment(self.sources, self.repo),
        }
        if parent is not None and self._run_started(parent) is None:
            raise RecorderError(f"unknown parent run {parent}")
        with self._tx():
            self._append("run.started", body, run_id)
        handle = Run(self, run_id, name, stage)
        start = time.perf_counter()
        try:
            yield handle
        except BaseException as exc:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self._finish(run_id, "FAILED", start, {
                "type": type(exc).__name__,
                "message": str(exc)[:2000],
                "traceback_sha256": sha256_text(tb),
                "traceback_blob": self.put_blob(tb),
            })
            raise
        else:
            self._finish(run_id, "SUCCEEDED", start, None)

    def _finish(self, run_id, status, start, error):
        body = {"status": status, "seconds": round(time.perf_counter() - start, 6), "peak_rss_mb": _peak_rss_mb()}
        if error:
            body["error"] = error
        with self._tx():
            if self._run_state(run_id) != "RUNNING":
                raise RecorderError(f"run {run_id} is not running")
            self._append("run.finished", body, run_id)

    def _run_started(self, run_id):
        row = self._db.execute(
            "SELECT body FROM events WHERE run_id=? AND kind='run.started'", (run_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def _run_state(self, run_id):
        started = self._db.execute(
            "SELECT 1 FROM events WHERE run_id=? AND kind='run.started' LIMIT 1", (run_id,)
        ).fetchone()
        if started is None:
            return None
        ended = self._db.execute(
            "SELECT 1 FROM events WHERE run_id=? AND kind IN ('run.finished','run.abandoned') LIMIT 1", (run_id,)
        ).fetchone()
        return "ENDED" if ended else "RUNNING"

    def _run_event(self, run_id, kind, body):
        with self._tx():
            if self._run_state(run_id) != "RUNNING":
                raise RecorderError(f"run {run_id} is not running; events cannot be added")
            if kind == "metric" and body["split"] == "test":
                body["preregistered"] = self._preregistered(body.get("entity"))
                body["holdout_tainted"] = self._tainted()
            return self._append(kind, body, run_id)

    def recover(self, force=False):
        """Mark runs whose process died as ABANDONED (same host, dead pid)."""
        host = socket.gethostname()
        abandoned = []
        for run_id, body in self._open_runs():
            env = body["env"]
            same_host = env.get("host") == host
            if env.get("pid") == os.getpid() and same_host:
                continue
            if (same_host and not _pid_alive(int(env.get("pid", -1)))) or force:
                with self._tx():
                    if self._run_state(run_id) == "RUNNING":
                        self._append("run.abandoned", {"reason": "process not alive" if not force else "forced"}, run_id)
                        abandoned.append(run_id)
        return abandoned

    def _open_runs(self):
        started = {}
        for run_id, kind, body in self._db.execute(
            "SELECT run_id, kind, body FROM events WHERE kind IN ('run.started','run.finished','run.abandoned') ORDER BY seq"
        ):
            if kind == "run.started":
                started[run_id] = json.loads(body)
            else:
                started.pop(run_id, None)
        return list(started.items())

    # -- holdout discipline ---------------------------------------------------
    def holdout_access(self, dataset, purpose, study=None, allow_reuse=False, justification=None):
        """Register one access to a sealed test set identified by ``dataset``.

        A second access raises HoldoutReuseError and records the refusal, unless
        ``allow_reuse`` with a written ``justification``; that taints the journal.
        """
        _name(dataset, "dataset fingerprint", 200)
        _name(purpose, "purpose", 1000)
        refused = None
        with self._tx():
            prior = self._db.execute(
                "SELECT seq FROM events WHERE kind IN ('holdout.access','holdout.reuse_declared') "
                "AND json_extract(body, '$.dataset')=? ORDER BY seq LIMIT 1",
                (dataset,),
            ).fetchone()
            body = {"dataset": dataset, "purpose": purpose, "study": study}
            if prior is None:
                return self._append("holdout.access", body, None)
            if allow_reuse:
                if not isinstance(justification, str) or len(justification.strip()) < 20:
                    raise ValueError("declared holdout reuse needs a written justification (20+ characters)")
                return self._append("holdout.reuse_declared", {**body, "justification": justification, "first_access_seq": prior[0]}, None)
            refused = self._append("holdout.refused", {**body, "first_access_seq": prior[0]}, None)
        raise HoldoutReuseError(
            f"test set {dataset[:16]}... was already used (event {prior[0]}); refusal recorded as event {refused}"
        )

    def _tainted(self):
        return self._db.execute("SELECT 1 FROM events WHERE kind='holdout.reuse_declared' LIMIT 1").fetchone() is not None

    def preregister(self, entities, statement, metric=None, split="development", maximize=True):
        """Freeze which candidates go to the test set, before any test access."""
        if not isinstance(entities, (list, tuple)) or not entities:
            raise ValueError("preregistration needs at least one entity")
        for e in entities:
            _name(e, "entity id")
        _name(statement, "statement", 4000)
        body = {"entities": sorted(set(entities)), "statement": statement}
        if metric is not None:
            board = self.leaderboard(metric, split=split, maximize=maximize, k=50)
            body["leaderboard_metric"] = metric
            body["leaderboard_sha256"] = sha256_text(canonical(board))
            body["leaderboard_top"] = board[:10]
        with self._tx():
            return self._append("preregistration", body, None)

    def _preregistered(self, entity):
        if entity is None:
            return False
        for (body,) in self._db.execute("SELECT body FROM events WHERE kind='preregistration'"):
            if entity in json.loads(body)["entities"]:
                return True
        return False

    # -- queries --------------------------------------------------------------
    def events(self, kind=None, run_id=None):
        query, args = "SELECT seq, ts, run_id, kind, body, hash FROM events", []
        where = []
        if kind is not None:
            where.append("kind=?")
            args.append(kind)
        if run_id is not None:
            where.append("run_id=?")
            args.append(run_id)
        if where:
            query += " WHERE " + " AND ".join(where)
        for seq, ts, rid, k, body, digest in self._db.execute(query + " ORDER BY seq", args):
            yield {"seq": seq, "ts": ts, "run_id": rid, "kind": k, "body": json.loads(body), "hash": digest}

    def head(self) -> dict:
        row = self._db.execute("SELECT seq, hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        return {"journal_id": self.journal_id, "seq": row[0], "hash": row[1]}

    def runs(self, stage=None, status=None):
        out = {}
        for e in self.events():
            rid = e["run_id"]
            if e["kind"] == "run.started":
                b = e["body"]
                out[rid] = {"run_id": rid, "name": b["name"], "stage": b["stage"], "params": b["params"],
                            "parent": b["parent"], "status": "RUNNING", "started_seq": e["seq"]}
            elif e["kind"] == "run.finished" and rid in out:
                out[rid].update(status=e["body"]["status"], seconds=e["body"]["seconds"])
            elif e["kind"] == "run.abandoned" and rid in out:
                out[rid]["status"] = "ABANDONED"
        rows = list(out.values())
        if stage is not None:
            rows = [r for r in rows if r["stage"] == stage]
        if status is not None:
            rows = [r for r in rows if r["status"] == status]
        return rows

    def metrics(self, name=None, split=None, entity=None):
        rows = []
        for e in self.events(kind="metric"):
            b = e["body"]
            if (name is None or b["name"] == name) and (split is None or b["split"] == split) and (
                entity is None or b.get("entity") == entity
            ):
                rows.append({"seq": e["seq"], "run_id": e["run_id"], **b})
        return rows

    def leaderboard(self, metric, split="development", maximize=True, k=10, stage=None):
        """Rank entities by the mean of a development (or train) metric.

        Ranking by test metrics is refused: that is selection on the holdout.
        Ties go to the entity first recorded, so a later duplicate cannot win.
        """
        if split == "test":
            raise SelectionOnTestError("ranking candidates by test metrics is selection on the holdout")
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        stage_of = {r["run_id"]: r["stage"] for r in self.runs()} if stage else None
        agg = {}
        for m in self.metrics(metric, split):
            entity = m.get("entity")
            if entity is None or (stage and stage_of.get(m["run_id"]) != stage):
                continue
            a = agg.setdefault(entity, {"entity": entity, "values": [], "first_seq": m["seq"]})
            a["values"].append(m["value"])
        rows = []
        for a in agg.values():
            v = a["values"]
            mean = sum(v) / len(v)
            sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
            rows.append({"entity": a["entity"], "mean": mean, "sd": sd, "n": len(v), "first_seq": a["first_seq"]})
        rows.sort(key=lambda r: ((-r["mean"]) if maximize else r["mean"], r["first_seq"]))
        for i, r in enumerate(rows, 1):
            r["rank"] = i
        return rows[:k]

    def lineage(self, entity):
        """Every event that mentions an entity, in order, with its run's stage."""
        stage = {r["run_id"]: (r["name"], r["stage"]) for r in self.runs()}
        out = []
        for e in self.events():
            b = e["body"]
            hit = b.get("entity_id") == entity or b.get("entity") == entity or (
                e["kind"] == "preregistration" and entity in b.get("entities", [])
            )
            if hit:
                name, st = stage.get(e["run_id"], (None, None))
                out.append({"seq": e["seq"], "kind": e["kind"], "run": name, "stage": st, "body": b})
        return out

    def diff(self, run_a, run_b):
        a, b = self._run_started(run_a), self._run_started(run_b)
        if a is None or b is None:
            raise RecorderError("unknown run")

        def flat(prefix, value, out):
            if isinstance(value, dict):
                for k, v in value.items():
                    flat(f"{prefix}.{k}" if prefix else k, v, out)
            else:
                out[prefix] = value
            return out

        fa = flat("", {"params": a["params"], "packages": a["env"]["packages"], "sources": a["env"]["sources"],
                       "git": a["env"].get("git")}, {})
        fb = flat("", {"params": b["params"], "packages": b["env"]["packages"], "sources": b["env"]["sources"],
                       "git": b["env"].get("git")}, {})
        changed = {k: [fa.get(k), fb.get(k)] for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)}
        ma = {m["name"]: m["value"] for m in self.metrics() if m["run_id"] == run_a}
        mb = {m["name"]: m["value"] for m in self.metrics() if m["run_id"] == run_b}
        return {"changed": changed, "metrics": {k: [ma.get(k), mb.get(k)] for k in sorted(set(ma) | set(mb))}}

    def report(self):
        runs = self.runs()
        statuses = {}
        for r in runs:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        test = self.metrics(split="test")
        return {
            "head": self.head(),
            "events": self.head()["seq"],
            "runs": statuses,
            "holdout_accesses": sum(1 for _ in self.events(kind="holdout.access")),
            "holdout_refusals": sum(1 for _ in self.events(kind="holdout.refused")),
            "holdout_declared_reuse": sum(1 for _ in self.events(kind="holdout.reuse_declared")),
            "test_metrics": len(test),
            "test_metrics_not_preregistered": sum(1 for m in test if not m.get("preregistered")),
            "test_metrics_after_declared_reuse": sum(1 for m in test if m.get("holdout_tainted")),
            "blobs": self._db.execute("SELECT COUNT(*), COALESCE(SUM(size),0) FROM blobs").fetchone(),
        }

    # -- verification -----------------------------------------------------------
    def verify(self, expected_head=None):
        """Recompute the chain, blob digests and recorder rules. Returns a report."""
        rows = self._db.execute(
            "SELECT seq, ts, run_id, kind, body, body_hash, prev_hash, hash FROM events ORDER BY seq"
        ).fetchall()
        blobs = self._db.execute("SELECT sha256, size, codec, data FROM blobs").fetchall()
        meta = dict(self._db.execute("SELECT key, value FROM meta").fetchall())
        return _verify_rows(meta.get("journal_id"), rows, blobs, expected_head)

    # -- export -----------------------------------------------------------------
    def export_bundle(self, path):
        """Write the whole journal as portable JSON lines (verifiable without SQLite)."""
        head = self.head()
        with open(path, "w", encoding="utf-8") as fh:
            blob_count = self._db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
            fh.write(canonical({"bundle": VERSION, "journal_id": self.journal_id, "head": head,
                                "blobs": blob_count}) + "\n")
            for row in self._db.execute(
                "SELECT seq, ts, run_id, kind, body, body_hash, prev_hash, hash FROM events ORDER BY seq"
            ):
                fh.write(canonical(dict(zip(("seq", "ts", "run_id", "kind", "body", "body_hash", "prev_hash", "hash"), row))) + "\n")
            for digest, size, codec, data in self._db.execute("SELECT sha256, size, codec, data FROM blobs ORDER BY sha256"):
                fh.write(canonical({"blob": digest, "size": size, "codec": codec,
                                    "data": base64.b64encode(bytes(data)).decode("ascii")}) + "\n")
        return head


class Run:
    """Handle for an active run; created by Recorder.run()."""

    def __init__(self, recorder, run_id, name, stage):
        self.recorder = recorder
        self.run_id = run_id
        self.name = name
        self.stage = stage

    def metric(self, name, value, split="development", step=None, entity=None, dataset=None):
        _name(name, "metric name")
        body = {"name": name, "value": _number(value, "metric value"), "split": split}
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if step is not None:
            if isinstance(step, bool) or not isinstance(step, int):
                raise ValueError("step must be an integer")
            body["step"] = step
        if entity is not None:
            body["entity"] = _name(entity, "entity id")
        if dataset is not None:
            body["dataset"] = _name(dataset, "dataset")
        return self.recorder._run_event(self.run_id, "metric", body)

    def artifact(self, data, name, role="output", media=None):
        _name(name, "artifact name")
        if role not in ("input", "output"):
            raise ValueError("role must be input or output")
        digest = self.recorder.put_blob(data)
        size = self.recorder._db.execute("SELECT size FROM blobs WHERE sha256=?", (digest,)).fetchone()[0]
        self.recorder._run_event(self.run_id, "artifact", {"sha256": digest, "name": name, "role": role,
                                                           "media": media, "size": size})
        return digest

    def entity(self, entity_id, type, attrs=None):
        _name(entity_id, "entity id")
        _name(type, "entity type", 80)
        return self.recorder._run_event(self.run_id, "entity", {"entity_id": entity_id, "type": type,
                                                                "attrs": attrs or {}})

    def decision(self, entity_id, stage, status, reasons=()):
        _name(entity_id, "entity id")
        _name(stage, "decision stage", 80)
        _name(status, "decision status", 80)
        return self.recorder._run_event(self.run_id, "decision", {"entity_id": entity_id, "stage": stage,
                                                                  "status": status, "reasons": list(reasons)})

    def note(self, text):
        _name(text, "note", 10_000)
        return self.recorder._run_event(self.run_id, "note", {"text": text})


# --------------------------------------------------------------------------
# Verification shared by SQLite journals and exported bundles


def _verify_rows(journal_id, rows, blobs, expected_head=None):
    problems = []

    def problem(seq, text):
        if len(problems) < 200:
            problems.append({"seq": seq, "problem": text})

    if not journal_id:
        problem(None, "journal id missing")
        journal_id = ""
    prev = _genesis(journal_id)
    runs = {}
    holdout = {}
    preregistered = set()
    tainted = False
    referenced_blobs = set()
    by_seq = {}
    for index, (seq, ts, run_id, kind, body, body_hash, prev_hash, digest) in enumerate(rows, start=1):
        by_seq[seq] = digest
        if seq != index:
            problem(seq, f"sequence gap or reorder: expected {index}")
        if sha256_text(body) != body_hash:
            problem(seq, "event body does not match its hash (body edited)")
        if prev_hash != prev:
            problem(seq, "chain broken: previous-hash link does not match")
        if _event_hash(seq, ts, run_id, kind, body_hash, prev_hash) != digest:
            problem(seq, "event hash does not match its contents")
        prev = digest
        try:
            b = json.loads(body)
        except ValueError:
            problem(seq, "event body is not JSON")
            continue
        if index == 1 and (kind != "journal.created" or b.get("journal_id") != journal_id):
            problem(seq, "first event is not this journal's genesis")
        if kind == "run.started":
            if run_id in runs:
                problem(seq, "run started twice")
            runs[run_id] = "RUNNING"
        elif kind in ("run.finished", "run.abandoned"):
            if runs.get(run_id) != "RUNNING":
                problem(seq, f"{kind} for a run that is not running")
            runs[run_id] = "ENDED"
        elif kind in ("metric", "artifact", "entity", "decision", "note"):
            if runs.get(run_id) != "RUNNING":
                problem(seq, f"{kind} outside a running run")
            if kind == "artifact":
                referenced_blobs.add(b.get("sha256"))
            if kind == "metric" and b.get("split") == "test":
                expected = b.get("entity") in preregistered
                if b.get("preregistered") != expected:
                    problem(seq, "test metric preregistration flag is wrong")
                if b.get("holdout_tainted") != tainted:
                    problem(seq, "test metric taint flag is wrong")
        elif kind == "holdout.access":
            if b.get("dataset") in holdout:
                problem(seq, "second holdout access without declared reuse")
            holdout.setdefault(b.get("dataset"), seq)
        elif kind == "holdout.reuse_declared":
            tainted = True
            holdout.setdefault(b.get("dataset"), seq)
        elif kind == "preregistration":
            preregistered.update(b.get("entities", []))
        if kind == "run.finished" and isinstance(b.get("error"), dict):
            referenced_blobs.add(b["error"].get("traceback_blob"))
    stored = set()
    for digest, size, codec, data in blobs:
        stored.add(digest)
        try:
            raw = zlib.decompress(bytes(data)) if codec == "zlib" else bytes(data)
        except zlib.error:
            problem(None, f"blob {digest[:12]} cannot be decompressed")
            continue
        if hashlib.sha256(raw).hexdigest() != digest or len(raw) != size:
            problem(None, f"blob {digest[:12]} does not match its digest")
    for missing in sorted(d for d in referenced_blobs - stored if d):
        problem(None, f"artifact blob {missing[:12]} is missing")
    if expected_head is not None:
        seq = expected_head.get("seq")
        if by_seq.get(seq) != expected_head.get("hash"):
            problem(seq, "journal does not contain the externally published head (truncated or rewritten)")
    head = {"seq": rows[-1][0], "hash": rows[-1][7]} if rows else None
    return {
        "ok": not problems,
        "events": len(rows),
        "blobs": len(stored),
        "head": head,
        "open_runs": sorted(r for r, s in runs.items() if s == "RUNNING"),
        "problems": problems,
    }


def verify_bundle(path, expected_head=None):
    """Verify an exported bundle on its own (no SQLite involved)."""
    with open(path, encoding="utf-8") as fh:
        header = json.loads(fh.readline())
        rows, blobs = [], []
        for line in fh:
            item = json.loads(line)
            if "blob" in item:
                blobs.append((item["blob"], item["size"], item["codec"], base64.b64decode(item["data"])))
            else:
                rows.append(tuple(item[k] for k in ("seq", "ts", "run_id", "kind", "body", "body_hash", "prev_hash", "hash")))
    result = _verify_rows(header.get("journal_id"), rows, blobs, expected_head)
    if header.get("head", {}).get("hash") != (result["head"] or {}).get("hash"):
        result["problems"].append({"seq": None, "problem": "bundle header head does not match its events"})
        result["ok"] = False
    return result


# --------------------------------------------------------------------------
# CLI


def main(argv=None):
    ap = argparse.ArgumentParser(description="Experiment recorder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("journal")
    v.add_argument("--head", help="externally published head as SEQ:HASH")
    sub.add_parser("report").add_argument("journal")
    lb = sub.add_parser("leaderboard")
    lb.add_argument("journal")
    lb.add_argument("metric")
    lb.add_argument("--minimize", action="store_true")
    ln = sub.add_parser("lineage")
    ln.add_argument("journal")
    ln.add_argument("entity")
    ex = sub.add_parser("export")
    ex.add_argument("journal")
    ex.add_argument("bundle")
    vb = sub.add_parser("verify-bundle")
    vb.add_argument("bundle")
    vb.add_argument("--head")
    args = ap.parse_args(argv)
    head = None
    if getattr(args, "head", None):
        seq, digest = args.head.split(":", 1)
        head = {"seq": int(seq), "hash": digest}
    if args.cmd == "verify-bundle":
        result = verify_bundle(args.bundle, head)
    else:
        rec = Recorder(args.journal, create=False)
        if args.cmd == "verify":
            result = rec.verify(head)
        elif args.cmd == "report":
            result = rec.report()
        elif args.cmd == "leaderboard":
            result = rec.leaderboard(args.metric, maximize=not args.minimize, k=20)
        elif args.cmd == "lineage":
            result = rec.lineage(args.entity)
        else:
            result = rec.export_bundle(args.bundle)
    print(json.dumps(result, indent=2, sort_keys=True))
    if isinstance(result, dict) and result.get("ok") is False:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
