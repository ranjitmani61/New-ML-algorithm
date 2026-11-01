"""Contract tests for the experiment recorder, including adversarial ones."""

import json
import multiprocessing
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recorder as rc  # noqa: E402


def tamper(path, sql, args=()):
    """Edit a journal the way an attacker with file access would: drop the guards first."""
    db = sqlite3.connect(path, isolation_level=None)
    for name in ("events_no_update", "events_no_delete", "blobs_no_update", "blobs_no_delete"):
        db.execute(f"DROP TRIGGER IF EXISTS {name}")
    db.execute(sql, args)
    db.close()


def _writer(path, worker, count):
    rec = rc.Recorder(path)
    with rec.run(f"worker-{worker}", "stress") as run:
        for i in range(count):
            run.metric("value", float(i), entity=f"w{worker}-{i % 5}")
    rec.close()


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "journal.sqlite")
        self.rec = rc.Recorder(self.path)

    def tearDown(self):
        self.rec.close()
        self.dir.cleanup()


class ChainContract(Base):
    def test_genesis_and_verify(self):
        report = self.rec.verify()
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["events"], 1)
        first = next(self.rec.events())
        self.assertEqual(first["kind"], "journal.created")
        self.assertEqual(first["body"]["journal_id"], self.rec.journal_id)

    def test_triggers_reject_update_and_delete(self):
        with self.assertRaises(sqlite3.DatabaseError):
            self.rec._db.execute("UPDATE events SET body='{}' WHERE seq=1")
        with self.assertRaises(sqlite3.DatabaseError):
            self.rec._db.execute("DELETE FROM events WHERE seq=1")

    def test_edit_detected(self):
        with self.rec.run("r", "s") as run:
            run.metric("acc", 0.9, entity="c1")
        self.rec.close()
        tamper(self.path, "UPDATE events SET body=replace(body, '0.9', '0.99') WHERE kind='metric'")
        self.rec = rc.Recorder(self.path, create=False)
        report = self.rec.verify()
        self.assertFalse(report["ok"])
        self.assertTrue(any("body edited" in p["problem"] for p in report["problems"]))

    def test_consistent_rehash_still_breaks_the_chain(self):
        with self.rec.run("r", "s") as run:
            run.metric("acc", 0.5, entity="c1")
            run.metric("acc", 0.6, entity="c2")
        self.rec.close()
        db = sqlite3.connect(self.path)
        seq, ts, run_id, kind, body, prev = db.execute(
            "SELECT seq, ts, run_id, kind, body, prev_hash FROM events WHERE kind='metric' ORDER BY seq LIMIT 1").fetchone()
        db.close()
        forged = body.replace("0.5", "0.95")
        bh = rc.sha256_text(forged)
        h = rc._event_hash(seq, ts, run_id, kind, bh, prev)
        tamper(self.path, "UPDATE events SET body=?, body_hash=?, hash=? WHERE seq=?", (forged, bh, h, seq))
        self.rec = rc.Recorder(self.path, create=False)
        report = self.rec.verify()
        self.assertFalse(report["ok"])
        self.assertTrue(any("chain broken" in p["problem"] for p in report["problems"]))

    def test_truncation_needs_published_head(self):
        with self.rec.run("r", "s") as run:
            run.metric("acc", 0.7, entity="c1")
        published = self.rec.head()
        self.rec.close()
        tamper(self.path, "DELETE FROM events WHERE seq >= ?", (published["seq"] - 1,))
        self.rec = rc.Recorder(self.path, create=False)
        self.assertTrue(self.rec.verify()["ok"], "a clean truncation is invisible without an anchor")
        anchored = self.rec.verify(expected_head=published)
        self.assertFalse(anchored["ok"])

    def test_foreign_and_missing_files(self):
        with self.assertRaises(rc.RecorderError):
            rc.Recorder(os.path.join(self.dir.name, "absent.sqlite"), create=False)
        other = os.path.join(self.dir.name, "other.sqlite")
        sqlite3.connect(other).execute("CREATE TABLE x(a)")
        with self.assertRaises(rc.RecorderError):
            rc.Recorder(other)

    def test_reopen_keeps_identity(self):
        jid = self.rec.journal_id
        self.rec.close()
        self.rec = rc.Recorder(self.path, create=False)
        self.assertEqual(self.rec.journal_id, jid)
        self.assertEqual(self.rec.head()["seq"], 1)


class RunContract(Base):
    def test_success_failure_and_rules(self):
        src = os.path.join(self.dir.name, "code.py")
        with open(src, "w") as fh:
            fh.write("print(1)\n")
        rec = rc.Recorder(self.path, sources=[src])
        with rec.run("train", "fit", params={"lr": 0.1}) as run:
            run.metric("loss", 0.25, split="train", step=1)
            run.note("ok")
        with self.assertRaises(ZeroDivisionError):
            with rec.run("broken", "fit") as run:
                1 / 0
        runs = {r["name"]: r for r in rec.runs()}
        self.assertEqual(runs["train"]["status"], "SUCCEEDED")
        self.assertEqual(runs["broken"]["status"], "FAILED")
        failed = [e for e in rec.events(kind="run.finished") if e["body"]["status"] == "FAILED"][0]
        self.assertEqual(failed["body"]["error"]["type"], "ZeroDivisionError")
        self.assertIn(b"ZeroDivisionError", rec.get_blob(failed["body"]["error"]["traceback_blob"]))
        started = next(rec.events(kind="run.started"))
        self.assertIn(src, started["body"]["env"]["sources"])
        with self.assertRaises(rc.RecorderError):
            run.metric("late", 1.0)  # run already ended
        rec.close()

    def test_input_validation(self):
        with self.rec.run("r", "s") as run:
            for bad in (float("nan"), float("inf"), True, "1"):
                with self.assertRaises(ValueError):
                    run.metric("m", bad)
            with self.assertRaises(ValueError):
                run.metric("m", 1.0, split="holdout")
            with self.assertRaises(ValueError):
                run.note("x" * 20_000)
            with self.assertRaises(ValueError):
                run.metric("", 1.0)
        with self.assertRaises(rc.RecorderError):
            with self.rec.run("child", "s", parent="no-such-run"):
                pass
        with self.assertRaises(ValueError):
            self.rec.append("run.fake", {})

    def test_blobs_dedup_and_corruption(self):
        with self.rec.run("r", "s") as run:
            a = run.artifact(b"x" * 10000, "big.bin")
            b = run.artifact(b"x" * 10000, "copy.bin")
            run.artifact({"k": [1, 2]}, "obj.json", media="application/json")
        self.assertEqual(a, b)
        self.assertEqual(self.rec.get_blob(a), b"x" * 10000)
        self.assertEqual(self.rec._db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0], 2)
        self.rec.close()
        tamper(self.path, "UPDATE blobs SET data=? WHERE sha256=?", (b"evil", a))
        self.rec = rc.Recorder(self.path, create=False)
        with self.assertRaises(rc.RecorderError):
            self.rec.get_blob(a)
        self.assertFalse(self.rec.verify()["ok"])

    def test_diff_shows_source_change(self):
        src = os.path.join(self.dir.name, "algo.py")
        with open(src, "w") as fh:
            fh.write("v = 1\n")
        rec = rc.Recorder(self.path, sources=[src])
        with rec.run("a", "s", params={"lr": 0.1}) as ra:
            ra.metric("acc", 0.8)
        with open(src, "w") as fh:
            fh.write("v = 2\n")
        rec.close()
        rec = rc.Recorder(self.path, sources=[src])
        with rec.run("b", "s", params={"lr": 0.2}) as rb:
            rb.metric("acc", 0.85)
        d = rec.diff(ra.run_id, rb.run_id)
        rec.close()
        self.assertIn("params.lr", d["changed"])
        self.assertTrue(any(k.startswith("sources.") for k in d["changed"]))
        self.assertEqual(d["metrics"]["acc"], [0.8, 0.85])


class HoldoutContract(Base):
    def test_single_use_refusal_and_declared_reuse(self):
        self.rec.holdout_access("fp-test", purpose="final evaluation")
        with self.assertRaises(rc.HoldoutReuseError):
            self.rec.holdout_access("fp-test", purpose="peek again")
        refused = list(self.rec.events(kind="holdout.refused"))
        self.assertEqual(len(refused), 1, "the refusal itself must be on record")
        with self.assertRaises(ValueError):
            self.rec.holdout_access("fp-test", purpose="again", allow_reuse=True, justification="short")
        self.rec.holdout_access("fp-test", purpose="again", allow_reuse=True,
                                justification="reviewer requested a recomputation with a corrected metric")
        with self.rec.run("eval", "science") as run:
            run.metric("acc", 0.9, split="test", entity="c1")
        report = self.rec.report()
        self.assertEqual(report["holdout_refusals"], 1)
        self.assertEqual(report["test_metrics_after_declared_reuse"], 1)
        self.assertTrue(self.rec.verify()["ok"])

    def test_forged_second_access_is_caught(self):
        self.rec.holdout_access("fp", purpose="first use")
        self.rec.close()
        # Append a second access that bypasses the API but keeps the chain valid.
        r = rc.Recorder(self.path, create=False)
        with r._tx():
            r._append("holdout.access", {"dataset": "fp", "purpose": "sneaky", "study": None}, None)
        report = r.verify()
        self.assertFalse(report["ok"])
        self.assertTrue(any("second holdout access" in p["problem"] for p in report["problems"]))
        r.close()
        self.rec = rc.Recorder(self.path, create=False)


class SelectionContract(Base):
    def test_leaderboard_ranks_development_and_refuses_test(self):
        with self.rec.run("round-1", "screen") as run:
            for entity, values in (("a", [0.8, 0.82]), ("b", [0.9]), ("c", [0.9])):
                for v in values:
                    run.metric("dev_acc", v, entity=entity)
            run.metric("dev_acc", 0.99, split="train", entity="d")
        board = self.rec.leaderboard("dev_acc")
        self.assertEqual([r["entity"] for r in board], ["b", "c", "a"], "ties go to the first recorded")
        self.assertEqual(board[2]["n"], 2)
        with self.assertRaises(rc.SelectionOnTestError):
            self.rec.leaderboard("dev_acc", split="test")
        self.assertEqual(self.rec.leaderboard("dev_acc", maximize=False)[0]["entity"], "a")

    def test_preregistration_flags_test_metrics(self):
        with self.rec.run("dev", "screen") as run:
            run.metric("dev_acc", 0.9, entity="winner")
            run.metric("dev_acc", 0.8, entity="other")
        seq = self.rec.preregister(["winner"], "best development candidate goes to test", metric="dev_acc")
        pre = [e for e in self.rec.events(kind="preregistration")][0]
        self.assertEqual(pre["seq"], seq)
        self.assertEqual(pre["body"]["leaderboard_top"][0]["entity"], "winner")
        with self.rec.run("final", "science") as run:
            run.metric("test_acc", 0.88, split="test", entity="winner")
            run.metric("test_acc", 0.91, split="test", entity="other")
        flags = {m["entity"]: m["preregistered"] for m in self.rec.metrics(split="test")}
        self.assertEqual(flags, {"winner": True, "other": False})
        self.assertEqual(self.rec.report()["test_metrics_not_preregistered"], 1)
        self.assertTrue(self.rec.verify()["ok"])

    def test_lineage(self):
        with self.rec.run("gen", "generator") as run:
            run.entity("H-1", "hypothesis", {"round": 1})
        with self.rec.run("filter", "hypothesis_filter") as run:
            run.decision("H-1", "hypothesis_filter", "READY", ["coherent"])
        with self.rec.run("score", "development") as run:
            run.metric("dev_acc", 0.7, entity="H-1")
        trail = self.rec.lineage("H-1")
        self.assertEqual([t["kind"] for t in trail], ["entity", "decision", "metric"])
        self.assertEqual([t["stage"] for t in trail], ["generator", "hypothesis_filter", "development"])


class ProcessContract(Base):
    def test_concurrent_writers_keep_a_gap_free_chain(self):
        procs = [multiprocessing.Process(target=_writer, args=(self.path, w, 60)) for w in range(4)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(120)
            self.assertEqual(p.exitcode, 0)
        report = self.rec.verify()
        self.assertTrue(report["ok"], report["problems"][:3])
        self.assertEqual(report["events"], 1 + 4 * (60 + 2))
        self.assertEqual(len(self.rec.runs(status="SUCCEEDED")), 4)

    def test_crashed_process_run_is_recovered(self):
        code = (
            "import os, sys; sys.path.insert(0, %r); import recorder as rc\n"
            "rec = rc.Recorder(%r)\n"
            "ctx = rec.run('doomed', 'fit'); run = ctx.__enter__(); run.metric('m', 1.0)\n"
            "os._exit(3)\n"
        ) % (HERE, self.path)
        result = subprocess.run([sys.executable, "-c", code], timeout=60)
        self.assertEqual(result.returncode, 3)
        self.assertEqual(self.rec.runs()[0]["status"], "RUNNING")
        self.assertEqual(self.rec.verify()["open_runs"], [self.rec.runs()[0]["run_id"]])
        abandoned = self.rec.recover()
        self.assertEqual(len(abandoned), 1)
        self.assertEqual(self.rec.runs()[0]["status"], "ABANDONED")
        self.assertTrue(self.rec.verify()["ok"])


class BundleContract(Base):
    def test_export_verify_and_tamper(self):
        with self.rec.run("r", "s") as run:
            run.metric("acc", 0.5, entity="x")
            run.artifact(b"payload", "p.bin")
        self.rec.holdout_access("fp", "final")
        bundle = os.path.join(self.dir.name, "bundle.jsonl")
        head = self.rec.export_bundle(bundle)
        ok = rc.verify_bundle(bundle, expected_head=head)
        self.assertTrue(ok["ok"], ok)
        lines = open(bundle).read().splitlines()
        lines[3] = lines[3].replace("0.5", "0.95")
        with open(bundle, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        self.assertFalse(rc.verify_bundle(bundle)["ok"])

    def test_cli(self):
        with self.rec.run("r", "s") as run:
            run.metric("acc", 0.5, entity="x")
        self.assertEqual(rc.main(["verify", self.path]), 0)
        self.assertEqual(rc.main(["leaderboard", self.path, "acc"]), 0)
        bundle = os.path.join(self.dir.name, "b.jsonl")
        self.assertEqual(rc.main(["export", self.path, bundle]), 0)
        self.assertEqual(rc.main(["verify-bundle", bundle]), 0)


if __name__ == "__main__":
    unittest.main()
