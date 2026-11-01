"""End-to-end pipeline run, recorded by the experiment recorder.

data engine (real CSV) -> data-derived TaskSpec -> generator rounds ->
hypothesis filter -> math gate -> development scoring with feedback to the
generator -> cross-round leaderboard (with tuned baselines) -> preregistration
-> one sealed test evaluation through the recorder's holdout ledger -> refused
second look -> journal verification and portable bundle.

Also measures recorder throughput and verification cost.

Usage:
  python demo_pipeline.py --csv heart.csv --target HeartDisease --journal run.sqlite [--rounds 5 --per-round 100]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "data_engine"))
sys.path.insert(0, os.path.join(ROOT, "math_update_gate"))
import data_engine as de  # noqa: E402
import recorder as rc  # noqa: E402

SOURCES = [
    "generator_only/hypothesis_generator.py",
    "hypothesis_filter/hypothesis_filter.py",
    "math_verifier/math_checker.py",
    "math_update_gate/math_filters.py",
    "data_engine/data_engine.py",
    "recorder/recorder.py",
    "recorder/demo_pipeline.py",
]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relative))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def balanced_accuracy(truth, pred):
    return float(np.mean([np.mean(pred[truth == c] == c) for c in np.unique(truth)]))


def bootstrap_difference(truth, a, b, reps=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(truth)
    diffs = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, n, n)
        t = truth[idx]
        if len(np.unique(t)) < 2:
            diffs[i] = np.nan
            continue
        diffs[i] = balanced_accuracy(t, a[idx]) - balanced_accuracy(t, b[idx])
    diffs = diffs[np.isfinite(diffs)]
    return {"mean": float(diffs.mean()), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "share_positive": float(np.mean(diffs > 0)), "reps": int(len(diffs))}


def throughput(directory):
    path = os.path.join(directory, "throughput.sqlite")
    rec = rc.Recorder(path)
    out = {}
    with rec.run("throughput", "benchmark") as run:
        t0 = time.perf_counter()
        with rec.batch():
            for i in range(20000):
                run.metric("x", float(i), entity=f"e{i % 100}")
        out["batched_events_per_second"] = round(20000 / (time.perf_counter() - t0))
        t0 = time.perf_counter()
        for i in range(1000):
            run.metric("y", float(i))
        out["single_transaction_events_per_second"] = round(1000 / (time.perf_counter() - t0))
    t0 = time.perf_counter()
    report = rec.verify()
    out["verify_seconds"] = round(time.perf_counter() - t0, 3)
    out["verify_events"] = report["events"]
    out["verify_ok"] = report["ok"]
    t0 = time.perf_counter()
    rec.leaderboard("x", k=5)
    out["leaderboard_seconds_over_20k_metrics"] = round(time.perf_counter() - t0, 3)
    out["journal_bytes"] = os.path.getsize(path)
    rec.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--per-round", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output")
    args = ap.parse_args(argv)
    if os.path.exists(args.journal):
        raise SystemExit("refusing to reuse an existing journal; choose a new path")
    started = time.perf_counter()
    gen = load("generator_only/hypothesis_generator.py", "hypothesis_generator")
    hf = load("hypothesis_filter/hypothesis_filter.py", "hypothesis_filter")
    import math_filters as mf
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rec = rc.Recorder(args.journal, sources=[os.path.join(ROOT, s) for s in SOURCES], repo=ROOT)
    work = tempfile.mkdtemp(prefix="pipeline_")
    summary = {"journal_id": rec.journal_id, "environment": {"python": platform.python_version(), "numpy": np.__version__}}

    # 1. Data engine on a real CSV.
    with rec.run("data engine", "data", params={"csv": os.path.basename(args.csv), "target": args.target}) as run:
        table = de.read_csv(args.csv)
        card = de.DataEngine().analyze(table, args.target, seed=args.seed)
        run.artifact(json.dumps(card.to_dict(), sort_keys=True, default=de._default), "data_card.json", media="application/json")
        run.entity(f"dataset:{table.name}", "dataset", {
            "raw_sha256": table.raw_sha256, "verdict": card.verdict,
            "test_fingerprint": card.record["split"]["fingerprints"]["test"],
            "goals": [g for g, v in card.record["goals"].items() if v["active"]],
        })
        if card.verdict == "BLOCKED":
            raise SystemExit(card.summary())
        train, dev = card.generator_partitions(gen)
        spec = card.task_spec(gen)
    summary["data"] = {"dataset": table.name, "verdict": card.verdict, "task_spec": {
        "kind": spec.kind, "features": spec.features, "classes": spec.classes, "goals": list(spec.goals)},
        "split": card.record["split"]["sizes"], "card_sha256": card.card_hash()}
    dev_hash = hashlib.sha256(dev.x.tobytes() + dev.y.tobytes()).hexdigest()

    # 2. Rounds: generate -> filter -> math gate -> development score -> feedback.
    generator = gen.HypothesisGenerator(seed=args.seed)
    screen = hf.FiveLayerFilter(os.path.join(work, "filter.sqlite"), generator_path=os.path.join(ROOT, SOURCES[0]))
    math_engine = mf.MathFilterEngine()
    rounds = []
    for r in range(1, args.rounds + 1):
        info = {"round": r}
        with rec.run(f"generate round {r}", "generator", params={"count": args.per_round, "seed": args.seed}) as run:
            hyps = generator.generate(spec, args.per_round)
            records = {h.signature: h for h in hyps}
            with rec.batch():
                for h in hyps:
                    a = h.algorithm
                    run.entity(h.signature, "hypothesis", {"round": r, "hid": h.hid, "strategy": h.strategy,
                               "parents": list(h.parents), "mechanism": a.mechanism, "loss": a.loss,
                               "feature_map": a.feature_map, "preconditioner": a.preconditioner})
            run.artifact("\n".join(json.dumps(h.to_dict(), sort_keys=True) for h in hyps), f"round{r}_hypotheses.jsonl")
        info["generated"] = len(hyps)
        with rec.run(f"filter round {r}", "hypothesis_filter") as run:
            audit = screen.evaluate_batch([h.to_dict() for h in hyps])
            passed = []
            with rec.batch():
                for d in audit["decisions"]:
                    sig = d["record"]["signature"] if d.get("record") else None
                    if sig is None:
                        continue
                    run.decision(sig, "hypothesis_filter", d["status"],
                                 [f"{x['name']}:{x['status']}" for x in d["layers"] if x["status"] != "PASS"])
                    if d["eligible_for_math"]:
                        passed.append(sig)
        info["filter_passed"] = len(passed)
        with rec.run(f"math round {r}", "math_gate") as run:
            math_ok = []
            with rec.batch():
                for sig in passed:
                    result = math_engine.assess(records[sig].to_dict())
                    run.decision(sig, "math_gate", result["mathematical_status"], result["reasons"][:6])
                    if result["mathematical_status"] == "PASS":
                        math_ok.append(sig)
        info["math_passed"] = len(math_ok)
        with rec.run(f"development round {r}", "development", params={"steps": 160, "learning_rate": 0.08}) as run:
            scored = []
            for sig in math_ok:
                h = records[sig]
                try:
                    model = gen.GeneratedModel(h).fit(train)
                    pred = model.predict(dev.x)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    run.decision(sig, "development", "FIT_FAILED", [type(exc).__name__])
                    continue
                score = balanced_accuracy(dev.y, np.asarray(pred))
                run.metric("dev_balanced_accuracy", score, split="development", entity=sig, dataset=table.name)
                digest = hashlib.sha256(np.asarray(pred).tobytes() + dev_hash.encode()).hexdigest()
                generator.observe(gen.Feedback(sig, score, "development", f"{table.name}:development", digest))
                scored.append(score)
        info["scored"] = len(scored)
        info["best_dev_this_round"] = max(scored) if scored else None
        rounds.append(info)
    summary["rounds"] = rounds

    # 3. Tuned baselines on the same development split.
    with rec.run("baselines", "baseline") as run:
        configs = [("logreg", {"C": c}) for c in (0.01, 0.1, 1.0, 10.0)] + [
            ("boosting", {"learning_rate": lr, "max_depth": md}) for lr in (0.05, 0.1) for md in (None, 3)]
        for family, params in configs:
            if family == "logreg":
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, **params))
            else:
                model = HistGradientBoostingClassifier(random_state=0, **params)
            model.fit(train.x, train.y)
            entity = f"baseline:{family}:" + ",".join(f"{k}={v}" for k, v in params.items())
            run.entity(entity, "baseline", {"family": family, "params": params})
            run.metric("dev_balanced_accuracy", balanced_accuracy(dev.y, model.predict(dev.x)),
                       split="development", entity=entity, dataset=table.name)

    board = rec.leaderboard("dev_balanced_accuracy", k=200)
    generated = [b for b in board if not b["entity"].startswith("baseline:")]
    baselines = [b for b in board if b["entity"].startswith("baseline:")]
    winner, best_baseline = generated[0], baselines[0]
    summary["leaderboard_top5"] = [{"entity": b["entity"][:16] if not b["entity"].startswith("baseline") else b["entity"],
                                    "dev": round(b["mean"], 6)} for b in board[:5]]
    summary["winner_round"] = next(e["body"]["attrs"]["round"] for e in rec.lineage(winner["entity"]) if e["kind"] == "entity")

    # 4. Preregister, then exactly one look at the sealed test partition.
    rec.preregister([winner["entity"], best_baseline["entity"]],
                    "best generated candidate across all rounds vs best tuned baseline, both chosen on development "
                    "balanced accuracy; each evaluated once on the sealed test partition",
                    metric="dev_balanced_accuracy")
    test_x, test_y = card.unseal_test("final preregistered comparison", ledger=rec)
    with rec.run("final test evaluation", "science") as run:
        h = gen.Hypothesis.from_dict(json.loads(next(
            line for line in rec.get_blob(next(
                e["body"]["sha256"] for e in rec.events(kind="artifact")
                if e["body"]["name"] == f"round{summary['winner_round']}_hypotheses.jsonl")).decode().splitlines()
            if json.loads(line)["signature"] == winner["entity"])))
        pred_gen = np.asarray(gen.GeneratedModel(h).fit(train).predict(test_x))
        family, _, params = best_baseline["entity"].split(":", 2)
        kwargs = {k: (None if v == "None" else (int(v) if v.isdigit() else float(v)))
                  for k, v in (kv.split("=") for kv in params.split(","))}
        base = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, **kwargs)) if family == "logreg"
                else HistGradientBoostingClassifier(random_state=0, **kwargs))
        pred_base = base.fit(train.x, train.y).predict(test_x)
        fp = card.record["split"]["fingerprints"]["test"]
        run.metric("test_balanced_accuracy", balanced_accuracy(test_y, pred_gen), split="test", entity=winner["entity"], dataset=fp)
        run.metric("test_balanced_accuracy", balanced_accuracy(test_y, pred_base), split="test", entity=best_baseline["entity"], dataset=fp)
        run.artifact(json.dumps({"generated": pred_gen.tolist(), "baseline": pred_base.tolist()}), "test_predictions.json")
        comparison = bootstrap_difference(test_y, pred_gen, pred_base)
        run.note(f"paired bootstrap of test balanced-accuracy difference: {json.dumps(comparison)}")
    summary["final"] = {
        "generated_candidate": winner["entity"][:16], "generated_dev": winner["mean"],
        "generated_test": balanced_accuracy(test_y, pred_gen),
        "baseline": best_baseline["entity"], "baseline_dev": best_baseline["mean"],
        "baseline_test": balanced_accuracy(test_y, pred_base),
        "difference_bootstrap": comparison, "test_rows": int(len(test_y)),
    }
    try:
        card.unseal_test("one more look after seeing the result", ledger=rec)
        summary["second_look"] = "ALLOWED (unexpected)"
    except rc.HoldoutReuseError as exc:
        summary["second_look"] = f"REFUSED: {exc}"

    # 5. Verify, anchor, export.
    head = rec.head()
    report = rec.verify(expected_head=head)
    bundle = os.path.splitext(args.journal)[0] + ".bundle.jsonl"
    rec.export_bundle(bundle)
    summary["journal"] = {"verify_ok": report["ok"], "events": report["events"], "blobs": report["blobs"],
                          "head": head, "bundle_verify_ok": rc.verify_bundle(bundle, expected_head=head)["ok"],
                          "report": rec.report()}
    summary["winner_lineage"] = [{"seq": e["seq"], "kind": e["kind"], "stage": e["stage"],
                                  "status": e["body"].get("status"), "value": e["body"].get("value"),
                                  "split": e["body"].get("split")} for e in rec.lineage(winner["entity"])]
    summary["throughput"] = throughput(work)
    summary["seconds"] = round(time.perf_counter() - started, 2)
    rec.close()
    text = json.dumps(summary, indent=2, sort_keys=True, default=str)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
