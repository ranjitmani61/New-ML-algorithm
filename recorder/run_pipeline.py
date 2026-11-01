"""Research pipeline, recorded end to end by the experiment recorder.

Discovery (discovery datasets only; validation data is never read here)
  data engine -> data-derived TaskSpec -> generator rounds -> hypothesis filter
  -> math gate -> development scoring, fed back to the generator
  -> cross-round leaderboard -> top-K handoff

Validation (a separate cohort, checked to be disjoint from discovery)
  data engine per dataset, with encoders fitted on exactly the rows that
  scientific_validation will train on -> predeclared exclusion rules
  -> cohort-overlap and statistical-power checks
  -> scientific_validation develop (chooses one of the K on its own development
     splits) -> freeze -> recorder preregistration bound to the frozen manifest
  -> recorder holdout ledger (refuses a cohort dataset already used for a final
     evaluation in this journal) -> evaluate (single consumption)
  -> decision, gates, comparison statistics and per-dataset test scores recorded

Close: journal verification against its head and a portable bundle.

Usage
  python run_pipeline.py --example-config > config.json        # then edit paths
  python run_pipeline.py config.json --journal run.sqlite \\
      --science-journal science.sqlite --output summary.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import sqlite3
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _sub in ("recorder", "data_engine", "math_update_gate", "scientific_validation"):
    sys.path.insert(0, os.path.join(ROOT, _sub))
import data_engine as de  # noqa: E402
import recorder as rc  # noqa: E402

GENERATOR = os.path.join(ROOT, "generator_only", "hypothesis_generator.py")
SOURCES = [
    "generator_only/hypothesis_generator.py",
    "hypothesis_filter/hypothesis_filter.py",
    "math_verifier/math_checker.py",
    "math_update_gate/math_filters.py",
    "scientific_validation/scientific_validation.py",
    "data_engine/data_engine.py",
    "recorder/recorder.py",
    "recorder/run_pipeline.py",
]
ALIGNMENT_PASSES = 12

EXAMPLE_CONFIG = {
    "data_root": ".",
    "discovery": [{"name": "heart", "source": "heart.csv", "target": "HeartDisease"}],
    "validation": [
        {"name": "titanic", "source": "Titanic-Dataset.csv", "target": "Survived"},
        {"name": "telco_churn", "source": "WA_Fn-UseC_-Telco-Customer-Churn.csv", "target": "Churn"},
        {"name": "pima", "source": "diabetes.csv", "target": "Outcome"},
        {"name": "breast_cancer", "source": "data.csv", "target": "diagnosis"},
        {"name": "liver", "source": "indian_liver_patient.csv", "target": "Dataset"},
        {"name": "orthopedic", "source": "column_3C_weka.csv", "target": "class"},
        {"name": "wine_quality", "source": "winequality-red.csv", "target": "quality"},
        {"name": "glass", "source": "glass.csv", "target": "Type"},
        {"name": "mushrooms", "source": "mushrooms.csv", "target": "class"},
        {"name": "wine", "source": "builtin:wine", "target": "target"},
        {"name": "iris", "source": "builtin:iris", "target": "target"},
        {"name": "digits", "source": "builtin:digits", "target": "target"},
    ],
    "rounds": 5,
    "per_round": 100,
    "candidates": 12,
    "seed": 7,
    "max_rows": 10000,
    "protocol": {},
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_table(spec, root):
    if "table" in spec:
        return spec["table"]
    source = spec["source"]
    if source.startswith("builtin:"):
        table = de.load_builtin(source.split(":", 1)[1])
        table.name = spec["name"]
        return table
    path = source if os.path.isabs(source) else os.path.join(root, source)
    return de.read_csv(path, spec["name"])


def adapt(gen, h, features, classes):
    """Same hypothesis, re-declared for a dataset's width (as scientific_validation does)."""
    t = h.task
    if t.features == features and t.classes == classes:
        return h
    task = gen.TaskSpec(t.kind, features, classes, goals=t.goals, max_parameters=t.max_parameters,
                        max_state_vectors=t.max_state_vectors)
    return gen.Hypothesis(h.algorithm, task, "construct", (), h.seed, h.round_no)


def development_score(kind, truth, pred, train_y):
    """Balanced accuracy, or 1 - MSE / training variance clipped to [0, 1]."""
    pred = np.asarray(pred)
    if kind == "classification":
        return float(np.mean([np.mean(pred[truth == c] == c) for c in np.unique(truth)]))
    variance = max(float(np.var(train_y)), 1e-12)
    return float(min(1.0, max(0.0, 1.0 - np.mean((truth - pred) ** 2) / variance)))


def power(n, alpha, family):
    """Smallest attainable Holm-adjusted p for the exact dataset-level sign-flip test."""
    minimum = 2.0 / 2**n
    needed = next(k for k in range(1, 64) if family * 2.0 / 2**k <= alpha)
    return {
        "independent_datasets": n,
        "holm_family_size": family,
        "alpha": alpha,
        "minimum_two_sided_p": minimum,
        "minimum_holm_adjusted_p": min(1.0, family * minimum),
        "significance_attainable": family * minimum <= alpha,
        "datasets_needed": needed,
    }


class Pipeline:
    def __init__(self, config, journal, science_journal):
        for path in (journal, science_journal, science_journal + ".filter.sqlite"):
            if os.path.exists(path):
                raise FileExistsError(f"refusing to reuse {path}; journals are single-study")
        self.config = config
        self.root = config.get("data_root", ".")
        self.science_path = science_journal
        self.gen = load_module(GENERATOR, "hypothesis_generator")
        self.hf = load_module(os.path.join(ROOT, "hypothesis_filter", "hypothesis_filter.py"), "hypothesis_filter")
        import math_filters
        import scientific_validation

        self.mf = math_filters
        self.sv = scientific_validation
        self.engine = de.DataEngine()
        self.rec = rc.Recorder(journal, sources=[os.path.join(ROOT, s) for s in SOURCES], repo=ROOT)
        self.protocol = self.sv.Protocol.parse(config.get("protocol") or {})
        self.summary = {"journal_id": self.rec.journal_id,
                        "environment": {"python": platform.python_version(), "numpy": np.__version__}}

    # -- discovery ------------------------------------------------------------
    def discovery(self):
        cfg, rec, gen = self.config, self.rec, self.gen
        cards, tables = [], []
        with rec.run("discovery data", "data") as run:
            for spec in cfg["discovery"]:
                table = load_table(spec, self.root)
                card = self.engine.analyze(table, spec["target"], kind=spec.get("kind"), groups=spec.get("groups"),
                                           drop=spec.get("drop", ()), seed=cfg.get("seed", 0))
                run.artifact(json.dumps(card.to_dict(), sort_keys=True, default=de._default),
                             f"data_card_{spec['name']}.json", media="application/json")
                run.entity(f"dataset:{spec['name']}", "dataset", {
                    "role": "discovery", "content_sha256": table.content_hash(), "verdict": card.verdict,
                    "goals": [g for g, v in card.record.get("goals", {}).items() if v["active"]]})
                if card.verdict == "BLOCKED":
                    raise SystemExit(f"discovery dataset blocked:\n{card.summary()}")
                cards.append(card)
                tables.append(table)
        kinds = {c.record["kind"] for c in cards}
        if len(kinds) != 1:
            raise SystemExit("discovery datasets must share one task kind")
        self.kind = kinds.pop()
        spec = cards[0].task_spec(gen)
        goals = sorted({g for c in cards for g, v in c.record["goals"].items() if v["active"]} & set(gen.GOALS))
        if self.kind == "regression":
            goals = [g for g in goals if g != "class_imbalance"]
        spec = gen.TaskSpec(spec.kind, spec.features, spec.classes, goals=tuple(goals))
        parts = [c.generator_partitions(gen) for c in cards]
        self.discovery_tables = tables
        self.summary["discovery"] = {
            "datasets": [{"name": s["name"], "verdict": c.verdict, "split": c.record["split"]["sizes"]}
                         for s, c in zip(cfg["discovery"], cards)],
            "task_spec": {"kind": spec.kind, "features": spec.features, "classes": spec.classes,
                          "goals": list(spec.goals)},
            "test_partitions": "sealed and never opened (final evidence comes from the validation cohort)",
        }
        generator = gen.HypothesisGenerator(seed=cfg.get("seed", 0))
        screen = self.hf.FiveLayerFilter(self.science_path + ".filter.sqlite", generator_path=GENERATOR)
        math_engine = self.mf.MathFilterEngine()
        self.records = {}
        rounds = []
        for r in range(1, cfg["rounds"] + 1):
            info = {"round": r}
            with rec.run(f"generate round {r}", "generator", params={"count": cfg["per_round"]}) as run:
                hyps = generator.generate(spec, cfg["per_round"])
                self.records.update({h.signature: h for h in hyps})
                with rec.batch():
                    for h in hyps:
                        a = h.algorithm
                        run.entity(h.signature, "hypothesis", {
                            "round": r, "hid": h.hid, "strategy": h.strategy, "parents": list(h.parents),
                            "mechanism": a.mechanism, "loss": a.loss, "feature_map": a.feature_map,
                            "preconditioner": a.preconditioner})
                run.artifact("\n".join(json.dumps(h.to_dict(), sort_keys=True) for h in hyps),
                             f"round{r}_hypotheses.jsonl")
            info["generated"] = len(hyps)
            with rec.run(f"filter round {r}", "hypothesis_filter") as run:
                audit = screen.evaluate_batch([h.to_dict() for h in hyps])
                passed = []
                with rec.batch():
                    for d in audit["decisions"]:
                        if not d.get("record"):
                            continue
                        sig = d["record"]["signature"]
                        run.decision(sig, "hypothesis_filter", d["status"],
                                     [f"{x['name']}:{x['status']}" for x in d["layers"] if x["status"] != "PASS"])
                        if d["eligible_for_math"]:
                            passed.append(sig)
            info["filter_passed"] = len(passed)
            with rec.run(f"math round {r}", "math_gate") as run:
                admitted = []
                with rec.batch():
                    for sig in passed:
                        result = math_engine.assess(self.records[sig].to_dict())
                        run.decision(sig, "math_gate", result["mathematical_status"], result["reasons"][:6])
                        if result["mathematical_status"] == "PASS":
                            admitted.append(sig)
            info["math_passed"] = len(admitted)
            with rec.run(f"development round {r}", "development", params={"steps": 160, "learning_rate": 0.08}) as run:
                scores = []
                for sig in admitted:
                    values, digests = [], []
                    for (train, dev), spec_d in zip(parts, cfg["discovery"]):
                        try:
                            h = adapt(gen, self.records[sig], train.x.shape[1],
                                      len(np.unique(train.y)) if self.kind == "classification" else 1)
                            pred = gen.GeneratedModel(h).fit(train).predict(dev.x)
                        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                            values = None
                            run.decision(sig, "development", "FIT_FAILED", [type(exc).__name__, spec_d["name"]])
                            break
                        values.append(development_score(self.kind, dev.y, pred, train.y))
                        digests.append(hashlib.sha256(np.asarray(pred).tobytes() + dev.x.tobytes()).hexdigest())
                    if values is None:
                        continue
                    score = float(np.mean(values))
                    run.metric("dev_score", score, split="development", entity=sig,
                               dataset="+".join(s["name"] for s in cfg["discovery"]))
                    generator.observe(gen.Feedback(sig, score, "development", "discovery-development",
                                                   hashlib.sha256("".join(digests).encode()).hexdigest()))
                    scores.append(score)
            info["scored"] = len(scores)
            info["best_dev_this_round"] = max(scores) if scores else None
            rounds.append(info)
        self.summary["rounds"] = rounds
        k = min(cfg["candidates"], self.protocol.budget // 2)
        board = rec.leaderboard("dev_score", k=k)
        if not board:
            raise SystemExit("no candidate survived discovery")
        self.handoff = [b["entity"] for b in board]
        with rec.run("handoff", "handoff", params={"k": k}) as run:
            for b in board:
                run.decision(b["entity"], "handoff", "HANDED_OFF", [f"discovery dev {b['mean']:.6f}", f"rank {b['rank']}"])
        self.summary["handoff"] = [{"entity": b["entity"][:16], "dev_score": round(b["mean"], 6),
                                    "round": self._round_of(b["entity"])} for b in board]

    def _round_of(self, entity):
        return next(e["body"]["attrs"]["round"] for e in self.rec.lineage(entity) if e["kind"] == "entity")

    # -- validation cohort ----------------------------------------------------
    def validation_cohort(self):
        cfg, rec, sv, gen = self.config, self.rec, self.sv, self.gen
        accepted, prepared, rows = [], [], []
        with rec.run("validation data", "data", params={"max_rows": cfg.get("max_rows", 10000)}) as run:
            for spec in cfg["validation"]:
                row = {"name": spec["name"]}
                rows.append(row)
                ds, card, reason = self._prepare(spec, row)
                if ds is not None:
                    try:
                        part = sv.prepare_dataset(ds, self.protocol, gen)
                        sv.check_cohort(prepared + [part])
                    except ValueError as exc:
                        ds, reason = None, f"validator preparation or cohort check: {exc}"
                if ds is None:
                    row["status"] = "EXCLUDED"
                    row["reason"] = reason
                    run.decision(f"dataset:{spec['name']}", "validation_cohort", "EXCLUDED", [reason])
                    continue
                prepared.append(part)
                accepted.append(ds)
                row.update(status="INCLUDED", content_hash=part["meta"]["content_hash"],
                           science_split={k: part["meta"]["partitions"][k]["size"] for k in ("train", "development", "test")})
                run.artifact(json.dumps(card.to_dict(), sort_keys=True, default=de._default),
                             f"data_card_{spec['name']}.json", media="application/json")
                run.entity(f"dataset:{spec['name']}", "dataset", {
                    "role": "validation", "content_hash": part["meta"]["content_hash"], "verdict": card.verdict,
                    "encoder_alignment_passes": row["alignment_passes"], "rows": row["rows"]})
                run.decision(f"dataset:{spec['name']}", "validation_cohort", "INCLUDED", [card.verdict])
        self.cohort, self.prepared = accepted, prepared
        family = 9  # reference + five baseline families + three controller controls
        self.summary["validation_cohort"] = rows
        self.summary["power"] = power(len(accepted), self.protocol.alpha, family)
        if len(accepted) < 2:
            raise SystemExit("fewer than two validation datasets remain; scientific validation needs 2..20")

    def _prepare(self, spec, row):
        cfg, gen = self.config, self.gen
        table = load_table(spec, self.root)
        row["rows"] = len(table.rows)
        if len(table.rows) > cfg.get("max_rows", 10000):
            return None, None, f"{len(table.rows)} rows exceed the predeclared compute limit {cfg.get('max_rows', 10000)}"
        for disc in self.discovery_tables:
            o = de.cohort_overlap([disc, table])[0]
            if o["identical_content"] or (o["comparable"] and o["shared_rows"]):
                return None, None, f"overlaps discovery dataset {disc.name}"
        kw = {"kind": spec.get("kind"), "groups": spec.get("groups"), "drop": spec.get("drop", ()),
              "seed": cfg.get("seed", 0)}
        card = self.engine.analyze(table, spec["target"], **kw)
        row["verdict_first_pass"] = card.verdict
        if card.verdict == "BLOCKED":
            return None, card, "data card BLOCKED: " + "; ".join(f"{w['code']}" for w in card.blocking)
        if card.record["kind"] != self.kind:
            return None, card, f"task kind {card.record['kind']} differs from discovery kind {self.kind}"
        # Fit encoders on exactly the rows the validator will train on. The
        # validator keeps encoded duplicates together, and which rows collide
        # after encoding depends on the encoders, so the group labels handed to
        # it accumulate every collision seen (they only coarsen, hence the loop
        # terminates); once they are stable the split is fixed and the next pass
        # confirms that the validator reproduces exactly the declared split.
        source = f"{card.record['source']} (raw sha256 {table.raw_sha256})"
        exposure = spec.get("historical_exposure", True)
        parent = list(range(len(card.used_rows)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def merge(labels):
            first = {}
            for i, key in enumerate(labels):
                if key in first:
                    a, b = find(i), find(first[key])
                    if a != b:
                        parent[max(a, b)] = min(a, b)
                else:
                    first[key] = i

        merge(card.science_dataset(self.sv, name=spec["name"]).groups)
        for attempt in range(1, ALIGNMENT_PASSES + 1):
            ds = card.science_dataset(self.sv, name=spec["name"], historical_exposure=exposure, source=source)
            merge([gen._row_hash(r) for r in ds.x])
            labels = [f"{spec['name']}:g{find(i)}" for i in range(len(parent))]
            ds = card.science_dataset(self.sv, name=spec["name"], historical_exposure=exposure, source=source,
                                      groups=labels)
            parts = gen.split_data(ds.x, ds.y, seed=self.protocol.split_seed, groups=np.asarray(ds.groups))
            used = card.used_rows
            declared = {p.role: sorted(used[int(i)] for i in p.row_ids) for p in parts}
            if card.record["split"]["method"] == "declared" and card.partition_rows == declared:
                row["alignment_passes"] = attempt
                row["validator_groups"] = len(set(labels))
                row["verdict"] = card.verdict
                row["warnings"] = sorted({w["code"] for w in card.warnings if w["severity"] != "info"})
                row["goals"] = [g for g, v in card.record["goals"].items() if v["active"]]
                return ds, card, None
            card = self.engine.analyze(table, spec["target"], split=declared, **kw)
            if card.verdict == "BLOCKED":
                return None, card, "data card BLOCKED under the validator's split: " + "; ".join(
                    w["code"] for w in card.blocking)
        return None, card, f"encoder/validator split did not reach a fixed point in {ALIGNMENT_PASSES} passes"

    # -- scientific validation -------------------------------------------------
    def science(self):
        rec, sv = self.rec, self.sv
        engine = sv.ScientificEngine(self.science_path, generator_path=GENERATOR, protocol=self.protocol)
        candidates = [self.records[s].to_dict() for s in self.handoff]
        with rec.run("scientific development", "science_development",
                     params={"protocol": json.loads(json.dumps(dataclasses.asdict(self.protocol), default=str)),
                             "candidates": len(candidates), "datasets": [d.name for d in self.cohort]}) as run:
            choices = engine.develop(self.cohort, candidates=candidates)
            development = self._science_state("development")
            by_config = development["configurations"]
            with rec.batch():
                for trial in development["trials"]:
                    spec = by_config[trial["config_id"]]
                    if trial["family"] == "candidate" and trial["success"]:
                        run.metric(f"science_dev_score[lr={spec['learning_rate']}]", trial["score"],
                                   split="development", entity=spec["record"]["signature"], dataset=trial["dataset"])
            selected_spec = by_config[choices["candidate"]]
            self.selected = selected_spec["record"]["signature"]
            run.decision(self.selected, "science_development", "SELECTED",
                         [f"config {choices['candidate'][:16]}", f"learning_rate {selected_spec['learning_rate']}"])
            run.note("validator reference families by dataset: " + json.dumps(choices["reference"], sort_keys=True))
        with rec.run("scientific freeze", "science_freeze") as run:
            manifest = engine.freeze()
            manifest_hash = sv.digest(manifest)
            run.artifact(sv.canonical(manifest), "science_manifest.json", media="application/json")
        family = len(manifest["comparison_family"])
        self.summary["power"] = power(len(self.cohort), self.protocol.alpha, family)
        comparators = ["science:reference"] + [f"science:{m}" for m in sv.METHODS if m != "candidate"]
        rec.preregister([self.selected] + comparators,
                        f"scientific_validation frozen manifest {manifest_hash}: the selected candidate and its "
                        f"{family}-comparison family are evaluated once on the sealed test partitions of "
                        f"{len(self.cohort)} validation datasets; selection used development splits only",
                        metric="dev_score")
        for part in self.prepared:
            rec.holdout_access(f"science-dataset:{part['meta']['content_hash']}",
                               purpose=f"scientific_validation final evaluation, manifest {manifest_hash[:16]}",
                               study=os.path.basename(self.science_path))
        with rec.run("scientific evaluation", "science", params={"manifest": manifest_hash}) as run:
            decision = engine.evaluate()
            evidence = engine.journal.evidence()
            means = {}
            for r in evidence["runs"]:
                if r.get("success") and "scores" in r:
                    means.setdefault((r["dataset"], r["method"]), []).append(r["scores"]["clean"])
            reference = manifest["reference"]
            with rec.batch():
                for (dataset, method), values in sorted(means.items()):
                    entity = self.selected if method == "candidate" else f"science:{method}"
                    run.metric("science_test_score", float(np.mean(values)), split="test", entity=entity,
                               dataset=dataset)
                    if method == reference[dataset]:
                        run.metric("science_test_score", float(np.mean(values)), split="test",
                                   entity="science:reference", dataset=dataset)
                for comparison, stats in decision["statistics"].items():
                    for key in ("mean_effect", "p_value", "holm_p"):
                        if isinstance(stats.get(key), (int, float)):
                            run.metric(f"{key}_vs_{comparison}", float(stats[key]), split="test",
                                       entity=self.selected, dataset="cohort")
                run.decision(self.selected, "scientific_validation", decision["status"],
                             [f"{g}:{v['status']}" for g, v in decision["gates"].items()])
            run.artifact(json.dumps(decision, sort_keys=True), "science_decision.json", media="application/json")
            run.note(f"science journal {os.path.basename(self.science_path)} evidence {decision['evidence_hash']}")
        self.summary["science"] = {
            "selected": self.selected[:16],
            "selected_round": self._round_of(self.selected),
            "selected_learning_rate": selected_spec["learning_rate"],
            "status": decision["status"],
            "accepted": decision["accepted"],
            "gates": {g: v["status"] for g, v in decision["gates"].items()},
            "statistics": {c: {k: s.get(k) for k in ("mean_effect", "p_value", "holm_p", "simultaneous_ci")}
                           for c, s in decision["statistics"].items()},
            "per_dataset": {d: {"candidate": round(float(np.mean(means[(d, "candidate")])), 6),
                                "reference": round(float(np.mean(means[(d, reference[d])])), 6),
                                "reference_family": reference[d],
                                "fixed_matched": round(float(np.mean(means[(d, "fixed_matched")])), 6)}
                            for d in sorted(reference) if (d, "candidate") in means},
            "fit_failures": len(decision["fit_failures"]),
            "manifest_hash": manifest_hash,
            "evidence_hash": decision["evidence_hash"],
        }

    def _science_state(self, column):
        db = sqlite3.connect(self.science_path)
        try:
            return json.loads(db.execute(f"SELECT {column} FROM state WHERE id=1").fetchone()[0])
        finally:
            db.close()

    # -- close ------------------------------------------------------------------
    def close(self, bundle):
        head = self.rec.head()
        report = self.rec.verify(expected_head=head)
        self.rec.export_bundle(bundle)
        self.summary["journal"] = {
            "verify_ok": report["ok"], "events": report["events"], "blobs": report["blobs"], "head": head,
            "bundle": os.path.basename(bundle), "bundle_verify_ok": rc.verify_bundle(bundle, expected_head=head)["ok"],
            "report": self.rec.report()}
        if getattr(self, "selected", None):
            trail = []
            for e in self.rec.lineage(self.selected):
                b = e["body"]
                if e["kind"] == "metric" and b["split"] == "development" and b["name"] != "dev_score":
                    continue  # per-dataset validator development scores are in the journal
                trail.append({"seq": e["seq"], "kind": e["kind"], "stage": e["stage"], "status": b.get("status"),
                              "metric": b.get("name"), "split": b.get("split"), "dataset": b.get("dataset")})
            self.summary["selected_lineage"] = trail[:60]
        self.rec.close()

    def run(self, bundle):
        start = time.perf_counter()
        timings = {}
        for stage in (self.discovery, self.validation_cohort, self.science):
            t0 = time.perf_counter()
            stage()
            timings[stage.__name__] = round(time.perf_counter() - t0, 2)
        self.close(bundle)
        timings["total"] = round(time.perf_counter() - start, 2)
        self.summary["seconds"] = timings
        return self.summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Recorded research pipeline")
    ap.add_argument("config", nargs="?")
    ap.add_argument("--journal")
    ap.add_argument("--science-journal")
    ap.add_argument("--output")
    ap.add_argument("--example-config", action="store_true", help="print an example configuration and exit")
    args = ap.parse_args(argv)
    if args.example_config:
        print(json.dumps(EXAMPLE_CONFIG, indent=2))
        return 0
    if not (args.config and args.journal and args.science_journal):
        ap.error("config, --journal and --science-journal are required")
    with open(args.config) as fh:
        config = json.load(fh)
    # Relative data paths are relative to the configuration file, not the shell.
    root = config.get("data_root", ".")
    if not os.path.isabs(root):
        config["data_root"] = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(args.config)), root))
    pipeline = Pipeline(config, args.journal, args.science_journal)
    summary = pipeline.run(os.path.splitext(args.journal)[0] + ".bundle.jsonl")
    text = json.dumps(summary, indent=2, sort_keys=True, default=str)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
