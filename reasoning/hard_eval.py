"""Hard-setting (MATH level 5) evaluation of terminal-ratio voting against tuned baselines.

    python3 hard_eval.py dev  POOLS --out results/hard_dev_eval.json
    python3 hard_eval.py freeze POOLS --spec results/hard_spec.json --journal results/journal.sqlite
    python3 hard_eval.py test POOLS --spec results/hard_spec.json --journal results/journal.sqlite --out ...

``freeze`` picks the confidence-vote baseline settings on development data, writes the
specification (methods, budgets, primary statistic) and preregisters it in the recorder
journal. ``test`` refuses to run unless that specification is preregistered, registers the
single access to the sealed test pools (by content fingerprint) before reading them, and
evaluates exactly the frozen specification once.

All methods see the same replayed orderings of each question's 32 attempts, so comparisons
at equal k are paired at equal generated tokens (the same attempts are used).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "recorder"))
import grading  # noqa: E402
import lab  # noqa: E402

KS = (4, 8, 16, 32)
PRIMARY_KS = (8, 16, 32)
GAMMA = 2.0
REPEATS = 40
SEED = 20261101


def fingerprint(directory):
    h = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(directory, "**", "part*.*"), recursive=True)):
        h.update(os.path.relpath(path, directory).encode())
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def tune_conf(questions, repeats=20):
    """Dev-only choice of (measure, eta) for the confidence-weighted vote at each k."""
    out = {}
    for k in KS:
        best = None
        for m in ("conf_tail", "conf_mean", "conf_low"):
            for e in (1.0, 0.9, 0.75, 0.5):
                r = lab.simulate(questions, "conf", {"k": k, "measure": m, "eta": e}, repeats=repeats, seed=1)
                if best is None or r["accuracy"] > best[0]:
                    best = (r["accuracy"], m, e)
        out[str(k)] = {"measure": best[1], "eta": best[2], "dev_accuracy": best[0]}
    return out


def method_params(spec, name, k):
    if name == "sc":
        return "sc", {"k": k}
    if name == "conf":
        c = spec["conf"][str(k)]
        return "conf", {"k": k, "measure": c["measure"], "eta": c["eta"]}
    if name == "trv_vote":
        return "trv", {"k": k, "gamma": spec["gamma"], "measure": None}
    if name == "trv_conf":  # the tuned baseline's own weights, times tau**gamma: isolates the new factor
        c = spec["conf"][str(k)]
        return "trv", {"k": k, "gamma": spec["gamma"], "measure": c["measure"], "eta": c["eta"]}
    raise ValueError(name)


def evaluate(questions, spec):
    per_q, summary = {}, {}
    for name in ("sc", "conf", "trv_vote", "trv_conf"):
        for k in spec["ks"]:
            m, p = method_params(spec, name, k)
            r = lab.simulate(questions, m, p, repeats=spec["repeats"], seed=spec["seed"])
            per_q[(name, k)] = r["per_question"]
            summary[f"{name}@{k}"] = {"accuracy": r["accuracy"], "tokens": r["tokens"], "params": p}
    comparisons = {}
    for k in spec["ks"]:
        for a, b in (("trv_conf", "conf"), ("trv_vote", "sc"), ("trv_conf", "sc")):
            comparisons[f"{a}-{b}@{k}"] = lab.paired_bootstrap(per_q[(a, k)], per_q[(b, k)], seed=k)
    pa = np.mean([per_q[("trv_conf", k)] for k in spec["primary_ks"]], axis=0)
    pb = np.mean([per_q[("conf", k)] for k in spec["primary_ks"]], axis=0)
    primary = lab.paired_bootstrap(pa, pb, reps=10000, seed=0)
    primary["passed"] = primary["ci95"][0] > 0
    return {"questions": len(questions), "summary": summary, "comparisons": comparisons, "primary": primary}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("dev", "freeze", "test"))
    ap.add_argument("pools")
    ap.add_argument("--spec", default=os.path.join(HERE, "results", "hard_spec.json"))
    ap.add_argument("--journal", default=os.path.join(HERE, "results", "journal.sqlite"))
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    from recorder import Recorder

    if args.mode in ("dev", "freeze"):
        questions = grading.load_pools(args.pools, {"hard_dev"})
        conf = tune_conf(questions)
        spec = {"ks": list(KS), "primary_ks": list(PRIMARY_KS), "gamma": GAMMA, "repeats": REPEATS, "seed": SEED,
                "conf": conf, "dev_pools_sha256": fingerprint(args.pools),
                "primary": ("mean over k in {8,16,32} of per-question accuracy(trv_conf@k) - accuracy(conf@k), "
                            "conf = confidence-weighted vote with (measure, eta) tuned on dev per k; "
                            "paired bootstrap over test questions (10,000 resamples); "
                            "passes iff the 95% interval lies above 0")}
        result = evaluate(questions, spec)
        if args.mode == "freeze":
            text = json.dumps(spec, indent=2, sort_keys=True)
            with open(args.spec, "w") as fh:
                fh.write(text)
            with Recorder(args.journal, sources=[os.path.join(HERE, f) for f in
                                                 ("lab.py", "selection.py", "grading.py", "hard_eval.py")]) as rec:
                with rec.run("hard-dev-final", "development", params={"spec_sha256": hashlib.sha256(text.encode()).hexdigest()}) as run:
                    run.artifact(text.encode(), "hard_spec.json", media="application/json")
                    for name, s in result["summary"].items():
                        run.metric(f"accuracy@{name.split('@')[1]}", s["accuracy"], entity=name, dataset="hard_dev")
                    run.metric("primary_diff", result["primary"]["mean"], entity="trv_conf", dataset="hard_dev")
                rec.preregister(["trv_conf"], "Terminal-ratio voting (gamma=2) applied to the dev-tuned "
                                "confidence-weighted vote vs that vote alone, at equal k. Primary statistic: " + spec["primary"]
                                + f". Spec sha256 {hashlib.sha256(text.encode()).hexdigest()}.")
        out = json.dumps(result, indent=2, sort_keys=True)
        if args.out:
            with open(args.out, "w") as fh:
                fh.write(out)
        print(out[:6000])
        return 0

    # test: frozen spec, preregistered, single access
    text = open(args.spec).read()
    spec = json.loads(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    with Recorder(args.journal, create=False) as rec:
        prereg = [e for e in rec.events("preregistration") if digest in e["body"]["statement"]]
        if not prereg:
            raise SystemExit("this specification is not preregistered; run `freeze` first")
        fp = fingerprint(args.pools)
        rec.holdout_access(fp, "hard_test single evaluation of preregistered terminal-ratio voting", study="hard-math")
        questions = grading.load_pools(args.pools, {"hard_test"})
        result = evaluate(questions, spec)
        result["test_pools_sha256"], result["spec_sha256"] = fp, digest
        with rec.run("hard-test-final", "test", params={"spec_sha256": digest, "pools_sha256": fp}) as run:
            for name, s in result["summary"].items():
                run.metric(f"accuracy@{name.split('@')[1]}", s["accuracy"], split="test", entity=name, dataset="hard_test")
            run.metric("primary_diff", result["primary"]["mean"], split="test", entity="trv_conf", dataset="hard_test")
            run.artifact(json.dumps(result, sort_keys=True).encode(), "hard_test_result.json", media="application/json")
    out = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(out)
    print(out[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
