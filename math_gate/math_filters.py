"""Autonomous integrity and mathematical admission, before scientific validation.

One implementation file. Reuses explicitly pinned, trusted generator/verifier
sources. No data fitting, scientific scores, novelty decisions or gate upgrades.
READY means eligibility under the displayed policy, never a proven new algorithm.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, dataclass
from fractions import Fraction as F
import hashlib
import importlib.util
from itertools import product
import json
import math
from pathlib import Path
import re
import sys

VERSION = "math-admission-1.0"
GENERATOR_SHA256 = "8ce3010e741603c9627c85f37944586c4cde2155eeb24660adcfd58ca5931b3f"
VERIFIER_SHA256 = "37dc50931bfe2512875558e5660de692680c8b232333a98885503da9697205fc"
SIGNALS = ("disagreement", "dispersion", "alignment", "progress", "time", "energy")
FEEDBACK = tuple(s for s in SIGNALS if s != "time")
MAX_BYTES = 300_000
MAX_RECORD_NODES = 20_000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


LOADED_ADMISSION_SHA256 = file_hash(__file__)


def read_json(text):
    """Strict transport: duplicate keys and non-finite JSON numbers are errors."""
    if not isinstance(text, str) or len(text.encode()) > MAX_BYTES:
        raise ValueError("bounded UTF-8 JSON required")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("non-finite JSON number: " + value)

    def decimal(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite JSON number")
        return result

    try:
        return json.loads(
            text, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=decimal
        )
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds parser limit") from exc


def bounded_payload(value):
    """Check structure before serialization or generator recursion."""
    pending = [(value, 0)]
    count = 0
    text_bytes = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > 24 or count > MAX_RECORD_NODES:
            raise ValueError("record structure budget exceeded")
        if isinstance(item, dict):
            if any(type(k) is not str for k in item):
                raise ValueError("JSON object keys must be strings")
            text_bytes += sum(len(k.encode()) for k in item)
            pending.extend((x, depth + 1) for x in item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend((x, depth + 1) for x in item)
        elif item is None or type(item) in (bool, int, float, str):
            if type(item) is int and item.bit_length() > 1024:
                raise ValueError("record integer too large")
            if isinstance(item, str) and len(item) > MAX_BYTES:
                raise ValueError("record string too long")
            if isinstance(item, str):
                text_bytes += len(item.encode())
        else:
            raise ValueError("JSON values only")
        if text_bytes > MAX_BYTES:
            raise ValueError("record aggregate text budget exceeded")
    return read_json(canonical(value))


def strict_fields(value, fields):
    if not isinstance(value, dict) or set(value) - set(fields):
        raise ValueError("unknown fields or invalid object")


def exact_number(value):
    if type(value) is int and value.bit_length() > 1024:
        raise ValueError("exact integer too large")
    if type(value) not in (int, str) or len(str(value)) > 100:
        raise ValueError("exact integer/rational/decimal required")
    if isinstance(value, str):
        match = re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE]([+-]?\d{1,3}))?(?:/[+-]?\d+)?", value
        )
        if not match or (match[1] is not None and abs(int(match[1])) > 100):
            raise ValueError("bounded exact numeric syntax required")
    try:
        result = F(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid exact number") from exc
    if max(abs(result.numerator).bit_length(), result.denominator.bit_length()) > 1024:
        raise ValueError("exact number too large")
    return result


@dataclass(frozen=True)
class Runtime:
    steps: int = 160
    learning_rate: str = "2/25"
    seed: int = 0

    @classmethod
    def parse(cls, value=None):
        if isinstance(value, cls):
            value = asdict(value)
        value = {} if value is None else value
        strict_fields(value, cls.__dataclass_fields__)
        body = {**asdict(cls()), **value}
        if type(body["steps"]) is not int or not 1 <= body["steps"] <= 20_000:
            raise ValueError("steps must be an integer in [1,20000]")
        if type(body["seed"]) is not int or not 0 <= body["seed"] < 2**32:
            raise ValueError("execution seed must be an unsigned 32-bit integer")
        rate = exact_number(body["learning_rate"])
        if not 0 < rate <= 1:
            raise ValueError("learning_rate must be in (0,1]")
        return cls(body["steps"], str(rate), body["seed"])


@dataclass(frozen=True)
class Policy:
    require_convex_objective: bool = True
    min_feedback_signals: int = 2
    min_influence: str = "1/64"

    @classmethod
    def parse(cls, value=None):
        if isinstance(value, cls):
            value = asdict(value)
        value = {} if value is None else value
        strict_fields(value, cls.__dataclass_fields__)
        body = {**asdict(cls()), **value}
        if type(body["require_convex_objective"]) is not bool:
            raise ValueError("require_convex_objective must be Boolean")
        if (
            type(body["min_feedback_signals"]) is not int
            or not 1 <= body["min_feedback_signals"] <= 5
        ):
            raise ValueError("min_feedback_signals must be in [1,5]")
        influence = exact_number(body["min_influence"])
        if not 0 < influence <= 1:
            raise ValueError("min_influence must be in (0,1]")
        return cls(
            body["require_convex_objective"],
            body["min_feedback_signals"],
            str(influence),
        )


def scalar(lhs, op, rhs, variables=None):
    return {
        "kind": "scalar",
        "variables": {} if variables is None else variables,
        "lhs": str(lhs),
        "op": op,
        "rhs": str(rhs),
    }


def derivative(expression, variable, proposed, variables=None):
    return {
        "kind": "derivative",
        "variables": variables or {variable: {}},
        "expression": expression,
        "variable": variable,
        "proposed": proposed,
    }


def load_trusted(path, expected, name):
    path = Path(path).resolve()
    if file_hash(path) != expected:
        raise ValueError("unrecognized trusted source: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def controller_value(node, point):
    """Independent rational interpreter; no floating probes or evaluation of text."""
    op = node["op"]
    if op == "signal":
        return point[node["name"]]
    if op == "half":
        return F(1, 2)
    values = [controller_value(n, point) for n in node["children"]]
    if op == "complement":
        return 1 - values[0]
    if op == "mean":
        return sum(values) / 2
    if op == "product":
        return values[0] * values[1]
    if op == "minimum":
        return min(values)
    if op == "maximum":
        return max(values)
    raise ValueError("unsupported controller")


def effect(algorithm, q):
    """An actually used update coefficient, with its coupling explicitly bound."""
    mechanism = algorithm["mechanism"]
    if mechanism in ("clip_memory", "clip_step"):
        return F(1, 4) + F(3, 2) * (1 - q)
    if mechanism == "memory_step":
        return F(19, 20) * q
    if mechanism == "dual_memory":
        return q
    if mechanism in ("penalty_step", "variance_penalty"):
        return F(1, 4) + F(3, 2) * q
    raise ValueError("unsupported mechanism")


def effect_name(algorithm):
    return {
        "clip_memory": "clipping_threshold_multiplier",
        "clip_step": "clipping_threshold_multiplier",
        "memory_step": "fast_momentum_coefficient",
        "dual_memory": "fast_slow_mix_weight",
        "penalty_step": "penalty_multiplier",
        "variance_penalty": "penalty_multiplier",
    }[algorithm["mechanism"]]


class MathFilterEngine:
    """Trusted local admission engine. Session issuance is not a public signature.

    assess() can analyze external records; unissued records cannot become READY.
    run() registers actual generator emissions and checks their parent lineage.
    Receipt verification reconstructs the decision and explicitly replays proofs.
    """

    def __init__(self, generator_path=None, verifier_path=None):
        root = Path(__file__).resolve().parent.parent
        self.generator_path = Path(
            generator_path or root / "generator_only" / "hypothesis_generator.py"
        ).resolve()
        self.verifier_path = Path(
            verifier_path or root / "math_verifier" / "math_checker.py"
        ).resolve()
        self.gen = load_trusted(
            self.generator_path, GENERATOR_SHA256, "_admission_generator"
        )
        self.verifier = load_trusted(
            self.verifier_path, VERIFIER_SHA256, "_admission_verifier"
        )
        self.sources = {
            "generator": GENERATOR_SHA256,
            "verifier": VERIFIER_SHA256,
            "admission": LOADED_ADMISSION_SHA256,
        }
        self.checker = self.verifier.MathCheckerEngine()
        self._issued = {}
        self._issued_limit = 20_000

    def _sources_unchanged(self):
        return (
            file_hash(self.generator_path) == self.sources["generator"]
            and file_hash(self.verifier_path) == self.sources["verifier"]
            and file_hash(__file__) == self.sources["admission"]
        )

    def _check(self, name, claim):
        result = self.checker.check(claim)
        return {
            "name": name,
            "claim": claim,
            "status": result.status,
            "certificate": result.certificate,
            "reasons": result.reasons,
        }

    def _required_base(self, record):
        # Build the prescribed obligations independently of checker responses.
        # This trusted, pinned declarative compiler performs no proof discovery.
        verifier = self.verifier

        class DeclarationOnly:
            def check(self, claim):
                return verifier.Verdict("UNVERIFIED", "obligation declaration only")

        return verifier.generator_audit(record, DeclarationOnly()).subclaims

    def _objective(self, h):
        loss = h.algorithm.loss
        checks = []

        def add(name, claim):
            checks.append(self._check("objective/" + name, claim))

        positive = {"lower": "0", "lower_open": True}
        objective = {
            "loss": loss,
            "coordinates": "trainable scores / weights with fixed features",
            "scope": "per-example and mean data-fit loss; no claim about the adaptive update field",
            "status": "UNRESOLVED",
            "scheme": "",
            "witness": None,
        }
        if loss == "squared":
            add("squared_gradient", derivative("r*r/2", "r", "r"))
            add("squared_hessian", derivative("r", "r", "1"))
            add("squared_positive_curvature", scalar("1", ">", "0"))
            objective.update(
                status="CONVEX",
                scheme="affine_composition_of_nonnegative_second_derivative",
            )
        elif loss == "huber":
            for label, f, g in [
                ("left", "-r-1/2", "-1"),
                ("center", "r*r/2", "r"),
                ("right", "r-1/2", "1"),
            ]:
                add("huber_" + label + "_gradient", derivative(f, "r", g))
                add(
                    "huber_" + label + "_slope",
                    derivative(g, "r", "1" if label == "center" else "0"),
                )
            add("huber_nonnegative_slopes", scalar("1", ">=", "0"))
            for x in (-1, 1):
                add(
                    "huber_value_join_" + str(x),
                    scalar(f"({x})^2/2", "==", f"abs({x})-1/2"),
                )
                add(
                    "huber_gradient_join_" + str(x), scalar(x, "==", -1 if x < 0 else 1)
                )
            objective.update(
                status="CONVEX",
                scheme="continuous_piecewise_nondecreasing_gradient_huber_C1",
            )
        elif loss == "pseudo_huber":
            add(
                "pseudo_huber_gradient",
                derivative("sqrt(1+r*r)-1", "r", "r/sqrt(1+r*r)"),
            )
            v = {"r": {}, "s": positive}
            add("positive_root_exists", scalar("1+r*r", ">", "0", {"r": {}}))
            add(
                "root_relation_reduction",
                scalar("(s-r*(r/s))/(s*s)-1/(s*s*s)", "==", "(s*s-r*r-1)/(s*s*s)", v),
            )
            add(
                "positive_reduced_hessian",
                scalar("1/(s*s*s)", ">", "0", {"s": positive}),
            )
            objective.update(
                status="CONVEX",
                scheme="sqrt_relation_second_derivative",
                derived_identity="s=sqrt(1+r*r)>0; s*s=1+r*r; f_second=1/(s*s*s)",
            )
        elif loss == "cross_entropy":
            v = {"a": positive, "b": positive, "x": {}, "y": {}}
            add(
                "weighted_variance_merge",
                scalar("a*x*x+b*y*y-(a*x+b*y)^2/(a+b)", "==", "a*b*(x-y)^2/(a+b)", v),
            )
            add(
                "weighted_variance_nonnegative",
                scalar("a*b*(x-y)^2/(a+b)", ">=", "0", v),
            )
            add(
                "binary_curvature",
                scalar("p*(1-p)", ">=", "0", {"p": {"lower": "0", "upper": "1"}}),
            )
            objective.update(
                status="CONVEX",
                scheme="softmax_Hessian_weighted_variance_induction",
                derived_identity="v^T H v=sum_i p_i*(v_i-sum_j p_j*v_j)^2 >=0; p_i>0; sum_i p_i=1",
                classes=h.task.classes,
            )
        elif loss == "cauchy":
            add("cauchy_gradient", derivative("log(1+r*r)/2", "r", "r/(1+r*r)"))
            add("cauchy_hessian", derivative("r/(1+r*r)", "r", "(1-r*r)/(1+r*r)^2"))
            add("cauchy_negative_curvature", scalar("(1-2*2)/(1+2*2)^2", "<", "0"))
            objective.update(
                status="NONCONVEX",
                scheme="negative_second_derivative_on_affine_line",
                witness={"residual": "2", "curvature": "-3/25", "exact": True},
            )
        elif loss == "brier":
            c = h.task.classes
            a = str(F(c, c - 1))
            curvature = F(-9 * c, 256 * (c - 1))
            v = {
                "p": {
                    "lower": "0",
                    "lower_open": True,
                    "upper": "1",
                    "upper_open": True,
                }
            }
            add(
                "brier_multiclass_line",
                scalar(
                    f"((p-1)^2+{c-1}*((1-p)/{c-1})^2)/2", "==", f"({a})*(1-p)^2/2", v
                ),
            )
            add(
                "brier_logit_gradient_derivative",
                derivative(f"-({a})*p*(1-p)^2", "p", f"({a})*(1-p)*(3*p-1)", v),
            )
            add(
                "brier_chain_curvature",
                scalar(
                    f"p*(1-p)*({a})*(1-p)*(3*p-1)", "==", f"({a})*p*(1-p)^2*(3*p-1)", v
                ),
            )
            add(
                "brier_curvature_value",
                scalar(f"({a})*(1/4)*(1-1/4)^2*(3/4-1)", "==", curvature),
            )
            add("brier_negative_curvature", scalar(curvature, "<", "0"))
            objective.update(
                status="NONCONVEX",
                scheme="negative_logit_curvature_on_equal_other_class_line",
                witness={
                    "p": "1/4",
                    "true_logit": f"log({F(c-1,3)})",
                    "other_logits": "0",
                    "curvature": str(curvature),
                    "exact": True,
                },
            )
        elif loss == "focal":
            v = {
                "p": {
                    "lower": "0",
                    "lower_open": True,
                    "upper": "1",
                    "upper_open": True,
                }
            }
            add(
                "focal_probability_gradient",
                derivative("-(1-p)^2*log(p)", "p", "2*(1-p)*log(p)-(1-p)^2/p", v),
            )
            add(
                "focal_logit_gradient_derivative",
                derivative(
                    "2*p*(1-p)^2*log(p)-(1-p)^3",
                    "p",
                    "2*(1-p)*(1-3*p)*log(p)+5*(1-p)^2",
                    v,
                ),
            )
            add(
                "focal_chain_curvature",
                scalar(
                    "p*(1-p)*(2*(1-p)*(1-3*p)*log(p)+5*(1-p)^2)",
                    "==",
                    "p*(1-p)^2*((2-6*p)*log(p)+5*(1-p))",
                    v,
                ),
            )
            # log(2)>2/3 follows by integrating a strict tangent lower bound.
            # The analytic inference uses continuity and the fundamental theorem
            # of calculus; all algebra, derivative and endpoint facts are replayed.
            tv = {"t": {"lower": "1", "upper": "2"}}
            add("log2_tangent_lower_bound", scalar("1/t", ">=", "4/3-4*t/9", tv))
            add("log2_tangent_strict_point", scalar("1-(4/3-4/9)", ">", "0"))
            add("log2_integrand", derivative("log(t)", "t", "1/t", tv))
            add(
                "log2_tangent_primitive",
                derivative("4*t/3-2*t*t/9", "t", "4/3-4*t/9", tv),
            )
            add("log2_integral_value", scalar("(4*2/3-2*2*2/9)-(4/3-2/9)", "==", "2/3"))
            bound = F(-110515, 3145728)
            add(
                "focal_negative_curvature_bound",
                scalar(
                    "(1/32)*(31/32)^2*((29/16)*(-5*l)+155/32)",
                    "<=",
                    bound,
                    {"l": {"lower": "2/3"}},
                ),
            )
            add("focal_bound_negative", scalar(bound, "<", "0"))
            objective.update(
                status="NONCONVEX",
                scheme="log_integral_lower_bound_and_logit_chain_curvature",
                witness={
                    "p": "1/32",
                    "log2_lower_bound": "2/3",
                    "curvature_upper_bound": str(bound),
                    "exact": True,
                },
            )
        if any(c["status"] != "VERIFIED" for c in checks):
            objective["status"] = "UNRESOLVED"
        return objective, checks

    def _feedback(self, h, policy):
        algorithm = h.algorithm.to_dict()
        node = algorithm["controller"]
        checks = []
        possible = sorted(h.algorithm.controller.signals.intersection(FEEDBACK))
        result = {
            "status": "PENDING",
            "effect": effect_name(algorithm),
            "possible_training_signals": possible,
            "witnesses": [],
            "scope": "existential exact influence on a used coefficient over the independent signal box; not trajectory reachability",
        }
        variables = {s: {"lower": "0", "upper": "1"} for s in SIGNALS}
        expression = self.verifier.expr_text(self.verifier.controller_expression(node))
        center = controller_value(node, {s: F(1, 2) for s in SIGNALS})
        const = self._check(
            "feedback/constant_controller",
            scalar(expression, "==", str(center), variables),
        )
        if const["status"] == "VERIFIED":
            checks.append(const)
            result.update(status="REJECTED", reason="CONSTANT_CONTROLLER")
            return result, checks
        if len(possible) < policy.min_feedback_signals:
            result.update(status="REJECTED", reason="INSUFFICIENT_TRAINING_FEEDBACK")
            return result, checks
        grid = (F(0), F(1, 2), F(1))
        values = {}
        for point in product(grid, repeat=len(SIGNALS)):
            values[point] = controller_value(node, dict(zip(SIGNALS, point)))
        threshold = exact_number(policy.min_influence)
        for signal in possible:
            index = SIGNALS.index(signal)
            best = None
            for point in values:
                if point[index] != 0:
                    continue
                for low_value, high_value in (
                    (grid[0], grid[1]),
                    (grid[0], grid[2]),
                    (grid[1], grid[2]),
                ):
                    low = list(point)
                    low[index] = low_value
                    low = tuple(low)
                    high = list(point)
                    high[index] = high_value
                    high = tuple(high)
                    qlo, qhi = values[low], values[high]
                    delta = abs(effect(algorithm, qhi) - effect(algorithm, qlo))
                    if best is None or delta > best[0]:
                        best = (delta, low, high, qlo, qhi)
            if best and best[0] >= threshold:
                delta, low, high, qlo, qhi = best
                result["witnesses"].append(
                    {
                        "signal": signal,
                        "low": dict(zip(SIGNALS, map(str, low))),
                        "high": dict(zip(SIGNALS, map(str, high))),
                        "q_low": str(qlo),
                        "q_high": str(qhi),
                        "effect_low": str(effect(algorithm, qlo)),
                        "effect_high": str(effect(algorithm, qhi)),
                        "effect_delta": str(delta),
                    }
                )
        if len(result["witnesses"]) >= policy.min_feedback_signals:
            result["status"] = "PASS"
        else:
            result["reason"] = "INFLUENCE_WITNESS_UNRESOLVED"
        return result, checks

    def assess(self, record, runtime=None, policy=None):
        """Full-record integrity precedes any mathematical interpretation."""
        result = {
            "version": VERSION,
            "status": "REJECTED",
            "mathematical_status": "NOT_RUN",
            "reasons": [],
            "integrity": {"status": "REJECTED"},
            "math_checks": [],
            "objective": None,
            "feedback": None,
            "scientific_status": "NOT_EVALUATED",
            "sources": dict(self.sources),
        }
        try:
            if not self._sources_unchanged():
                raise ValueError("trusted source changed")
            normalized = bounded_payload(record)
            h = self.gen.Hypothesis.from_dict(normalized)
            # Equality of the complete regenerated payload is enforced by from_dict.
            if canonical(h.to_dict()) != canonical(normalized):
                raise ValueError("noncanonical record")
            runtime = Runtime.parse(runtime)
            policy = Policy.parse(policy)
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            OverflowError,
            RecursionError,
            OSError,
        ) as exc:
            result["reasons"] = ["RECORD_INTEGRITY"]
            result["integrity"]["detail"] = str(exc)
            return result
        key = fingerprint(normalized)
        issuance_key = fingerprint(
            {"record": key, "runtime": asdict(runtime), "policy": asdict(policy)}
        )
        issued = self._issued.get(issuance_key)
        result.update(
            record_hash=key,
            algorithm_signature=h.signature,
            runtime=asdict(runtime),
            policy=asdict(policy),
            integrity={
                "status": "PASS",
                "canonical_full_record": True,
                "origin": "SESSION_ISSUED" if issued else "UNATTESTED",
            },
            lineage=copy.deepcopy(issued),
            scientific_gates=copy.deepcopy(normalized["acceptance_gates"]),
        )
        base = self.checker.check(normalized)
        required = self._required_base(normalized)
        supplied = getattr(base, "subclaims", None)
        matches = False
        try:
            matches = isinstance(supplied, list) and [
                (x["obligation"], fingerprint(x["claim"])) for x in supplied
            ] == [(x["obligation"], fingerprint(x["claim"])) for x in required]
            matches = matches and all(
                isinstance(x["verdict"], dict)
                and all(k in x["verdict"] for k in ("status", "certificate", "reasons"))
                for x in supplied
            )
        except (ValueError, TypeError, KeyError):
            matches = False
        for i, item in enumerate(required):
            v = supplied[i]["verdict"] if matches else item["verdict"]
            result["math_checks"].append(
                {
                    "name": "base/" + item["obligation"],
                    "claim": item["claim"],
                    "status": v["status"],
                    "certificate": v["certificate"],
                    "reasons": v["reasons"],
                }
            )
        unresolved = []
        rejected = []
        if not matches or not required:
            unresolved.append("BASE_OBLIGATION_SET_MISMATCH")
        objective, checks = self._objective(h)
        result["objective"] = objective
        result["math_checks"].extend(checks)
        if objective["status"] == "NONCONVEX" and policy.require_convex_objective:
            rejected.append("NONCONVEX_OBJECTIVE_POLICY")
        elif objective["status"] == "UNRESOLVED":
            unresolved.append("OBJECTIVE_PROOF_UNRESOLVED")
        feedback, checks = self._feedback(h, policy)
        result["feedback"] = feedback
        result["math_checks"].extend(checks)
        if feedback["status"] == "REJECTED":
            rejected.append(feedback["reason"])
        elif feedback["status"] != "PASS":
            unresolved.append(feedback["reason"])
        for item in result["math_checks"]:
            if item["status"] != "VERIFIED":
                unresolved.append("MATHEMATICAL_OBLIGATION_UNRESOLVED")
            elif not self.verifier.verify_certificate(
                item["claim"], item["certificate"]
            ):
                unresolved.append("CERTIFICATE_REPLAY_FAILED")
        result["mathematical_status"] = (
            "FAIL" if rejected else "PENDING" if unresolved else "PASS"
        )
        if not issued:
            unresolved.append("UNATTESTED_ORIGIN")
        result["reasons"] = list(dict.fromkeys(rejected + unresolved))
        result["status"] = (
            "REJECTED"
            if rejected
            else "PENDING" if unresolved else "READY_FOR_SCIENTIFIC_VALIDATION"
        )
        result["receipt_hash"] = fingerprint(result)
        return result

    def verify_receipt(self, record, receipt, runtime=None, policy=None):
        """Check every binding, independently replay receipts, reconstruct decision.

        Reconstructing the eligibility decision is deliberately stricter than
        trusting a serialized READY field. Some gate searches are repeated; the
        underlying mathematical certificates use the verifier's search-free replay.
        """
        try:
            if (
                not isinstance(receipt, dict)
                or not isinstance(receipt.get("integrity"), dict)
                or receipt.get("integrity", {}).get("status") != "PASS"
            ):
                return False
            body = {k: v for k, v in receipt.items() if k != "receipt_hash"}
            if receipt.get("receipt_hash") != fingerprint(body):
                return False
            if runtime is not None and asdict(Runtime.parse(runtime)) != receipt.get(
                "runtime"
            ):
                return False
            if policy is not None and asdict(Policy.parse(policy)) != receipt.get(
                "policy"
            ):
                return False
            for item in receipt["math_checks"]:
                if item[
                    "status"
                ] == "VERIFIED" and not self.verifier.verify_certificate(
                    item["claim"], item["certificate"]
                ):
                    return False
            expected = self.assess(
                record, runtime=receipt["runtime"], policy=receipt["policy"]
            )
            return canonical(receipt) == canonical(expected)
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            OverflowError,
            RecursionError,
            OSError,
        ):
            return False

    def run(self, task, count=100, seed=42, runtime=None, policy=None):
        if type(count) is not int or not 1 <= count <= 10_000:
            raise ValueError("count in [1,10000] required")
        if len(self._issued) + count > self._issued_limit:
            raise ValueError("session ledger capacity reached; no silent eviction")
        runtime = Runtime.parse(runtime)
        policy = Policy.parse(policy)
        task = self.gen.TaskSpec.from_dict(bounded_payload(task))
        generator = self.gen.HypothesisGenerator(seed)
        seen = set()
        results = []
        for h in generator.iter_generate(task, count):
            record = bounded_payload(h.to_dict())
            key = fingerprint(record)
            if any(parent not in seen for parent in h.parents):
                raise ValueError(
                    "parent was not emitted earlier in this generation session"
                )
            issuance_key = fingerprint(
                {"record": key, "runtime": asdict(runtime), "policy": asdict(policy)}
            )
            self._issued[issuance_key] = {
                "seed": seed,
                "round_no": h.round_no,
                "parents": list(h.parents),
                "session_task_hash": fingerprint(asdict(task)),
            }
            seen.add(h.signature)
            audit = self.assess(record, runtime=runtime, policy=policy)
            results.append(
                {"candidate": record, "audit": audit, "status": audit["status"]}
            )
        return {
            "version": VERSION,
            "generated": len(results),
            "status_counts": dict(Counter(r["status"] for r in results)),
            "rejection_counts": dict(
                Counter(
                    reason
                    for r in results
                    if r["status"] == "REJECTED"
                    for reason in r["audit"]["reasons"]
                )
            ),
            "scientific_status": "NOT_EVALUATED",
            "results": results,
        }

    def handoff(self, record, receipt, runtime=None, policy=None):
        if (
            not isinstance(receipt, dict)
            or receipt.get("status") != "READY_FOR_SCIENTIFIC_VALIDATION"
            or not self.verify_receipt(record, receipt, runtime, policy)
        ):
            raise ValueError(
                "no replayed, session-issued mathematical admission receipt"
            )
        return {
            "candidate": bounded_payload(record),
            "math_admission": copy.deepcopy(receipt),
            "scientific_status": "NOT_EVALUATED",
            "handoff_hash": fingerprint({"candidate": record, "receipt": receipt}),
        }

    def build_model(self, record, receipt, runtime=None, policy=None):
        self.handoff(record, receipt, runtime, policy)
        h = self.gen.Hypothesis.from_dict(bounded_payload(record))
        r = Runtime.parse(receipt["runtime"])
        return self.gen.GeneratedModel(
            h, steps=r.steps, learning_rate=float(F(r.learning_rate)), seed=r.seed
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path)
    parser.add_argument("--verifier", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("assess")
    check.add_argument("input", type=Path)
    run = commands.add_parser("run")
    run.add_argument(
        "--kind", choices=("regression", "classification"), default="regression"
    )
    run.add_argument("--features", type=int, default=8)
    run.add_argument("--classes", type=int)
    run.add_argument("--count", type=int, default=100)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument(
        "--output",
        type=Path,
        help="optional full JSON audit artifact; contains no scientific measurements",
    )
    args = parser.parse_args()
    engine = MathFilterEngine(args.generator, args.verifier)
    if args.command == "assess":
        raw = args.input.read_bytes()
        if len(raw) > MAX_BYTES:
            raise SystemExit("record payload too large")
        print(json.dumps(engine.assess(read_json(raw.decode())), indent=2))
        return
    task = {"kind": args.kind, "features": args.features}
    if args.classes is not None:
        task["classes"] = args.classes
    batch = engine.run(task, count=args.count, seed=args.seed)
    if args.output:
        args.output.write_text(json.dumps(batch, indent=2))
    print(json.dumps({k: v for k, v in batch.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
