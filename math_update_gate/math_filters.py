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

VERSION = "math-update-admission-2.0"
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
    if type(value) not in (int, str) or len(str(value)) > 1000:
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
    require_convex_objective: bool = False
    min_feedback_signals: int = 2
    min_influence: str = "1/64"
    curvature_upper: str = "50"
    require_stable_envelope: bool = True

    @classmethod
    def parse(cls, value=None):
        if isinstance(value, cls):
            value = asdict(value)
        value = {} if value is None else value
        strict_fields(value, cls.__dataclass_fields__)
        body = {**asdict(cls()), **value}
        if type(body["require_convex_objective"]) is not bool:
            raise ValueError("require_convex_objective must be Boolean")
        if type(body["require_stable_envelope"]) is not bool:
            raise ValueError("require_stable_envelope must be Boolean")
        if (
            type(body["min_feedback_signals"]) is not int
            or not 1 <= body["min_feedback_signals"] <= 5
        ):
            raise ValueError("min_feedback_signals must be in [1,5]")
        influence = exact_number(body["min_influence"])
        if not 0 < influence <= 1:
            raise ValueError("min_influence must be in (0,1]")
        curvature = exact_number(body["curvature_upper"])
        if not 0 < curvature <= 10**6:
            raise ValueError("curvature_upper in (0,1000000] required")
        return cls(
            body["require_convex_objective"],
            body["min_feedback_signals"],
            str(influence),
            str(curvature),
            body["require_stable_envelope"],
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


RMS_EPS = F(1, 10**6)
CLIP_EPS = F(1, 10**12)


def coefficients(algorithm, q):
    """Exact mathematical counterpart of the pinned executor's six branches."""
    q2 = q if algorithm["coupling"] == "coupled" else F(1, 2)
    mechanism = algorithm["mechanism"]
    beta = F(9, 10)
    step = F(1)
    penalty = F(1, 1000)
    clip = F(1)
    if mechanism in ("clip_memory", "clip_step"):
        clip = F(1, 4) + F(3, 2) * (1 - q)
    if mechanism == "clip_memory":
        beta = F(19, 20) * (1 - q2)
    elif mechanism == "clip_step":
        step = F(1, 4) + F(3, 4) * q2
    elif mechanism == "memory_step":
        beta = F(19, 20) * q
        step = 1 - F(3, 4) * q2
    elif mechanism == "dual_memory":
        beta = F(1, 2)
        step = 1 - F(1, 2) * q2
    elif mechanism == "penalty_step":
        penalty *= F(1, 4) + F(3, 2) * q
        step = 1 / (1 + q2)
    elif mechanism == "variance_penalty":
        penalty *= F(1, 4) + F(3, 2) * q
    return {
        "q2": q2,
        "beta": beta,
        "step_multiplier": step,
        "penalty": penalty,
        "clip_multiplier": clip,
    }


def denominator_floor(algorithm, q):
    if algorithm["preconditioner"] == "identity":
        return F(1)
    if algorithm["mechanism"] == "variance_penalty":
        q2 = q if algorithm["coupling"] == "coupled" else F(1, 2)
        return 1 - q2 + q2 * RMS_EPS
    return RMS_EPS


def quadratic_minimum(coefficients, lower, upper):
    coeff = list(map(exact_number, coefficients))
    if not 1 <= len(coeff) <= 3:
        raise ValueError("degree at most two required")
    a, b = exact_number(lower), exact_number(upper)
    if a > b:
        raise ValueError("ordered finite interval required")
    coeff += [F(0)] * (3 - len(coeff))
    candidates = [a, b]
    if coeff[2] > 0:
        vertex = -coeff[1] / (2 * coeff[2])
        if a <= vertex <= b:
            candidates.append(vertex)
    value = lambda q: coeff[0] + coeff[1] * q + coeff[2] * q * q
    where = min(candidates, key=lambda q: (value(q), q))
    return {
        "argument": str(where),
        "minimum": str(value(where)),
        "coefficients": list(map(str, coeff)),
        "interval": [str(a), str(b)],
        "rule": "quadratic_endpoint_or_interior_vertex",
    }


def verify_quadratic_minimum(coefficients, lower, upper, certificate):
    try:
        return quadratic_minimum(coefficients, lower, upper) == certificate
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        return False


def bounded_fraction(value, up):
    value = F(value)
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 4096:
        raise ValueError("interval arithmetic bit budget exceeded")
    if value.denominator.bit_length() <= 512:
        return value
    scale = 1 << 96
    n = value.numerator * scale
    integer = -((-n) // value.denominator) if up else n // value.denominator
    return F(integer, scale)


@dataclass(frozen=True)
class Box:
    """Finite outward rational intervals. No floating comparison proves a sign."""

    lo: F
    hi: F

    def __post_init__(self):
        lo, hi = bounded_fraction(self.lo, False), bounded_fraction(self.hi, True)
        if lo > hi:
            raise ValueError("inverted interval")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @classmethod
    def point(cls, x):
        return cls(F(x), F(x))

    def __add__(self, other):
        other = as_box(other)
        return Box(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return Box(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -as_box(other)

    def __rsub__(self, other):
        return as_box(other) + -self

    def __mul__(self, other):
        other = as_box(other)
        v = [x * y for x in (self.lo, self.hi) for y in (other.lo, other.hi)]
        return Box(min(v), max(v))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = as_box(other)
        if other.lo <= 0 <= other.hi:
            raise ValueError("interval denominator contains zero")
        return self * Box(1 / other.hi, 1 / other.lo)

    def __rtruediv__(self, other):
        return as_box(other) / self

    def square(self):
        return Box(
            (
                F(0)
                if self.lo <= 0 <= self.hi
                else min(self.lo * self.lo, self.hi * self.hi)
            ),
            max(self.lo * self.lo, self.hi * self.hi),
        )

    def sqrt(self):
        if self.lo < 0:
            raise ValueError("negative square-root interval")
        return Box(sqrt_rational(self.lo)[0], sqrt_rational(self.hi)[1])

    def maximum(self, other):
        other = as_box(other)
        return Box(max(self.lo, other.lo), max(self.hi, other.hi))

    def minimum(self, other):
        other = as_box(other)
        return Box(min(self.lo, other.lo), min(self.hi, other.hi))

    def data(self):
        return {"lo": str(self.lo), "hi": str(self.hi)}


def as_box(x):
    return x if isinstance(x, Box) else Box.point(x)


def sqrt_rational(x):
    x = F(x)
    if x < 0:
        raise ValueError("negative radicand")
    n, d = math.isqrt(x.numerator), math.isqrt(x.denominator)
    if n * n == x.numerator and d * d == x.denominator:
        return F(n, d), F(n, d)
    scale = 1 << 96
    k = math.isqrt((x.numerator * scale * scale) // x.denominator)
    return F(k, scale), F(k + 1, scale)


def dot(a, b):
    return sum((as_box(x) * as_box(y) for x, y in zip(a, b)), Box.point(0))


def polynomial_from_values(function):
    f0, f1, f2 = (function(F(i)) for i in range(3))
    c2 = (f2 - 2 * f1 + f0) / 2
    return [f0, f1 - f0 - c2, c2]


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

    def _update_invariants(self, h):
        checks = []

        def add(name, claim):
            checks.append(self._check("update/" + name, claim))

        variables = {"b": {"lower": "0", "upper": "19/20"}, "f": {}, "g": {}}
        add(
            "ema_convex_square_identity",
            scalar(
                "b*f*f+(1-b)*g*g-(b*f+(1-b)*g)^2", "==", "b*(1-b)*(f-g)^2", variables
            ),
        )
        add(
            "ema_jensen_remainder_nonnegative",
            scalar("b*(1-b)*(f-g)^2", ">=", "0", variables),
        )
        v = {"b": {"lower": "0", "upper": "19/20"}, "v": {"lower": "0"}, "g": {}}
        add(
            "ema_rms_energy_slack_identity",
            scalar(
                "20*(19/20*v+1/20*g*g)-(20*b*v+(1-b)*g*g)", "==", "(19-20*b)*v+b*g*g", v
            ),
        )
        add(
            "ema_rms_energy_slack_nonnegative",
            scalar("(19-20*b)*v+b*(g*g)", ">=", "0", v),
        )
        add(
            "variance_nonnegative",
            scalar("19/20*v+1/20*(g*g)", ">=", "0", {"v": {"lower": "0"}, "g": {}}),
        )
        add(
            "dual_memory_convex_square",
            scalar(
                "(1-q)*f*f+q*s*s-((1-q)*f+q*s)^2",
                "==",
                "q*(1-q)*(f-s)^2",
                {"q": {"lower": "0", "upper": "1"}, "f": {}, "s": {}},
            ),
        )
        sv = {"s": {"lower": "0"}}
        add("rms_denominator_positive", scalar("s+1/1000000", ">=", "1/1000000", sv))
        add(
            "rms_direction_energy_slack_identity",
            scalar(
                "20*(s+1/1000000)^2-20*s*s", "==", "40*s/1000000+20/1000000000000", sv
            ),
        )
        add(
            "rms_direction_energy_slack",
            scalar("40*s/1000000+20/1000000000000", ">=", "0", sv),
        )
        add(
            "normalization_small_branch",
            scalar(
                "a*a*s",
                "<=",
                "a*a",
                {"a": {"lower": "0"}, "s": {"lower": "0", "upper": "1"}},
            ),
        )
        add(
            "normalization_large_branch",
            scalar("a*a*s/s", "==", "a*a", {"a": {"lower": "0"}, "s": {"lower": "1"}}),
        )
        # q2*(sqrt(v)+eps)+(1-q2) >= eps for every q2 in [0,1].
        add(
            "blended_denominator_floor",
            scalar(
                "(1-q)+q*(s+1/1000000)-1/1000000",
                "==",
                "(1-q)*(1-1/1000000)+q*s",
                {"q": {"lower": "0", "upper": "1"}, "s": {"lower": "0"}},
            ),
        )
        add(
            "blended_floor_nonnegative",
            scalar(
                "(1-q)*(1-1/1000000)+q*s",
                ">=",
                "0",
                {"q": {"lower": "0", "upper": "1"}, "s": {"lower": "0"}},
            ),
        )
        return {
            "status": (
                "PASS" if all(c["status"] == "VERIFIED" for c in checks) else "PENDING"
            ),
            "scope": "exact-real finite optimizer states, all varying in-range controls; initialization and induction explicitly required",
            "rms_memory_bound": (
                "m_j^2 <= 20*v_j from zero initialization"
                if h.algorithm.preconditioner == "rms"
                else "not applicable: identity preconditioner"
            ),
            "normalized_step_bound": "||theta_next-theta|| <= eta <= learning_rate",
            "ordinary_rms_direction_squared_per_coordinate_bound": (
                "20"
                if h.algorithm.preconditioner == "rms"
                and h.algorithm.mechanism != "variance_penalty"
                else None
            ),
            "convergence": "UNVERIFIED",
        }, checks

    def _frozen_envelope(self, h, runtime, policy):
        algorithm = h.algorithm.to_dict()
        rate = F(runtime.learning_rate)
        H = F(policy.curvature_upper)
        variables = {s: {"lower": "0", "upper": "1"} for s in SIGNALS}
        expression = self.verifier.controller_expression(algorithm["controller"])
        context = self.verifier.Context(variables, [], self.verifier.Budget())
        interval = self.verifier.interval(expression, context)
        lo, hi = interval.lo, interval.hi
        if lo is None or hi is None or lo < 0 or hi > 1:
            raise ValueError("controller range proof unavailable")
        text = self.verifier.expr_text(expression)
        checks = [
            self._check(
                "update/controller_range_lower", scalar(text, ">=", lo, variables)
            ),
            self._check(
                "update/controller_range_upper", scalar(text, "<=", hi, variables)
            ),
        ]
        dual = algorithm["mechanism"] == "dual_memory"

        def gap(q, candidate_rate=rate):
            c = coefficients(algorithm, q)
            d = denominator_floor(algorithm, q)
            if dual:
                value = 234 * d - candidate_rate * c["step_multiplier"] * H * (
                    39 - 36 * q
                )
            else:
                value = 2 * (1 + c["beta"]) * d - candidate_rate * c[
                    "step_multiplier"
                ] * H * (1 - c["beta"])
            if algorithm["mechanism"] == "penalty_step":
                value *= 1 + c["q2"]
            return value

        polynomial = polynomial_from_values(gap)
        # Check at a fourth rational point so a changed mapping cannot silently
        # be treated as quadratic by this compiler.
        if sum(c * F(3) ** i for i, c in enumerate(polynomial)) != gap(F(3)):
            raise ValueError("unsupported nonquadratic stability mapping")
        minimum = quadratic_minimum(list(map(str, polynomial)), str(lo), str(hi))
        boundary_q = (
            hi
            if algorithm["mechanism"]
            in ("clip_memory", "clip_step", "variance_penalty")
            else lo
        )
        boundary_c = coefficients(algorithm, boundary_q)
        boundary_d = denominator_floor(algorithm, boundary_q)
        cap = (
            234
            * boundary_d
            / (H * boundary_c["step_multiplier"] * (39 - 36 * boundary_q))
            if dual
            else 2
            * (1 + boundary_c["beta"])
            * boundary_d
            / (H * boundary_c["step_multiplier"] * (1 - boundary_c["beta"]))
        )
        boundary_polynomial = polynomial_from_values(lambda q: gap(q, cap))
        boundary_minimum = quadratic_minimum(
            list(map(str, boundary_polynomial)), str(lo), str(hi)
        )
        if F(boundary_minimum["minimum"]) != 0:
            raise ValueError("maximal rate boundary certificate failed")
        result = {
            "status": "PENDING",
            "model": "dual_memory_cubic" if dual else "single_memory_quadratic",
            "scope": "frozen symmetric-feedback positive-curvature modes only; no adaptive/switching or full-program convergence claim",
            "curvature_interval": {"lower": "0", "lower_open": True, "upper": str(H)},
            "zero_curvature_modes": "MARGINAL, not strict contraction",
            "negative_curvature_modes": "not covered",
            "controller_interval": [str(lo), str(hi)],
            "gap_polynomial": list(map(str, polynomial)),
            "minimum_certificate": minimum,
            "max_certifiable_rate": str(cap),
            "recommended_certified_rate": str(min(F(1), cap / 2)),
            "rate_boundary": {
                "strict": True,
                "q": str(boundary_q),
                "runtime_rate_limit": "1",
                "minimum_certificate": boundary_minimum,
                "scope": "uncapped strict threshold for this conservative controller interval and frozen envelope only",
            },
            "witness": None,
            "assumptions": [
                "full penalized aggregate feedback has a fixed symmetric derivative H within the stated envelope, including frozen lambda*P*theta",
                "controller, preconditioner, regularizer, step and inactive normalizer are frozen for the local mode analysis",
                "curvature_upper is a declared reference contract, not inferred from the loss or a dataset",
            ],
        }
        if F(minimum["minimum"]) > 0 and all(x["status"] == "VERIFIED" for x in checks):
            result["status"] = "PASS"
            return result, checks
        # A failing conservative interval is not itself a realizable controller
        # counterexample. Search exact controller inputs with schedule time bound.
        times = sorted({0, runtime.steps // 2, runtime.steps - 1})
        free = tuple(s for s in SIGNALS if s != "time")
        for t in times:
            if t >= runtime.steps:
                continue
            base = Box.point(rate) / (Box.point(1 + F(t, runtime.steps)).sqrt())
            for point in product((F(0), F(1, 2), F(1)), repeat=len(free)):
                signals = dict(zip(free, point))
                signals["time"] = F(t, max(runtime.steps - 1, 1))
                q = controller_value(algorithm["controller"], signals)
                c = coefficients(algorithm, q)
                d = denominator_floor(algorithm, q)
                gain = base * c["step_multiplier"] * H / d
                slack = (
                    234 - gain * (39 - 36 * q)
                    if dual
                    else 2 * (1 + c["beta"]) - gain * (1 - c["beta"])
                )
                if slack.hi <= 0:
                    result.update(
                        status="FAIL",
                        witness={
                            "t": t,
                            "signals": {s: str(signals[s]) for s in SIGNALS},
                            "q": str(q),
                            "curvature": str(H),
                            "denominator": str(d),
                            "base_step": base.data(),
                            "jury_slack": slack.data(),
                            "interpretation": "violates the declared frozen-mode envelope, not a proved divergent adaptive trajectory",
                        },
                    )
                    return result, checks
        return result, checks

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

    def certify_transition(self, record, state, runtime=None):
        """Outward-rational verification of the actual optimizer-core equations.

        Per-example gradients and non-time signals are supplied mathematical
        inputs, not authenticated observations or a proof of trajectory reachability.
        Descent is conditional on their correctness and the stated smoothness bound.
        """
        if not self._sources_unchanged():
            raise ValueError("trusted source changed")
        record = bounded_payload(record)
        h = self.gen.Hypothesis.from_dict(record)
        runtime = Runtime.parse(runtime)
        state = bounded_payload(state)
        strict_fields(
            state,
            (
                "t",
                "theta",
                "fast",
                "slow",
                "variance",
                "example_gradients",
                "signals",
                "smoothness_bound",
            ),
        )
        for field in (
            "t",
            "theta",
            "fast",
            "example_gradients",
            "signals",
            "smoothness_bound",
        ):
            if field not in state:
                raise ValueError("missing transition field: " + field)
        t = state["t"]
        if type(t) is not int or not 0 <= t < runtime.steps:
            raise ValueError("t outside admitted schedule")
        p = (h.algorithm.representation_size(h.task.features) + 1) * h.task.classes
        if p > 128:
            raise ValueError("explicit transition dimension budget exceeded")

        def vector(value):
            if not isinstance(value, list) or len(value) != p:
                raise ValueError("transition vector dimension mismatch")
            return list(map(exact_number, value))

        theta = vector(state["theta"])
        fast = vector(state["fast"])
        gradients = state["example_gradients"]
        if not isinstance(gradients, list) or not 2 <= len(gradients) <= 32:
            raise ValueError("2..32 per-example gradients required")
        gradients = [vector(row) for row in gradients]
        n = len(gradients)
        signals = state["signals"]
        if not isinstance(signals, dict) or set(signals) != set(SIGNALS):
            raise ValueError("all six signal inputs required")
        signals = {s: exact_number(v) for s, v in signals.items()}
        if any(v < 0 or v > 1 for v in signals.values()):
            raise ValueError("signal outside [0,1]")
        if signals["time"] != F(t, max(runtime.steps - 1, 1)):
            raise ValueError("time signal contradicts schedule")
        L = exact_number(state["smoothness_bound"])
        if L <= 0:
            raise ValueError("positive smoothness bound required")
        a = h.algorithm.to_dict()
        q = controller_value(a["controller"], signals)
        c = coefficients(a, q)
        if a["mechanism"] == "dual_memory" and "slow" not in state:
            raise ValueError("dual_memory requires slow state")
        if a["preconditioner"] == "rms" and "variance" not in state:
            raise ValueError("RMS requires variance state")
        slow = vector(state["slow"]) if a["mechanism"] == "dual_memory" else None
        if slow is None and "slow" in state:
            raise ValueError("slow state is absent from this mechanism")
        variance = vector(state["variance"]) if a["preconditioner"] == "rms" else None
        if variance is None and "variance" in state:
            raise ValueError("variance state is absent from identity preconditioner")
        if variance is not None and any(v < 0 for v in variance):
            raise ValueError("variance state must be nonnegative")
        norm_squared = [sum((g * g for g in row), F(0)) for row in gradients]
        norms = [Box.point(x).sqrt() for x in norm_squared]
        order = sorted(range(n), key=lambda i: norm_squared[i])
        position = F(3 * (n - 1), 4)
        lower = position.numerator // position.denominator
        weight = position - lower
        c0 = (
            (1 - weight) * norms[order[lower]]
            + weight * norms[order[min(lower + 1, n - 1)]]
        ).maximum(CLIP_EPS)
        threshold = c0 * c["clip_multiplier"]
        weights = [(threshold / (norm + CLIP_EPS)).minimum(1) for norm in norms]
        clipped = [[weight * x for x in row] for row, weight in zip(gradients, weights)]
        rng = self.gen.np.random.default_rng(runtime.seed)
        if a["feature_map"] == "linear_fourier":
            rng.normal(size=(h.task.features, min(h.task.features, 32)))
        blocks = [
            list(map(int, b))
            for b in self.gen.np.array_split(rng.permutation(n), min(5, n))
        ]

        def median(boxes):
            lower = sorted(b.lo for b in boxes)
            upper = sorted(b.hi for b in boxes)
            m = len(boxes)
            return (
                Box(lower[m // 2], upper[m // 2])
                if m % 2
                else Box(
                    (lower[m // 2 - 1] + lower[m // 2]) / 2,
                    (upper[m // 2 - 1] + upper[m // 2]) / 2,
                )
            )

        if a["aggregation"] == "mean":
            aggregate = [
                sum((row[j] for row in clipped), Box.point(0)) / n for j in range(p)
            ]
        else:
            means = [
                [
                    sum((clipped[i][j] for i in block), Box.point(0)) / len(block)
                    for j in range(p)
                ]
                for block in blocks
            ]
            aggregate = [median([row[j] for row in means]) for j in range(p)]
        mask = [int(j < p - h.task.classes) for j in range(p)]
        penalty = [c["penalty"] * x * m for x, m in zip(theta, mask)]
        grad = [g + v for g, v in zip(aggregate, penalty)]
        objective_gradient = [
            sum((row[j] for row in gradients), F(0)) / n + penalty[j] for j in range(p)
        ]
        fast_next = [
            c["beta"] * old + (1 - c["beta"]) * g for old, g in zip(fast, grad)
        ]
        slow_next = (
            [F(19, 20) * old + F(1, 20) * g for old, g in zip(slow, grad)]
            if slow is not None
            else None
        )
        momentum = (
            [(1 - q) * f + q * s for f, s in zip(fast_next, slow_next)]
            if slow_next is not None
            else fast_next
        )
        variance_next = (
            [F(19, 20) * old + F(1, 20) * g.square() for old, g in zip(variance, grad)]
            if variance is not None
            else None
        )
        if variance_next is None:
            denominators = [Box.point(1)] * p
        else:
            denominators = [v.sqrt() + RMS_EPS for v in variance_next]
            if a["mechanism"] == "variance_penalty":
                denominators = [(1 - c["q2"]) + c["q2"] * d for d in denominators]
        direction = [m / d for m, d in zip(momentum, denominators)]
        norm_direction = sum((d.square() for d in direction), Box.point(0)).sqrt()
        eta = (
            Box.point(F(runtime.learning_rate))
            * c["step_multiplier"]
            / Box.point(1 + F(t, runtime.steps)).sqrt()
        )
        alpha = eta / norm_direction.maximum(1)
        step = [-alpha * d for d in direction]
        alignment = dot(objective_gradient, direction)
        directional_change = dot(objective_gradient, step)
        squared_step = sum((s.square() for s in step), Box.point(0))
        upper = directional_change + F(L, 2) * squared_step
        status = (
            "PROVED_DESCENT"
            if upper.hi < 0
            else (
                "PROVED_NONDESCENT_DIRECTION"
                if directional_change.lo > 0
                else "INCONCLUSIVE"
            )
        )
        input_invariant = "NOT_APPLICABLE"
        if variance is not None:
            input_invariant = (
                "HOLDS"
                if all(f * f <= 20 * v for f, v in zip(fast, variance))
                and (
                    slow is None or all(s * s <= 20 * v for s, v in zip(slow, variance))
                )
                else "VIOLATED"
            )
        result = {
            "version": VERSION,
            "status": status,
            "scope": "supplied gradients and in-range controller inputs; exact-real optimizer core; conditional frozen-current-regularizer L-smooth objective",
            "record_hash": fingerprint(record),
            "state_hash": fingerprint(state),
            "runtime": asdict(runtime),
            "sources": dict(self.sources),
            "q": str(q),
            "coefficients": {k: str(v) for k, v in c.items()},
            "regularizer_mask": mask,
            "blocks": blocks,
            "clip_threshold": threshold.data(),
            "clip_weights": [w.data() for w in weights],
            "aggregate": [v.data() for v in aggregate],
            "objective_gradient": list(map(str, objective_gradient)),
            "fast_next": [v.data() for v in fast_next],
            "slow_next": None if slow_next is None else [v.data() for v in slow_next],
            "variance_next": (
                None if variance_next is None else [v.data() for v in variance_next]
            ),
            "denominator": [v.data() for v in denominators],
            "direction": [v.data() for v in direction],
            "eta": eta.data(),
            "step": [v.data() for v in step],
            "theta_next": [(s + x).data() for s, x in zip(step, theta)],
            "gradient_direction_alignment": alignment.data(),
            "directional_change": directional_change.data(),
            "smoothness_change_bound": upper.data(),
            "step_norm_upper_bound": str(eta.hi),
            "input_rms_energy_invariant": input_invariant,
            "assumptions": [
                "supplied per-example arrays are the true current data-fit gradients",
                "non-time signals are supplied in-range controls; their data derivation is not verified here",
                "smoothness_bound covers the objective along the entire proposed step, with current lambda held fixed",
                "supplied states are not certified reachable from the executor initialization",
            ],
            "full_algorithm_convergence": "UNVERIFIED",
            "scientific_status": "NOT_EVALUATED",
        }
        result["receipt_hash"] = fingerprint(result)
        return result

    def verify_transition(self, record, state, receipt, runtime=None):
        try:
            if not isinstance(receipt, dict):
                return False
            body = {k: v for k, v in receipt.items() if k != "receipt_hash"}
            if receipt.get("receipt_hash") != fingerprint(body):
                return False
            if runtime is not None and asdict(Runtime.parse(runtime)) != receipt.get(
                "runtime"
            ):
                return False
            return canonical(receipt) == canonical(
                self.certify_transition(record, state, runtime=receipt["runtime"])
            )
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
            "update": None,
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
        invariants, checks = self._update_invariants(h)
        result["math_checks"].extend(checks)
        envelope, checks = self._frozen_envelope(h, runtime, policy)
        result["math_checks"].extend(checks)
        result["update"] = {
            "invariants": invariants,
            "envelope": envelope,
            "full_algorithm_convergence": "UNVERIFIED",
        }
        if invariants["status"] != "PASS":
            unresolved.append("UPDATE_INVARIANTS_UNRESOLVED")
        if policy.require_stable_envelope:
            if envelope["status"] == "FAIL":
                rejected.append("FROZEN_UPDATE_ENVELOPE_VIOLATION")
            elif envelope["status"] != "PASS":
                unresolved.append("UPDATE_ENVELOPE_UNRESOLVED")
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
