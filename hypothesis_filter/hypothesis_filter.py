"""Five data-free research-priority gates before mathematical investigation.

PASS means eligibility within an explicit finite policy and checked prior-art
scope, never scientific novelty, correctness or predictive superiority.
No dataset, learner training, experimental outcome or scientific artifact API.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass
from fractions import Fraction as F
import hashlib
import importlib.util
from itertools import product
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

VERSION = "hypothesis-filter-1.0"
GENERATOR_SHA256 = "8ce3010e741603c9627c85f37944586c4cde2155eeb24660adcfd58ca5931b3f"
SIGNALS = ("disagreement", "dispersion", "alignment", "progress", "time", "energy")
FEEDBACK = tuple(x for x in SIGNALS if x != "time")
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
LAYERS = (
    "coherence",
    "uniqueness",
    "prior_art_screen",
    "technical_potential",
    "research_readiness",
)
MAX_TERMS = 2048


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SOURCE_SHA256 = file_hash(__file__)


def strict(value, fields, required=()):
    if (
        type(value) is not dict
        or set(value) - set(fields)
        or set(required) - set(value)
    ):
        raise ValueError("unknown/missing schema fields")


def bounded_json(value):
    pending = [(value, 0)]
    nodes = 0
    characters = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if depth > 32 or nodes > 30000:
            raise ValueError("record structure limit exceeded")
        if type(item) is dict:
            if any(type(k) is not str for k in item):
                raise ValueError("string object keys required")
            pending.extend((v, depth + 1) for v in item.values())
            characters += sum(len(k) for k in item)
        elif type(item) in (list, tuple):
            pending.extend((v, depth + 1) for v in item)
        elif type(item) is str:
            characters += len(item)
        elif item is None or type(item) in (bool, int):
            if type(item) is int and item.bit_length() > 256:
                raise ValueError("integer limit exceeded")
        elif type(item) is float:
            import math

            if not math.isfinite(item):
                raise ValueError("nonfinite values forbidden")
        else:
            raise ValueError("JSON values required")
        if characters > 250000:
            raise ValueError("record text limit exceeded")
    return json.loads(canonical(value))


def read_json(text):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate JSON key")
            out[key] = value
        return out

    def invalid(_):
        raise ValueError("nonfinite JSON forbidden")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def load_generator(path=None):
    path = Path(
        path
        or os.environ.get("HYPOTHESIS_GENERATOR")
        or Path(__file__).resolve().parent.parent
        / "generator_only"
        / "hypothesis_generator.py"
    ).resolve()
    if file_hash(path) != GENERATOR_SHA256:
        raise ValueError("unreviewed generator source")
    name = "_hypothesis_filter_generator"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REFERENCES = (
    {
        "id": "adam",
        "title": "Adam: A Method for Stochastic Optimization",
        "url": "https://arxiv.org/abs/1412.6980",
        "year": 2014,
        "overlap": "momentum and second-moment adaptive scaling; these are established components",
        "domains": ["optimization"],
    },
    {
        "id": "hypergradient",
        "title": "Online Learning Rate Adaptation with Hypergradient Descent",
        "url": "https://arxiv.org/abs/1703.04782",
        "year": 2017,
        "overlap": "gradient-based online learning-rate adaptation is established; formula-level identity is not asserted",
        "domains": ["optimization", "control_theory"],
    },
    {
        "id": "yellowfin",
        "title": "YellowFin and the Art of Momentum Tuning",
        "url": "https://arxiv.org/abs/1706.03471",
        "year": 2017,
        "overlap": "joint momentum and learning-rate adaptation and negative feedback predate this generator",
        "domains": ["optimization", "control_theory"],
    },
    {
        "id": "agc",
        "title": "High-Performance Large-Scale Image Recognition Without Normalization",
        "url": "https://arxiv.org/abs/2102.06171",
        "year": 2021,
        "overlap": "adaptive gradient clipping is established; this work uses a different stated clipping rule",
        "domains": ["optimization", "robust_statistics"],
    },
    {
        "id": "robust",
        "title": "Mean estimation and regression under heavy-tailed distributions—a survey",
        "url": "https://arxiv.org/abs/1906.04280",
        "year": 2019,
        "overlap": "median-of-means and heavy-tail robust estimation are established",
        "domains": ["robust_statistics", "probability"],
    },
    {
        "id": "random_features",
        "title": "Random Features for Large-Scale Kernel Machines",
        "url": "https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html",
        "year": 2007,
        "overlap": "random Fourier features are established; their presence alone supplies no novelty",
        "domains": ["harmonic_analysis", "randomized_methods"],
    },
)
REGISTRY = {
    "version": "mechanism-index-1",
    "scope": "six selected primary works; existing optimizer, feedback, clipping, robust estimation and random-feature components; not exhaustive literature coverage",
    "references": list(REFERENCES),
    "novelty_authority": "NONE; absence of an exact registered match is not proof of novelty",
}
REGISTRY_SHA256 = digest(REGISTRY)


@dataclass(frozen=True)
class Policy:
    min_feedback: int = 2
    min_domains: int = 2
    pareto: bool = True
    reject_parameter_variants: bool = True

    @classmethod
    def parse(cls, value=None):
        if isinstance(value, cls):
            value = asdict(value)
        value = {} if value is None else value
        strict(value, cls.__dataclass_fields__)
        body = {**asdict(cls()), **value}
        for key, low, high in [
            ("min_feedback", 1, 5),
            ("min_domains", 1, 8),
        ]:
            if type(body[key]) is not int or not low <= body[key] <= high:
                raise ValueError("strict bounded integer policy required")
        for key in ("pareto", "reject_parameter_variants"):
            if type(body[key]) is not bool:
                raise ValueError("strict Boolean policy required")
        return cls(**body)


# Sparse exact polynomial algebra. min() atoms retain explicit definitions.
# max(u,v)=u+v-min(u,v); rational ring operations remove canceled atoms.
def pc(value):
    return {} if not value else {(): F(value)}


def pa(a, b, scale=F(1)):
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, F(0)) + scale * v
        if not result[k]:
            del result[k]
    if len(result) > MAX_TERMS:
        raise ValueError("normal form term budget exceeded")
    return result


def ps(a, scale):
    return {k: v * scale for k, v in a.items() if v * scale}


def pm(a, b):
    if len(a) * len(b) > MAX_TERMS * 4:
        raise ValueError("polynomial product budget exceeded")
    result = {}
    for left, x in a.items():
        for right, y in b.items():
            key = tuple(sorted(left + right))
            if len(key) > 16:
                raise ValueError("degree budget exceeded")
            result[key] = result.get(key, F(0)) + x * y
    result = {k: v for k, v in result.items() if v}
    if len(result) > MAX_TERMS:
        raise ValueError("normal form term budget exceeded")
    return result


def wire(poly):
    return [[list(k), str(v)] for k, v in sorted(poly.items())]


def unwire(terms):
    return {tuple(k): F(v) for k, v in terms}


def normalize_controller(controller):
    atoms = {}
    count = 0
    arities = {
        "signal": 0,
        "half": 0,
        "complement": 1,
        "mean": 2,
        "product": 2,
        "minimum": 2,
        "maximum": 2,
    }

    def validate(node, depth):
        nonlocal count
        count += 1
        if depth > 4 or count > 31:
            raise ValueError("controller structure budget exceeded")
        strict(node, ("op", "name", "children"), ("op", "name", "children"))
        op = node["op"]
        children = node["children"]
        if (
            type(op) is not str
            or op not in arities
            or type(children) is not list
            or len(children) != arities[op]
        ):
            raise ValueError("unsupported controller")
        if (op == "signal" and node["name"] not in SIGNALS) or (
            op != "signal" and node["name"] != ""
        ):
            raise ValueError("invalid controller name")
        for child in children:
            validate(child, depth + 1)

    validate(controller, 0)

    def push_complement(node, negative=False):
        op = node["op"]
        if op == "complement":
            return push_complement(node["children"][0], not negative)
        if op in ("minimum", "maximum"):
            mapped = ("maximum" if op == "minimum" else "minimum") if negative else op
            return {
                "op": mapped,
                "name": "",
                "children": [push_complement(c, negative) for c in node["children"]],
            }
        value = {
            "op": op,
            "name": node["name"],
            "children": [push_complement(c) for c in node["children"]],
        }
        return (
            {"op": "complement", "name": "", "children": [value]}
            if negative and op != "half"
            else value
        )

    controller = push_complement(controller)

    def pair(a, ab, b, bb, op):
        if a == b:
            return a, ab
        if ab[1] <= bb[0]:
            return (a, ab) if op == "minimum" else (b, bb)
        if bb[1] <= ab[0]:
            return (b, bb) if op == "minimum" else (a, ab)
        definition = sorted((wire(a), wire(b)), key=canonical)
        key = "m:" + digest(definition)
        atoms[key] = definition
        minimum = {(key,): F(1)}
        result = minimum if op == "minimum" else pa(pa(a, b), minimum, F(-1))
        bounds = (
            (min(ab[0], bb[0]), min(ab[1], bb[1]))
            if op == "minimum"
            else (max(ab[0], bb[0]), max(ab[1], bb[1]))
        )
        return result, bounds

    def visit(node):
        op = node["op"]
        if op == "signal":
            return {("s:" + node["name"],): F(1)}, (F(0), F(1))
        if op == "half":
            return pc(F(1, 2)), (F(1, 2), F(1, 2))
        if op in ("minimum", "maximum"):
            # Canonical max-of-mins antichain handles lattice associativity,
            # commutativity, idempotence, absorption and distributivity exactly.
            leaves = {}

            def prune(terms):
                if len(terms) > 512:
                    raise ValueError("lattice expansion budget exceeded")
                ordered = sorted(set(terms), key=lambda x: (len(x), sorted(x)))
                kept = []
                for term in ordered:
                    if not any(old <= term for old in kept):
                        kept.append(term)
                return kept

            def collect(current):
                if current["op"] not in ("minimum", "maximum"):
                    value, bounds = visit(current)
                    key = canonical(wire(value))
                    leaves[key] = (value, bounds)
                    return [frozenset((key,))]
                a = collect(current["children"][0])
                b = collect(current["children"][1])
                if current["op"] == "maximum":
                    return prune(a + b)
                if len(a) * len(b) > 4096:
                    raise ValueError("lattice product budget exceeded")
                return prune([x | y for x in a for y in b])

            terms = collect(node)
            lowered = []
            for term in sorted(terms, key=lambda x: sorted(x)):
                pieces = [leaves[k] for k in sorted(term)]
                value, bounds = pieces[0]
                for other, other_bounds in pieces[1:]:
                    value, bounds = pair(value, bounds, other, other_bounds, "minimum")
                lowered.append((value, bounds))
            value, bounds = lowered[0]
            for other, other_bounds in lowered[1:]:
                value, bounds = pair(value, bounds, other, other_bounds, "maximum")
            return value, bounds
        a, ab = visit(node["children"][0])
        if op == "complement":
            value = pa(pc(1), a, F(-1))
            bounds = (1 - ab[1], 1 - ab[0])
        else:
            b, bb = visit(node["children"][1])
            if op == "mean":
                value = ps(pa(a, b), F(1, 2))
                bounds = ((ab[0] + bb[0]) / 2, (ab[1] + bb[1]) / 2)
            else:
                value = pm(a, b)
                bounds = (ab[0] * bb[0], ab[1] * bb[1])
        if not value or set(value) == {()}:
            bounds = (value.get((), F(0)),) * 2
        return value, bounds

    poly, _ = visit(controller)
    needed = set()

    def gather(p):
        for monomial in p:
            for variable in monomial:
                if variable.startswith("m:") and variable not in needed:
                    needed.add(variable)
                    for terms in atoms[variable]:
                        gather(unwire(terms))

    gather(poly)
    return {
        "terms": wire(poly),
        "min_atoms": {k: atoms[k] for k in sorted(needed)},
        "assumption": "each named signal lies in [0,1]",
    }


def polynomial_family(normal):
    # Numeric coefficients are hyperparameters inside this declared template.
    # This is a prioritization family, not an assertion of functional equality.
    return {
        "terms": [
            [powers, 1 if F(coefficient) > 0 else -1]
            for powers, coefficient in normal["terms"]
        ],
        "min_atoms": normal["min_atoms"],
    }


def update_description(algorithm, normal):
    q = unwire(normal["terms"])
    r = q if algorithm["coupling"] == "coupled" else pc(F(1, 2))
    d = {
        "clip": pc(1),
        "momentum": pc(F(9, 10)),
        "step_numerator": pc(1),
        "step_denominator": pc(1),
        "penalty": pc(F(1, 1000)),
        "memory_mix": pc(0),
        "memory_mode": "single",
        "denominator": algorithm["preconditioner"],
    }
    m = algorithm["mechanism"]
    if m in ("clip_memory", "clip_step"):
        d["clip"] = pa(pc(F(7, 4)), q, F(-3, 2))
    if m == "clip_memory":
        d["momentum"] = ps(pa(pc(1), r, F(-1)), F(19, 20))
    if m == "clip_step":
        d["step_numerator"] = pa(pc(F(1, 4)), r, F(3, 4))
    if m == "memory_step":
        d["momentum"] = ps(q, F(19, 20))
        d["step_numerator"] = pa(pc(1), r, F(-3, 4))
    if m == "dual_memory":
        d["momentum"] = pc(F(1, 2))
        d["memory_mix"] = q
        d["memory_mode"] = "fast=.5,slow=.95"
        d["step_numerator"] = pa(pc(1), r, F(-1, 2))
    if m in ("penalty_step", "variance_penalty"):
        d["penalty"] = pa(pc(F(1, 4000)), q, F(3, 2000))
    if m == "penalty_step":
        d["step_denominator"] = pa(pc(1), r)
    if m == "variance_penalty":
        d["denominator"] = "affine_identity_rms"
        d["rms_mixture"] = r
    return {k: wire(v) if isinstance(v, dict) else v for k, v in d.items()}


def identities(record, normal):
    a = record["algorithm"]
    parts = {k: a[k] for k in ("kind", "feature_map", "scaling", "loss", "aggregation")}
    update = update_description(a, normal)
    semantic = {"parts": parts, "update": update, "min_atoms": normal["min_atoms"]}
    family = {
        "parts": {
            **parts,
            "mechanism": a["mechanism"],
            "preconditioner": a["preconditioner"],
            "coupling": a["coupling"],
        },
        "controller": polynomial_family(normal),
    }
    return digest(semantic), digest(family), semantic


def evaluate_exact(node, signals):
    op = node["op"]
    if op == "signal":
        return signals[node["name"]]
    if op == "half":
        return F(1, 2)
    values = [evaluate_exact(c, signals) for c in node["children"]]
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
    raise ValueError("unsupported exact evaluation")


def controls_at(algorithm, q):
    r = q if algorithm["coupling"] == "coupled" else F(1, 2)
    out = {
        "clip": F(1),
        "momentum": F(9, 10),
        "step": F(1),
        "penalty": F(1, 1000),
        "memory_mix": F(0),
        "rms_mix": F(0),
    }
    m = algorithm["mechanism"]
    if m in ("clip_memory", "clip_step"):
        out["clip"] = F(7, 4) - F(3, 2) * q
    if m == "clip_memory":
        out["momentum"] = F(19, 20) * (1 - r)
    if m == "clip_step":
        out["step"] = F(1, 4) + F(3, 4) * r
    if m == "memory_step":
        out["momentum"] = F(19, 20) * q
        out["step"] = 1 - F(3, 4) * r
    if m == "dual_memory":
        out["momentum"] = F(1, 2)
        out["memory_mix"] = q
        out["step"] = 1 - r / 2
    if m in ("penalty_step", "variance_penalty"):
        out["penalty"] = F(1, 4000) + F(3, 2000) * q
    if m == "penalty_step":
        out["step"] = 1 / (1 + r)
    if m == "variance_penalty":
        out["rms_mix"] = r
    return out


def behavioral_profile(record, normal):
    """Finite exact construction probes; equality warrants HOLD, never proof."""
    a = record["algorithm"]
    points = []
    for anchor in (F(0), F(1, 2), F(1)):
        points.append({s: anchor for s in SIGNALS})
    for name, value in product(SIGNALS, (F(0), F(1))):
        point = {s: F(1, 2) for s in SIGNALS}
        point[name] = value
        points.append(point)
    for index in range(12):
        points.append(
            {
                name: F(
                    int(
                        hashlib.sha256(
                            f"public-signal-profile-v1:{index}:{name}".encode()
                        ).hexdigest()[:8],
                        16,
                    )
                    % 101,
                    100,
                )
                for name in SIGNALS
            }
        )
    values = [
        {
            k: str(v)
            for k, v in controls_at(a, evaluate_exact(a["controller"], point)).items()
        }
        for point in points
    ]
    update = update_description(a, normal)
    return digest(
        {
            "scope": "27 fixed public signal points, not training or scientific observations",
            "parts": {
                k: a[k]
                for k in ("kind", "feature_map", "scaling", "loss", "aggregation")
            },
            "memory_mode": update["memory_mode"],
            "denominator": update["denominator"],
            "responses": values,
        }
    )


def potential(record):
    a = record["algorithm"]
    witnesses = {}
    channels = set()
    for name in FEEDBACK:
        for anchor, low, high in product(
            (F(0), F(1, 2), F(1)), (F(0), F(1, 2)), (F(1, 2), F(1))
        ):
            if low >= high:
                continue
            left = {s: anchor for s in SIGNALS}
            right = dict(left)
            left[name] = low
            right[name] = high
            q1 = evaluate_exact(a["controller"], left)
            q2 = evaluate_exact(a["controller"], right)
            u = controls_at(a, q1)
            v = controls_at(a, q2)
            changed = sorted(k for k in u if u[k] != v[k])
            if changed:
                witnesses[name] = {
                    "left": {s: str(x) for s, x in left.items()},
                    "right": {s: str(x) for s, x in right.items()},
                    "controller_values": [str(q1), str(q2)],
                    "changed_controls": changed,
                }
                channels.update(changed)
                break

    def nodes(c):
        return 1 + sum(nodes(x) for x in c["children"])

    return {
        "feedback_witnesses": witnesses,
        "changed_controls": sorted(channels),
        "controller_nodes": nodes(a["controller"]),
        "stored_parameters": record["resources"]["stored_parameters"],
        "scope": "exact sensitivity on public synthetic signal points conditional on [0,1]; joint reachability in training is unproved",
        "quality_estimate": "NONE; no accuracy, convergence or novelty probability is estimated",
    }


def dossier(record, normal):
    a = record["algorithm"]
    domains = {"optimization", "robust_statistics"}
    if any(
        token.startswith("s:") or token.startswith("m:")
        for powers, _ in normal["terms"]
        for token in powers
    ):
        domains.add("control_theory")
    if a["aggregation"] == "median_of_means":
        domains.add("probability")
    if a["feature_map"] == "linear_fourier":
        domains.update(("harmonic_analysis", "randomized_methods"))
    if a["feature_map"] == "diagonal_quadratic":
        domains.add("algebra")
    if a["loss"] == "cross_entropy":
        domains.add("information_theory")
    refs = [
        r
        for r in REFERENCES
        if r["id"] != "random_features" or a["feature_map"] == "linear_fourier"
    ]
    return {
        "domains": sorted(domains),
        "domain_basis": "inferred from executable components; labels do not establish novelty",
        "prior_art": [
            {
                "id": r["id"],
                "url": r["url"],
                "title": r["title"],
                "overlap": r["overlap"],
                "relation": "component/design overlap, not an asserted full-algorithm equivalence",
            }
            for r in refs
        ],
        "registry_sha256": REGISTRY_SHA256,
        "coverage_scope": REGISTRY["scope"],
        "distinction_under_investigation": {
            "controller_normal_form": normal,
            "effective_controls": update_description(a, normal),
        },
        "search_queries": [
            f"{a['mechanism'].replace('_',' ')} joint adaptive clipping momentum learning rate training feedback",
            f"{a['loss'].replace('_',' ')} {a['feature_map'].replace('_',' ')} {a['aggregation'].replace('_',' ')} optimizer",
        ],
        "novelty_status": "UNVERIFIED",
        "screening_claim": "screening not yet completed",
    }


def layer(name, status, reason, **details):
    return {"name": name, "status": status, "reason": reason, **details}


def finalize(decision):
    statuses = [x["status"] for x in decision["layers"]]
    decision["eligible_for_math"] = statuses == ["PASS"] * 5
    decision["status"] = (
        "READY_FOR_MATH_INVESTIGATION"
        if decision["eligible_for_math"]
        else "HELD" if "HOLD" in statuses else "REJECTED"
    )
    return decision


def pareto_dominates(a, b):
    av = (
        len(a["potential"]["feedback_witnesses"]),
        len(a["potential"]["changed_controls"]),
        -a["potential"]["controller_nodes"],
        -a["potential"]["stored_parameters"],
    )
    bv = (
        len(b["potential"]["feedback_witnesses"]),
        len(b["potential"]["changed_controls"]),
        -b["potential"]["controller_nodes"],
        -b["potential"]["stored_parameters"],
    )
    return all(x >= y for x, y in zip(av, bv)) and any(x > y for x, y in zip(av, bv))


def prepare_inputs(records, gen):
    prepared = []
    for record in records:
        cid = digest(record)
        if record.get("schema_version") != gen.VERSION:
            code = "UNSUPPORTED_REPRESENTATION"
        else:
            try:
                gen.Hypothesis.from_dict(record)
                code = "CANONICAL"
            except (ValueError, TypeError, KeyError, IndexError, RecursionError):
                code = "NONCANONICAL_RECORD"
        prepared.append(
            {
                "input_digest": cid,
                "record": record if code == "CANONICAL" else None,
                "coherence_code": code,
            }
        )
    return prepared


def evaluate_records(records, gen, policy, before_seen, before_families, known):
    decisions = []
    seen = dict(before_seen)
    families = dict(before_families)
    # Canonical hashes make tie resolution independent of input ordering.
    order = sorted(range(len(records)), key=lambda i: (records[i]["input_digest"], i))
    for index in order:
        intake = records[index]
        record = intake["record"]
        cid = intake["input_digest"]
        code = intake["coherence_code"]
        if (
            type(cid) is not str
            or len(cid) != 64
            or any(c not in "0123456789abcdef" for c in cid)
        ):
            raise ValueError("invalid input digest")
        if code not in (
            "CANONICAL",
            "NONCANONICAL_RECORD",
            "UNSUPPORTED_REPRESENTATION",
        ):
            raise ValueError("invalid coherence outcome")
        if (
            code == "CANONICAL" and (type(record) is not dict or digest(record) != cid)
        ) or (code != "CANONICAL" and record is not None):
            raise ValueError("intake identity mismatch")
        d = {
            "index": index,
            "candidate_id": cid,
            "record": record,
            "coherence_code": code,
            "novelty_status": "UNVERIFIED",
            "scientific_gates": {g: "UNVERIFIED" for g in GATES},
            "layers": [
                layer(n, "SKIPPED", "earlier layer did not pass") for n in LAYERS
            ],
        }
        decisions.append(d)
        if code == "UNSUPPORTED_REPRESENTATION":
            d["layers"][0] = layer(
                LAYERS[0],
                "HOLD",
                "unsupported representation; payload discarded, only its digest retained",
            )
            continue
        if code == "NONCANONICAL_RECORD":
            d["layers"][0] = layer(
                LAYERS[0],
                "REJECT",
                "noncanonical or tainted record; payload discarded, only its digest retained",
            )
            continue
        gen.Hypothesis.from_dict(record)
        d["layers"][0] = layer(
            LAYERS[0],
            "PASS",
            "canonical executable specification, budgets, assumptions, ablations and falsification are present",
            origin="canonical consistency, not authenticated generator issuance",
        )
        try:
            normal = normalize_controller(record["algorithm"]["controller"])
            semantic, family, description = identities(record, normal)
        except ValueError as exc:
            d["layers"][1] = layer(
                LAYERS[1], "HOLD", "unsupported symbolic normalization: " + str(exc)
            )
            continue
        d.update(
            semantic_key=semantic,
            family_key=family,
            normal_form=normal,
            effective_update=description,
        )
        if semantic in seen:
            d["layers"][1] = layer(
                LAYERS[1],
                "REJECT",
                "exact normalized algorithm already registered",
                duplicate_of=seen[semantic],
            )
            continue
        seen[semantic] = cid
        profile = "profile:" + behavioral_profile(record, normal)
        d["profile_key"] = profile
        if profile in seen:
            d["layers"][1] = layer(
                LAYERS[1],
                "HOLD",
                "same finite public response profile as a previous candidate; equivalence unresolved, no mathematical identity asserted",
                possible_alias_of=seen[profile],
            )
            continue
        seen[profile] = cid
        d["layers"][1] = layer(
            LAYERS[1],
            "PASS",
            "distinct exact normal form within this persistent journal",
            proof_scope="sparse rational algebra with explicit min atoms, conditional on the signal cube",
        )
        d["dossier"] = dossier(record, normal)
        if semantic in known:
            d["dossier"][
                "screening_claim"
            ] = "caller-registered exclusion matched; no independent source-equivalence assertion"
            d["layers"][2] = layer(
                LAYERS[2],
                "REJECT",
                "matches a caller-registered known-implementation exclusion",
                reference=known[semantic],
            )
            continue
        if not normal["min_atoms"] and all(not powers for powers, _ in normal["terms"]):
            d["dossier"][
                "screening_claim"
            ] = "fixed-parameter controller excluded; no novelty established"
            d["layers"][2] = layer(
                LAYERS[2],
                "REJECT",
                "constant controller: only a fixed-parameter template, no new adaptive mechanism",
            )
            continue
        if family in families:
            d["dossier"]["parameter_variant_of"] = families[family]
        if policy.reject_parameter_variants and family in families:
            d["dossier"][
                "screening_claim"
            ] = "local coefficient-template variant excluded by the declared research policy"
            d["layers"][2] = layer(
                LAYERS[2],
                "REJECT",
                "coefficient-only variant within a previously seen template; policy excludes tuning-only novelty",
                variant_of=families[family],
            )
            continue
        families.setdefault(family, cid)
        d["dossier"][
            "screening_claim"
        ] = "no registered exact match; checked component overlaps and possible coefficient variants recorded; novelty remains unverified"
        d["layers"][2] = layer(
            LAYERS[2],
            "PASS",
            "no registered exact match; coefficient-template policy applied and selected prior-art overlaps/search gaps recorded",
            novelty_status="UNVERIFIED",
            scope=REGISTRY["scope"],
        )
        d["potential"] = potential(record)
        if len(d["potential"]["feedback_witnesses"]) < policy.min_feedback:
            d["layers"][3] = layer(
                LAYERS[3],
                "HOLD",
                "insufficient exact witnesses of distinct training-feedback influences; no performance inference",
            )
            continue
        if len(d["dossier"]["domains"]) < policy.min_domains:
            d["layers"][3] = layer(
                LAYERS[3],
                "HOLD",
                "insufficient executable cross-domain mechanisms under this policy",
            )
            continue
        d["layers"][3] = layer(
            LAYERS[3],
            "PASS",
            "conditional feedback witnesses and executable domain mechanisms meet the research policy; predictive merit unverified",
        )
        d["layers"][4] = layer(
            LAYERS[4],
            "PASS",
            (
                "falsifiable bounded candidate; Pareto priority applied"
                if policy.pareto
                else "falsifiable bounded candidate; Pareto priority disabled"
            ),
            priority_meaning="data-free research allocation, not a prediction of superiority",
        )
    # Compare only like scientific specifications; different losses, maps or
    # optimizer blueprints retain separate research opportunities.
    finalists = [d for d in decisions if d["layers"][3]["status"] == "PASS"]
    for d in finalists:
        record = d["record"]
        a = record["algorithm"]
        context = canonical(
            {
                "task": record["task"],
                "blueprint": {k: v for k, v in a.items() if k != "controller"},
            }
        )
        if policy.pareto:
            dominators = [
                x["candidate_id"]
                for x in finalists
                if x is not d
                and canonical(
                    {
                        "task": x["record"]["task"],
                        "blueprint": {
                            k: v
                            for k, v in x["record"]["algorithm"].items()
                            if k != "controller"
                        },
                    }
                )
                == context
                and pareto_dominates(x, d)
            ]
            if dominators:
                d["layers"][4] = layer(
                    LAYERS[4],
                    "REJECT",
                    "lower research priority under the declared like-for-like Pareto policy; not refuted or proven inferior",
                    dominated_by=sorted(dominators),
                )
    for d in decisions:
        finalize(d)
    return sorted(decisions, key=lambda d: d["index"]), seen, families


def make_audit(
    records, gen, policy, sources, round_no, previous_hash, seen, families, known
):
    decisions, new_seen, new_families = evaluate_records(
        records, gen, policy, seen, families, known
    )
    audit = {
        "schema_version": VERSION,
        "round_no": round_no,
        "previous_hash": previous_hash,
        "sources": sources,
        "policy": asdict(policy),
        "registry": REGISTRY,
        "before_seen": seen,
        "before_families": families,
        "known": known,
        "decisions": decisions,
        "decision_digest": digest(
            sorted(
                ({k: v for k, v in d.items() if k != "index"} for d in decisions),
                key=canonical,
            )
        ),
        "layer_counts": [
            sum(
                all(d["layers"][i]["status"] == "PASS" for i in range(n + 1))
                for d in decisions
            )
            for n in range(5)
        ],
        "novelty_status": "UNVERIFIED",
        "data_access": "metadata and public symbolic probes only; no fitting or experimental outcomes",
    }
    audit["audit_hash"] = digest(audit)
    return audit, new_seen, new_families


class FiveLayerFilter:
    def __init__(self, journal_path, generator_path=None, policy=None, create=True):
        self.path = Path(journal_path).resolve()
        self.generator_path = Path(
            generator_path
            or os.environ.get("HYPOTHESIS_GENERATOR")
            or Path(__file__).resolve().parent.parent
            / "generator_only"
            / "hypothesis_generator.py"
        ).resolve()
        self.gen = load_generator(self.generator_path)
        self.policy = Policy.parse(policy)
        self.sources = {
            "generator": GENERATOR_SHA256,
            "filter": SOURCE_SHA256,
            "registry": REGISTRY_SHA256,
        }
        if create:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            with self.connect() as db:
                db.executescript(
                    "CREATE TABLE config(id INTEGER PRIMARY KEY,body TEXT);"
                    "CREATE TABLE seen(semantic_key TEXT PRIMARY KEY,candidate_id TEXT);"
                    "CREATE TABLE families(family_key TEXT PRIMARY KEY,candidate_id TEXT);"
                    "CREATE TABLE known(semantic_key TEXT PRIMARY KEY,reference TEXT);"
                    "CREATE TABLE rounds(round_no INTEGER PRIMARY KEY,audit_hash TEXT UNIQUE,body TEXT);"
                    "CREATE TABLE review(name TEXT PRIMARY KEY,payload TEXT);"
                )
                db.execute(
                    "INSERT INTO config VALUES(1,?)",
                    (
                        canonical(
                            {
                                "version": VERSION,
                                "sources": self.sources,
                                "policy": asdict(self.policy),
                                "known_index_hash": digest({}),
                            }
                        ),
                    ),
                )
        else:
            if not self.path.exists():
                raise ValueError("journal missing")
            with self.connect() as db:
                config = json.loads(db.execute("SELECT body FROM config").fetchone()[0])
            if config["version"] != VERSION or config["sources"] != self.sources:
                raise ValueError("journal source/registry version mismatch")
            restored = Policy.parse(config["policy"])
            if policy is not None and restored != self.policy:
                raise ValueError("cannot change journal policy")
            self.policy = restored

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=120)
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def check_sources(self):
        if (
            file_hash(self.generator_path) != GENERATOR_SHA256
            or file_hash(__file__) != SOURCE_SHA256
            or digest(REGISTRY) != REGISTRY_SHA256
        ):
            raise ValueError("source changed during filter session")

    def register_prior_art(self, record, reference_id):
        self.check_sources()
        record = bounded_json(record)
        self.gen.Hypothesis.from_dict(record)
        if reference_id not in {r["id"] for r in REFERENCES}:
            raise ValueError("unregistered reference")
        key = identities(
            record, normalize_controller(record["algorithm"]["controller"])
        )[0]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT count(*) FROM rounds").fetchone()[0]:
                raise ValueError("known-index additions must precede the first round")
            db.execute(
                "INSERT INTO known VALUES(?,?)",
                (
                    key,
                    canonical(
                        {
                            "reference_id": reference_id,
                            "record": record,
                            "scope": "caller-curated exclusion; source-equivalence mapping requires human review",
                        }
                    ),
                ),
            )

            known = {k: json.loads(v) for k, v in db.execute("SELECT * FROM known")}
            config = json.loads(
                db.execute("SELECT body FROM config WHERE id=1").fetchone()[0]
            )
            config["known_index_hash"] = digest(known)
            db.execute("UPDATE config SET body=? WHERE id=1", (canonical(config),))

    def _check_journal(self, db):
        config = json.loads(
            db.execute("SELECT body FROM config WHERE id=1").fetchone()[0]
        )
        known = {k: json.loads(v) for k, v in db.execute("SELECT * FROM known")}
        expected = {
            "version": VERSION,
            "sources": self.sources,
            "policy": asdict(self.policy),
            "known_index_hash": digest(known),
        }
        if canonical(config) != canonical(expected):
            raise ValueError("journal configuration or known index altered")
        seen = {}
        families = {}
        previous = ""
        audits = []
        for number, (stored_number, stored_hash, body) in enumerate(
            db.execute(
                "SELECT round_no,audit_hash,body FROM rounds ORDER BY round_no"
            ).fetchall(),
            1,
        ):
            audit = bounded_audit(read_json(body))
            if (
                type(stored_number) is not int
                or stored_number != number
                or audit["round_no"] != number
            ):
                raise ValueError("journal round history is incomplete")
            expected, new_seen, new_families = make_audit(
                [
                    {
                        "input_digest": d["candidate_id"],
                        "record": d["record"],
                        "coherence_code": d["coherence_code"],
                    }
                    for d in audit["decisions"]
                ],
                self.gen,
                self.policy,
                self.sources,
                number,
                previous,
                seen,
                families,
                known,
            )
            if canonical(expected) != body or stored_hash != expected["audit_hash"]:
                raise ValueError("journal round or derived decision altered")
            audits.append(audit)
            seen = new_seen
            families = new_families
            previous = stored_hash
        if seen != dict(db.execute("SELECT * FROM seen")) or families != dict(
            db.execute("SELECT * FROM families")
        ):
            raise ValueError(
                "persistent duplicate registry disagrees with committed history"
            )
        return audits, seen, families, known

    def evaluate_batch(self, records):
        self.check_sources()
        if type(records) is not list or not 1 <= len(records) <= 1000:
            raise ValueError("batch of 1..1000 hypothesis records required")
        records = [bounded_json(r) for r in records]
        if any(type(r) is not dict for r in records):
            raise ValueError("hypothesis objects required")
        records = prepare_inputs(records, self.gen)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            audits, seen, families, known = self._check_journal(db)
            audit, new_seen, new_families = make_audit(
                records,
                self.gen,
                self.policy,
                self.sources,
                len(audits) + 1,
                audits[-1]["audit_hash"] if audits else "",
                seen,
                families,
                known,
            )
            db.executemany(
                "INSERT INTO seen VALUES(?,?)",
                [(k, v) for k, v in new_seen.items() if k not in seen],
            )
            db.executemany(
                "INSERT INTO families VALUES(?,?)",
                [(k, v) for k, v in new_families.items() if k not in families],
            )
            db.execute(
                "INSERT INTO rounds VALUES(?,?,?)",
                (audit["round_no"], audit["audit_hash"], canonical(audit)),
            )
        return audit

    def replay_audit(self, audit):
        try:
            self.check_sources()
            audit = bounded_audit(audit)
            with self.connect() as db:
                db.execute("BEGIN")
                audits, _, _, _ = self._check_journal(db)
            return 1 <= audit["round_no"] <= len(audits) and canonical(
                audits[audit["round_no"] - 1]
            ) == canonical(audit)
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
            RecursionError,
            OSError,
            OverflowError,
            sqlite3.Error,
        ):
            return False

    def audit(self, round_no):
        if type(round_no) is not int or round_no < 1:
            raise ValueError("positive strict round number required")
        with self.connect() as db:
            row = db.execute(
                "SELECT body FROM rounds WHERE round_no=?", (round_no,)
            ).fetchone()
        if row is None:
            raise ValueError("round missing")
        return json.loads(row[0])

    def handoff(self, candidate_id):
        if type(candidate_id) is not str or len(candidate_id) != 64:
            raise ValueError("candidate ID required")
        with self.connect() as db:
            rows = db.execute("SELECT body FROM rounds ORDER BY round_no").fetchall()
        matches = []
        for row in rows:
            audit = json.loads(row[0])
            for d in audit["decisions"]:
                if d["candidate_id"] == candidate_id and d["eligible_for_math"]:
                    matches.append((audit, d))
        if len(matches) != 1:
            raise ValueError("candidate has no unique committed five-layer pass")
        audit, d = matches[0]
        if not self.replay_audit(audit):
            raise ValueError("filter audit did not replay")
        receipt = {
            "status": "READY_FOR_MATH_INVESTIGATION",
            "candidate_id": candidate_id,
            "semantic_key": d["semantic_key"],
            "family_key": d["family_key"],
            "round_no": audit["round_no"],
            "audit_hash": audit["audit_hash"],
            "sources": self.sources,
            "policy_hash": digest(asdict(self.policy)),
            "novelty_status": "UNVERIFIED",
            "scientific_status": "NOT_EVALUATED",
        }
        packet = {"candidate": copy.deepcopy(d["record"]), "filter_receipt": receipt}
        packet["handoff_hash"] = digest(packet)
        return packet

    def verify_handoff(self, packet):
        try:
            strict(
                packet,
                ("candidate", "filter_receipt", "handoff_hash"),
                ("candidate", "filter_receipt", "handoff_hash"),
            )
            cid = digest(packet["candidate"])
            return canonical(packet) == canonical(self.handoff(cid))
        except (ValueError, TypeError, KeyError, IndexError, OSError, RecursionError):
            return False

    def forward_to_math(self, packet, math_engine, runtime=None, math_policy=None):
        """Analyze only this selected record through the existing public API.

        The upstream engine retains origin authority. An UNATTESTED origin is
        not promoted, and no generation prefix/private ledger is touched.
        """
        if not self.verify_handoff(packet):
            raise ValueError("invalid hypothesis-filter handoff")
        record = copy.deepcopy(packet["candidate"])
        receipt = math_engine.assess(record, runtime=runtime, policy=math_policy)
        if not math_engine.verify_receipt(record, receipt, runtime, math_policy):
            raise ValueError("mathematical receipt failed replay")
        return {
            "candidate": record,
            "filter_handoff": copy.deepcopy(packet),
            "math_admission": receipt,
            "math_status": receipt["status"],
            "novelty_status": "UNVERIFIED",
            "scientific_status": "NOT_EVALUATED",
            "origin_scope": "upstream origin authority retained; no issuance fabricated and no predecessor hypotheses analyzed",
        }


def bounded_audit(value):
    if (
        type(value) is not dict
        or type(value.get("round_no")) is not int
        or value["round_no"] < 1
    ):
        raise ValueError("strict audit round required")
    # Audits contain many individually bounded records. Serialize once to reject
    # non-JSON/nonfinite types; the journal equality binds all nested fields.
    if len(canonical(value)) > 250_000_000:
        raise ValueError("audit size limit exceeded")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("demo")
    run.add_argument("artifact", type=Path)
    run.add_argument("--counts", nargs="+", type=int, default=[100, 500, 1000])
    run.add_argument("--seed", type=int, default=42)
    run.add_argument(
        "--kind", choices=("classification", "regression"), default="classification"
    )
    inspect = commands.add_parser("inspect")
    inspect.add_argument("artifact", type=Path)
    args = parser.parse_args()
    if args.command == "demo":
        engine = FiveLayerFilter(args.artifact, args.generator)
        task = engine.gen.TaskSpec(
            args.kind,
            8,
            3 if args.kind == "classification" else 1,
            goals=("outliers", "ill_conditioned"),
        )
        generator = engine.gen.HypothesisGenerator(args.seed)
        for count in args.counts:
            if not 1 <= count <= 1000:
                raise ValueError("round count outside 1..1000")
            start = time.perf_counter()
            records = [h.to_dict() for h in generator.generate(task, count)]
            generation = time.perf_counter() - start
            start = time.perf_counter()
            audit = engine.evaluate_batch(records)
            elapsed = time.perf_counter() - start
            if not engine.replay_audit(audit):
                raise ValueError("demonstration audit failed replay")
            result = {
                "round": audit["round_no"],
                "generated": count,
                "layer_counts": audit["layer_counts"],
                "generation_seconds": generation,
                "filter_seconds": elapsed,
                "decision_digest": audit["decision_digest"],
                "novelty_status": "UNVERIFIED",
                "replayed": True,
            }
            with engine.connect() as db:
                db.execute(
                    "INSERT INTO review VALUES(?,?)",
                    (
                        "round_" + str(audit["round_no"]) + "_measurements",
                        canonical(result),
                    ),
                )
            print(canonical(result), flush=True)
    else:
        engine = FiveLayerFilter(args.artifact, args.generator, create=False)
        with engine.connect() as db:
            rounds = [
                x[0]
                for x in db.execute("SELECT round_no FROM rounds ORDER BY round_no")
            ]
        results = []
        for n in rounds:
            audit = engine.audit(n)
            results.append(
                {
                    "round": n,
                    "generated": len(audit["decisions"]),
                    "layer_counts": audit["layer_counts"],
                    "replayed": engine.replay_audit(audit),
                    "decision_digest": audit["decision_digest"],
                }
            )
        print(json.dumps({"rounds": results, "novelty_status": "UNVERIFIED"}, indent=2))


if __name__ == "__main__":
    main()
