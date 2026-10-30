"""Exact mathematical claim checking and certificate replay, not scientific ML evaluation.

One implementation file. No eval, pickle, user-code execution, training data,
family-battery scoring or numerical sampling as proof. VERIFIED always requires
certificate replay. Unsupported statements remain UNVERIFIED. See REPORT.md.
"""

from __future__ import annotations
import argparse
import ast
from collections import OrderedDict
import copy
from dataclasses import dataclass, field, asdict
from fractions import Fraction as F
import hashlib
from itertools import product, permutations
import json
import math
import random
import re
import sys
from time import perf_counter

VERSION = "exact-math-1.0"
MSC_SOURCE = "https://mathscinet.ams.org/msc/pdfs/classifications2020.pdf"
_MSC = """00|General and overarching topics; collections
01|History and biography
03|Mathematical logic and foundations
05|Combinatorics
06|Order, lattices, ordered algebraic structures
08|General algebraic systems
11|Number theory
12|Field theory and polynomials
13|Commutative algebra
14|Algebraic geometry
15|Linear and multilinear algebra; matrix theory
16|Associative rings and algebras
17|Nonassociative rings and algebras
18|Category theory; homological algebra
19|K-theory
20|Group theory and generalizations
22|Topological groups, Lie groups
26|Real functions
28|Measure and integration
30|Functions of a complex variable
31|Potential theory
32|Several complex variables and analytic spaces
33|Special functions
34|Ordinary differential equations
35|Partial differential equations
37|Dynamical systems and ergodic theory
39|Difference and functional equations
40|Sequences, series, summability
41|Approximations and expansions
42|Harmonic analysis on Euclidean spaces
43|Abstract harmonic analysis
44|Integral transforms, operational calculus
45|Integral equations
46|Functional analysis
47|Operator theory
49|Calculus of variations and optimal control; optimization
51|Geometry
52|Convex and discrete geometry
53|Differential geometry
54|General topology
55|Algebraic topology
57|Manifolds and cell complexes
58|Global analysis, analysis on manifolds
60|Probability theory and stochastic processes
62|Statistics
65|Numerical analysis
68|Computer science
70|Mechanics of particles and systems
74|Mechanics of deformable solids
76|Fluid mechanics
78|Optics, electromagnetic theory
80|Classical thermodynamics, heat transfer
81|Quantum theory
82|Statistical mechanics, structure of matter
83|Relativity and gravitational theory
85|Astronomy and astrophysics
86|Geophysics
90|Operations research, mathematical programming
91|Game theory, economics, finance, social and behavioral sciences
92|Biology and other natural sciences
93|Systems theory; control
94|Information and communication theory, circuits
97|Mathematics education"""
MATH_FAMILIES = dict(row.split("|", 1) for row in _MSC.splitlines())


class Invalid(ValueError):
    pass


class Unsupported(Exception):
    pass


class Limit(Exception):
    pass


class Undefined(Exception):
    pass


class NonExact(Exception):
    pass


@dataclass(frozen=True)
class Limits:
    timeout_seconds: float = 3.0
    max_expression_chars: int = 4096
    max_nodes: int = 192
    max_variables: int = 24
    max_terms: int = 4096
    max_degree: int = 32
    max_operations: int = 200000
    max_coefficient_bits: int = 1024
    max_enumeration: int = 4096
    counterexample_trials: int = 192
    max_matrix_size: int = 4
    max_payload_bytes: int = 300000
    cache_entries: int = 512

    def __post_init__(self):
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or not 0.01 <= self.timeout_seconds <= 60
        ):
            raise Invalid("timeout_seconds must be finite in [.01,60]")
        for name in self.__dataclass_fields__:
            if name != "timeout_seconds" and (
                type(getattr(self, name)) is not int or getattr(self, name) < 1
            ):
                raise Invalid("positive integer limits required")


class Budget:
    def __init__(self, limits=None):
        self.limits = limits or Limits()
        self.deadline = perf_counter() + self.limits.timeout_seconds
        self.operations = 0

    def tick(self, n=1):
        self.operations += n
        if (
            self.operations > self.limits.max_operations
            or perf_counter() > self.deadline
        ):
            raise Limit("arithmetic/time budget exceeded")

    def rational(self, x):
        self.tick()
        if (
            max(abs(x.numerator).bit_length(), x.denominator.bit_length())
            > self.limits.max_coefficient_bits
        ):
            raise Limit("coefficient bit limit exceeded")
        return x


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def strict(d, allowed, required=()):
    if not isinstance(d, dict) or set(d) - set(allowed) or set(required) - set(d):
        raise Invalid("unknown/missing fields")


def rational(x):
    if type(x) is int:
        if abs(x).bit_length() > 1024:
            raise Limit("integer too large")
        return F(x)
    if (
        not isinstance(x, str)
        or len(x) > 100
        or not re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,3})?(?:/[+-]?\d+)?", x
        )
    ):
        raise Invalid("use an integer or an exact rational/decimal string")
    if "e" in x.lower() and abs(int(x.lower().split("e")[1].split("/")[0])) > 100:
        raise Limit("numeric exponent too large")
    try:
        value = F(x)
    except (ValueError, ZeroDivisionError) as e:
        raise Invalid("invalid rational") from e
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 1024:
        raise Limit("rational too large")
    return value


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple = ()
    value: object = None

    def data(self):
        return [
            self.op,
            str(self.value) if self.op == "const" else self.value,
            [a.data() for a in self.args],
        ]

    @property
    def key(self):
        return canonical(self.data())


C = lambda x: Expr("const", value=F(x))
S = lambda x: Expr("symbol", value=x)


def E(op, *a):
    return Expr(op, tuple(a))


CALLS = {
    "exp": 1,
    "log": 1,
    "sqrt": 1,
    "abs": 1,
    "min": 2,
    "max": 2,
    "sin": 1,
    "cos": 1,
    "gcd": 2,
    "binomial": 2,
    "comb": 2,
}
RESERVED = set(CALLS) | {"pi", "e"}
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,31}\Z")


def parse(text, names, budget):
    if type(text) is int:
        return C(rational(text))
    if not isinstance(text, str) or len(text) > budget.limits.max_expression_chars:
        raise Invalid("bounded mathematical expression string required")
    source = text.replace("^", "**")
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as e:
        raise Invalid("invalid expression syntax") from e
    if sum(1 for _ in ast.walk(tree)) > budget.limits.max_nodes:
        raise Limit("expression node budget exceeded")

    def visit(n):
        budget.tick()
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            # Use original decimal lexeme, NEVER the rounded Python float.
            return C(rational(ast.get_source_segment(source, n)))
        if isinstance(n, ast.Name):
            if n.id in names:
                return S(n.id)
            if n.id == "e":
                return E("exp", C(1))
            if n.id == "pi":
                return E("pi")
            raise Invalid("undeclared symbol: " + n.id)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            a = visit(n.operand)
            return E("neg", a) if isinstance(n.op, ast.USub) else a
        if isinstance(n, ast.BinOp):
            ops = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.Div: "div",
                ast.Pow: "pow",
            }
            op = ops.get(type(n.op))
            if op is None:
                raise Invalid("unsupported arithmetic operator")
            a, b = visit(n.left), visit(n.right)
            if op == "pow":
                value = exact(b, {}, budget)
                if value.denominator != 1 or abs(value) > 8:
                    raise Limit("only integer powers -8..8 are supported")
                b = C(value)
            return E(op, a, b)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in CALLS
            and not n.keywords
            and len(n.args) == CALLS[n.func.id]
        ):
            return E(
                "binomial" if n.func.id == "comb" else n.func.id,
                *(visit(a) for a in n.args),
            )
        raise Invalid("expression uses an unsupported or executable Python construct")

    return visit(tree.body)


def exact(e, point, budget=None):
    budget = budget or Budget()
    budget.tick()
    checked = budget.rational
    if e.op == "const":
        return checked(e.value)
    if e.op == "symbol":
        return checked(point[e.value])
    if e.op == "pi":
        raise NonExact("pi is not rational")
    v = [exact(a, point, budget) for a in e.args]
    if e.op == "neg":
        return checked(-v[0])
    if e.op == "add":
        return checked(v[0] + v[1])
    if e.op == "sub":
        return checked(v[0] - v[1])
    if e.op == "mul":
        return checked(v[0] * v[1])
    if e.op == "div":
        if v[1] == 0:
            raise Undefined("division by zero")
        return checked(v[0] / v[1])
    if e.op == "pow":
        if v[0] == 0 and v[1] < 0:
            raise Undefined("negative power of zero")
        n = int(v[1])
        size = max(abs(v[0].numerator).bit_length(), v[0].denominator.bit_length())
        if abs(n) * max(0, size - 1) > budget.limits.max_coefficient_bits:
            raise Limit("exact exponentiation bit budget exceeded before computation")
        return checked(v[0] ** n)
    if e.op == "abs":
        return abs(v[0])
    if e.op == "min":
        return min(v)
    if e.op == "max":
        return max(v)
    if e.op == "sqrt":
        if v[0] < 0:
            raise Undefined("negative square root over reals")
        n, d = math.isqrt(v[0].numerator), math.isqrt(v[0].denominator)
        if n * n == v[0].numerator and d * d == v[0].denominator:
            return F(n, d)
        raise NonExact("irrational square root")
    if e.op == "log":
        if v[0] <= 0:
            raise Undefined("logarithm outside positive reals")
        if v[0] == 1:
            return F(0)
        raise NonExact("non-rational logarithm")
    if e.op == "exp" and v[0] == 0:
        return F(1)
    if e.op == "sin" and v[0] == 0:
        return F(0)
    if e.op == "cos" and v[0] == 0:
        return F(1)
    if e.op in ("gcd", "binomial"):
        if any(a.denominator != 1 for a in v):
            raise Undefined("integer-valued function requires integer arguments")
        a, b = map(int, v)
        if e.op == "gcd":
            return checked(F(math.gcd(a, b)))
        if a < 0 or b < 0:
            raise Undefined("binomial requires nonnegative integers")
        if b > a:
            return F(0)
        k = min(b, a - b)
        # Bound native comb's work BEFORE calling it; input length alone does
        # not bound k. The estimate is conservative, never evidence of truth.
        if k * max(1, a.bit_length()) > 2 * budget.limits.max_coefficient_bits:
            raise Limit("binomial work/bit budget exceeded before computation")
        budget.tick(k + 1)
        return checked(F(math.comb(a, b)))
    raise NonExact("no exact rational value available")


# Sparse multivariate polynomial arithmetic, independent of SymPy.
def pc(c):
    return {} if not c else {(): F(c)}


def pa(a, b, budget, scale=F(1)):
    out = dict(a)
    for m, c in b.items():
        budget.tick()
        value = budget.rational(out.get(m, F(0)) + scale * c)
        if value:
            out[m] = value
        elif m in out:
            del out[m]
    if len(out) > budget.limits.max_terms:
        raise Limit("polynomial term budget exceeded")
    return out


def pm(a, b, budget):
    out = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            budget.tick()
            m = dict(ma)
            for symbol, power in mb:
                m[symbol] = m.get(symbol, 0) + power
            if sum(m.values()) > budget.limits.max_degree:
                raise Limit("polynomial degree budget exceeded")
            key = tuple(sorted(m.items()))
            v = budget.rational(out.get(key, F(0)) + ca * cb)
            if v:
                out[key] = v
            elif key in out:
                del out[key]
            if len(out) > budget.limits.max_terms:
                raise Limit("polynomial term budget exceeded")
    return out


def pp(a, power, budget):
    out = pc(1)
    for _ in range(power):
        out = pm(out, a, budget)
    return out


def pack(p):
    return [[str(c), [[a, n] for a, n in m]] for m, c in sorted(p.items())]


def unpack(data, budget):
    if not isinstance(data, list) or len(data) > budget.limits.max_terms:
        raise Invalid("invalid polynomial certificate")
    p = {}
    for row in data:
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[1], list):
            raise Invalid("invalid polynomial term")
        c = rational(row[0])
        m = []
        for a, n in row[1]:
            if (
                not isinstance(a, str)
                or len(a) > 128
                or type(n) is not int
                or not 1 <= n <= budget.limits.max_degree
            ):
                raise Invalid("invalid monomial")
            m.append((a, n))
        if (
            m != sorted(m)
            or len({a for a, _ in m}) != len(m)
            or sum(n for _, n in m) > budget.limits.max_degree
        ):
            raise Invalid("noncanonical monomial")
        key = tuple(m)
        if key in p or not c:
            raise Invalid("noncanonical polynomial")
        p[key] = budget.rational(c)
    return p


@dataclass
class Rat:
    n: dict
    d: dict


def rn(n, d, budget):
    if not d:
        raise Undefined("zero formal denominator")
    if not n:
        return Rat({}, pc(1))
    if n == d:
        return Rat(pc(1), pc(1))
    if set(d) == {()}:
        factor = d[()]
        return Rat({m: budget.rational(c / factor) for m, c in n.items()}, pc(1))
    return Rat(n, d)


def radd(a, b, budget, sign=1):
    if a.d == b.d:
        return rn(pa(a.n, b.n, budget, F(sign)), a.d, budget)
    return rn(
        pa(pm(a.n, b.d, budget), pm(b.n, a.d, budget), budget, F(sign)),
        pm(a.d, b.d, budget),
        budget,
    )


def rmul(a, b, budget):
    an, ad, bn, bd = a.n, a.d, b.n, b.d
    if an == bd:
        an, bd = pc(1), pc(1)
    if bn == ad:
        bn, ad = pc(1), pc(1)
    return rn(pm(an, bn, budget), pm(ad, bd, budget), budget)


def normalize(e, ctx):
    b = ctx.budget
    b.tick()
    try:
        return Rat(pc(exact(e, {}, b)), pc(1))
    except (KeyError, NonExact):
        pass
    if e.op == "symbol":
        return Rat({((e.value, 1),): F(1)}, pc(1))
    if e.op == "neg":
        r = normalize(e.args[0], ctx)
        return Rat({m: -c for m, c in r.n.items()}, r.d)
    if e.op in ("add", "sub"):
        return radd(
            normalize(e.args[0], ctx),
            normalize(e.args[1], ctx),
            b,
            1 if e.op == "add" else -1,
        )
    if e.op == "mul":
        return rmul(normalize(e.args[0], ctx), normalize(e.args[1], ctx), b)
    if e.op == "div":
        a, c = normalize(e.args[0], ctx), normalize(e.args[1], ctx)
        return rmul(a, Rat(c.d, c.n), b)
    if e.op == "pow":
        a = normalize(e.args[0], ctx)
        n = int(e.args[1].value)
        if n < 0:
            a = Rat(a.d, a.n)
            n = -n
        return rn(pp(a.n, n, b), pp(a.d, n, b), b)
    # Elementary functions are opaque real atoms in algebra, never freely
    # simplified with unproved identities or unsafe branch conventions.
    atom = "@" + hashlib.sha256(e.key.encode()).hexdigest()
    ctx.atoms[atom] = e
    return Rat({((atom, 1),): F(1)}, pc(1))


@dataclass(frozen=True)
class Interval:
    lo: F | None = None
    hi: F | None = None
    lo_open: bool = False
    hi_open: bool = False

    def contains(self, x):
        return (
            self.lo is None or x > self.lo or (x == self.lo and not self.lo_open)
        ) and (self.hi is None or x < self.hi or (x == self.hi and not self.hi_open))

    @property
    def empty(self):
        return (
            self.lo is not None
            and self.hi is not None
            and (
                self.lo > self.hi
                or (self.lo == self.hi and (self.lo_open or self.hi_open))
            )
        )

    @property
    def positive(self):
        return self.lo is not None and (self.lo > 0 or (self.lo == 0 and self.lo_open))

    @property
    def nonnegative(self):
        return self.lo is not None and self.lo >= 0

    @property
    def negative(self):
        return self.hi is not None and (self.hi < 0 or (self.hi == 0 and self.hi_open))

    @property
    def nonzero(self):
        return self.positive or self.negative

    def data(self):
        return dict(
            lo=None if self.lo is None else str(self.lo),
            hi=None if self.hi is None else str(self.hi),
            lo_open=self.lo_open,
            hi_open=self.hi_open,
        )


WHOLE = Interval()


def iadd(a, b):
    return Interval(
        None if a.lo is None or b.lo is None else a.lo + b.lo,
        None if a.hi is None or b.hi is None else a.hi + b.hi,
        a.lo_open or b.lo_open,
        a.hi_open or b.hi_open,
    )


def ineg(a):
    return Interval(
        None if a.hi is None else -a.hi,
        None if a.lo is None else -a.lo,
        a.hi_open,
        a.lo_open,
    )


def ext(v, lower):
    return (-math.inf if lower else math.inf) if v is None else v


def imul(a, b):
    points = []
    for x, xo in ((ext(a.lo, True), a.lo_open), (ext(a.hi, False), a.hi_open)):
        for y, yo in ((ext(b.lo, True), b.lo_open), (ext(b.hi, False), b.hi_open)):
            if x == 0 or y == 0:
                value = F(0)
            elif isinstance(x, float) or isinstance(y, float):
                value = math.inf if (x > 0) == (y > 0) else -math.inf
            else:
                value = x * y
            points.append((value, xo or yo))
    low = min(v for v, _ in points)
    high = max(v for v, _ in points)
    zero_attained = a.contains(F(0)) or b.contains(F(0))
    lo_open = (not zero_attained) if low == 0 else all(o for v, o in points if v == low)
    hi_open = (
        (not zero_attained) if high == 0 else all(o for v, o in points if v == high)
    )
    return Interval(
        None if isinstance(low, float) else low,
        None if isinstance(high, float) else high,
        lo_open,
        hi_open,
    )


def reciprocal(a):
    if not a.nonzero:
        return WHOLE

    def inv(v):
        return None if v == 0 else F(0) if v is None else 1 / v

    return Interval(
        inv(a.hi), inv(a.lo), a.hi_open or a.hi is None, a.lo_open or a.lo is None
    )


def interval(e, ctx):
    result = _interval(e, ctx)
    for endpoint in (result.lo, result.hi):
        if endpoint is not None:
            ctx.budget.rational(endpoint)
    return result


def _interval(e, ctx):
    ctx.budget.tick()
    if e.op == "const":
        return Interval(e.value, e.value)
    if e.op == "symbol":
        return ctx.bounds[e.value]
    if e.op == "pi":
        return Interval(F(3), F(4), True, True)
    if e.op == "mul" and e.args[0] == e.args[1]:
        return interval(E("pow", e.args[0], C(2)), ctx)
    v = [interval(a, ctx) for a in e.args]
    if e.op == "neg":
        return ineg(v[0])
    if e.op == "add":
        return iadd(*v)
    if e.op == "sub":
        return iadd(v[0], ineg(v[1]))
    if e.op == "mul":
        return imul(*v)
    if e.op == "div":
        return imul(v[0], reciprocal(v[1]))
    if e.op == "pow":
        a = v[0]
        n = int(e.args[1].value)
        if n == 0:
            return Interval(F(1), F(1))
        if n < 0:
            return reciprocal(interval(E("pow", e.args[0], C(-n)), ctx))
        if n % 2 == 0:
            upper = None if a.lo is None or a.hi is None else max(a.lo**n, a.hi**n)
            low = (
                F(0)
                if a.contains(F(0)) or not a.nonzero
                else min(abs(a.lo or 0), abs(a.hi or 0)) ** n
            )
            # For unbounded strictly positive/negative domains, zero is only a lower bound.
            if a.lo is None or a.hi is None:
                low = F(0)
            return Interval(low, upper, low == 0 and a.nonzero, False)
        out = Interval(F(1), F(1))
        for _ in range(n):
            out = imul(out, a)
        return out
    if e.op == "abs":
        if v[0].nonnegative:
            return v[0]
        if ineg(v[0]).nonnegative:
            return ineg(v[0])
        upper = (
            None
            if v[0].lo is None or v[0].hi is None
            else max(abs(v[0].lo), abs(v[0].hi))
        )
        return Interval(F(0), upper)
    if e.op in ("min", "max"):
        a, b = v
        fn = min if e.op == "min" else max
        low = fn(ext(a.lo, True), ext(b.lo, True))
        high = fn(ext(a.hi, False), ext(b.hi, False))
        la = [z.lo_open for z in (a, b) if ext(z.lo, True) == low]
        ha = [z.hi_open for z in (a, b) if ext(z.hi, False) == high]
        return Interval(
            None if isinstance(low, float) else low,
            None if isinstance(high, float) else high,
            (all(la) if e.op == "min" else any(la)),
            (any(ha) if e.op == "min" else all(ha)),
        )
    if e.op == "sqrt":
        a = v[0]
        upper = None
        if a.hi is not None and a.hi >= 0:
            root = math.isqrt(a.hi.numerator // a.hi.denominator)
            upper = F(root if F(root * root) == a.hi else root + 1)
        return Interval(F(0), upper, a.positive, False)
    if e.op == "exp":
        a = v[0]
        low = F(1) if a.nonnegative else F(0)
        high = F(1) if ineg(a).nonnegative else None
        return Interval(
            low, high, low == 0 or (a.lo == 0 and a.lo_open), a.hi == 0 and a.hi_open
        )
    if e.op == "log":
        a = v[0]
        return Interval(
            F(0) if a.lo is not None and a.lo >= 1 else None,
            F(0) if a.hi is not None and a.hi <= 1 else None,
            a.lo is not None and (a.lo > 1 or (a.lo == 1 and a.lo_open)),
            a.hi is not None and (a.hi < 1 or (a.hi == 1 and a.hi_open)),
        )
    if e.op in ("sin", "cos"):
        return Interval(F(-1), F(1))
    return WHOLE


OPS = {"=": "==", "==": "==", "!=": "!=", "<": "<", ">": ">", "<=": "<=", ">=": ">="}


def relation(a, op, b):
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    if op == "<":
        return a < b
    if op == ">":
        return a > b
    if op == "<=":
        return a <= b
    if op == ">=":
        return a >= b
    raise Invalid("unknown relation")


class Context:
    def __init__(self, variables, assumptions, budget):
        if (
            not isinstance(variables, dict)
            or len(variables) > budget.limits.max_variables
        ):
            raise Invalid("bounded variable mapping required")
        self.budget = budget
        self.bounds = {}
        self.domains = {}
        self.atoms = {}
        self.assumptions = []
        for name, d in variables.items():
            if (
                not isinstance(name, str)
                or not NAME.fullmatch(name)
                or name in RESERVED
            ):
                raise Invalid("invalid/reserved symbol")
            strict(d, ("domain", "lower", "upper", "lower_open", "upper_open"))
            domain = d.get("domain", "real")
            if domain not in ("real", "integer"):
                raise Invalid("real/integer variable domains only")
            for key in ("lower_open", "upper_open"):
                if key in d and type(d[key]) is not bool:
                    raise Invalid("openness must be Boolean")
            self.bounds[name] = Interval(
                None if d.get("lower") is None else rational(d["lower"]),
                None if d.get("upper") is None else rational(d["upper"]),
                d.get("lower_open", False),
                d.get("upper_open", False),
            )
            self.domains[name] = domain
        if not isinstance(assumptions, list) or len(assumptions) > 32:
            raise Invalid("bounded assumption list required")
        for item in assumptions:
            strict(item, ("lhs", "op", "rhs"), ("lhs", "op", "rhs"))
            if item["op"] not in OPS:
                raise Invalid("invalid assumption relation")
            a, b = parse(item["lhs"], self.bounds, budget), parse(
                item["rhs"], self.bounds, budget
            )
            op = OPS[item["op"]]
            self.assumptions.append((a, op, b))
            # Only transparent single-variable bounds are propagated.
            if a.op == "const" and b.op == "symbol":
                a, b = b, a
                op = {
                    "<": ">",
                    ">": "<",
                    "<=": ">=",
                    ">=": "<=",
                    "==": "==",
                    "!=": "!=",
                }[op]
            if a.op == "symbol" and b.op == "const" and op != "!=":
                old = self.bounds[a.value]
                lo, hi, lop, hip = old.lo, old.hi, old.lo_open, old.hi_open
                if op in (">", ">=", "=="):
                    if lo is None or b.value > lo:
                        lo, lop = b.value, op == ">"
                    elif b.value == lo:
                        lop = lop or op == ">"
                if op in ("<", "<=", "=="):
                    if hi is None or b.value < hi:
                        hi, hip = b.value, op == "<"
                    elif b.value == hi:
                        hip = hip or op == "<"
                self.bounds[a.value] = Interval(lo, hi, lop, hip)

    def vacant(self):
        if any(i.empty for i in self.bounds.values()):
            return True
        for a, op, b in self.assumptions:
            try:
                if not relation(
                    exact(a, {}, self.budget), op, exact(b, {}, self.budget)
                ):
                    return True
            except (KeyError, NonExact, Undefined):
                pass
        return False

    def allowed(self, point):
        if set(point) != set(self.bounds):
            return False
        for k, x in point.items():
            if not self.bounds[k].contains(x) or (
                self.domains[k] == "integer" and x.denominator != 1
            ):
                return False
        # Undecidable/undefined is distinct from false. In particular finite
        # enumeration must not silently drop these points and certify emptiness.
        return all(
            relation(exact(a, point, self.budget), op, exact(b, point, self.budget))
            for a, op, b in self.assumptions
        )


# Exact LDL decomposition: certificate discovery and independent matrix replay.
def matrix_product(a, b):
    return [
        [
            sum((a[i][k] * b[k][j] for k in range(len(b))), F(0))
            for j in range(len(b[0]))
        ]
        for i in range(len(a))
    ]


def transpose(a):
    return list(map(list, zip(*a)))


def ldl(a, budget):
    n = len(a)
    work = copy.deepcopy(a)
    L = [[F(i == j) for j in range(n)] for i in range(n)]
    D = []
    steps = []

    def lift(x):
        for pivot, row in reversed(steps):
            x = [
                -sum((v * y for v, y in zip(row, x)), F(0)) / pivot if pivot else F(0)
            ] + x
        return x

    for k in range(n):
        budget.tick(n * n)
        pivot = work[0][0]
        if pivot < 0:
            return None, lift([F(1)] + [F(0)] * (len(work) - 1))
        if pivot == 0:
            off = next((j for j in range(1, len(work)) if work[0][j]), None)
            if off is not None:
                magnitude = (abs(work[off][off]) + 1) / (2 * abs(work[0][off])) + 1
                x = [F(0)] * len(work)
                x[0] = -magnitude if work[0][off] > 0 else magnitude
                x[off] = 1
                return None, lift(x)
            D.append(F(0))
            steps.append((F(0), work[0][1:]))
            work = [row[1:] for row in work[1:]]
            continue
        D.append(pivot)
        row = work[0][1:]
        for j, v in enumerate(row):
            L[k + 1 + j][k] = budget.rational(v / pivot)
        steps.append((pivot, row))
        work = [
            [
                budget.rational(work[i][j] - work[i][0] * work[0][j] / pivot)
                for j in range(1, len(work))
            ]
            for i in range(1, len(work))
        ]
    return {"L": [[str(x) for x in row] for row in L], "D": [str(x) for x in D]}, None


def replay_ldl(a, cert, budget):
    strict(cert, ("L", "D"), ("L", "D"))
    n = len(a)
    if not isinstance(cert["L"], list) or len(cert["L"]) != n or len(cert["D"]) != n:
        raise Invalid("LDL dimensions mismatch")
    L = [[rational(v) for v in row] for row in cert["L"]]
    D = list(map(rational, cert["D"]))
    if any(len(row) != n for row in L) or any(d < 0 for d in D):
        return False
    if any(L[i][i] != 1 or any(L[i][j] for j in range(i + 1, n)) for i in range(n)):
        return False
    budget.tick(n * n * n)
    return (
        matrix_product(
            [[L[i][j] * D[j] for j in range(n)] for i in range(n)], transpose(L)
        )
        == a
    )


def poly_atoms(p):
    return sorted({a for m in p for a, _ in m})


def poly_expr(p, ctx):
    out = C(0)
    for mon, c in p.items():
        term = C(c)
        for name, power in mon:
            atom = ctx.atoms[name] if name.startswith("@") else S(name)
            term = E("mul", term, E("pow", atom, C(power)))
        out = E("add", out, term)
    return out


def rank(a):
    if not a:
        return 0
    a = copy.deepcopy(a)
    r = 0
    for j in range(len(a[0])):
        pivot = next((i for i in range(r, len(a)) if a[i][j]), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        value = a[r][j]
        a[r] = [v / value for v in a[r]]
        for i in range(len(a)):
            if i != r:
                value = a[i][j]
                a[i] = [v - value * w for v, w in zip(a[i], a[r])]
        r += 1
        if r == len(a):
            break
    return r


def sos_discover(p, ctx):
    # Direct even-monomial decomposition, then exact quadratic completion.
    if all(c >= 0 and all(n % 2 == 0 for _, n in m) for m, c in p.items()):
        return {
            "rule": "sos",
            "terms": [
                [str(c), pack({tuple((a, n // 2) for a, n in m if n): F(1)})]
                for m, c in p.items()
            ],
        }
    if not p or any(sum(n for _, n in m) > 2 for m in p):
        return None
    atoms = poly_atoms(p)
    if len(atoms) > 16:
        return None
    indices = {a: i + 1 for i, a in enumerate(atoms)}
    n = len(atoms) + 1
    Q = [[F(0) for _ in range(n)] for _ in range(n)]
    for m, c in p.items():
        variables = [indices[a] for a, n in m for _ in range(n)]
        if not variables:
            Q[0][0] += c
        elif len(variables) == 1:
            Q[0][variables[0]] += c / 2
            Q[variables[0]][0] += c / 2
        else:
            i, j = variables
            if i == j:
                Q[i][i] += c
            else:
                Q[i][j] += c / 2
                Q[j][i] += c / 2
    cert, _ = ldl(Q, ctx.budget)
    if cert is None:
        return None
    L = [[rational(x) for x in row] for row in cert["L"]]
    D = list(map(rational, cert["D"]))
    terms = []
    for j, d in enumerate(D):
        if not d:
            continue
        form = pc(L[0][j])
        for i, a in enumerate(atoms, 1):
            if L[i][j]:
                form[((a, 1),)] = L[i][j]
        terms.append([str(d), pack(form)])
    return {"rule": "sos", "terms": terms}


def sos_replay(p, proof, ctx):
    strict(proof, ("rule", "terms"), ("rule", "terms"))
    if (
        proof["rule"] != "sos"
        or not isinstance(proof["terms"], list)
        or len(proof["terms"]) > ctx.budget.limits.max_terms
    ):
        return False, False
    total = {}
    linear = []
    atoms = set()
    for item in proof["terms"]:
        if not isinstance(item, list) or len(item) != 2:
            raise Invalid("invalid SOS term")
        weight = rational(item[0])
        q = unpack(item[1], ctx.budget)
        if weight < 0:
            return False, False
        total = pa(total, pm(q, q, ctx.budget), ctx.budget, weight)
        if weight > 0 and all(sum(n for _, n in m) <= 1 for m in q):
            linear.append(q)
            atoms.update(poly_atoms(q))
    if total != p:
        return False, False
    atoms = sorted(atoms)
    coeffs = [[q.get(((a, 1),), F(0)) for a in atoms] for q in linear]
    augmented = [row + [-q.get((), F(0))] for row, q in zip(coeffs, linear)]
    ctx.budget.tick(max(1, len(linear)) * max(1, len(atoms)) ** 2)
    # Sum of positive weighted squares is strictly positive iff these proven
    # linear forms cannot simultaneously vanish; higher-degree forms are ignored.
    strict_positive = rank(augmented) > rank(coeffs) if linear else False
    return True, strict_positive


def assumed_sign(e, ctx):
    out = set()
    for a, op, b in ctx.assumptions:
        if b.op == "const" and b.value == 0 and a == e:
            out.update({"nonnegative"} if op in (">=", ">", "==") else ())
            out.update(
                {"positive", "nonzero"}
                if op == ">"
                else {"nonzero"} if op == "!=" else ()
            )
        if a.op == "const" and a.value == 0 and b == e:
            out.update({"nonnegative"} if op in ("<=", "<", "==") else ())
            out.update(
                {"positive", "nonzero"}
                if op == "<"
                else {"nonzero"} if op == "!=" else ()
            )
    return out


def sign_proof(e, need, ctx):
    if need in assumed_sign(e, ctx):
        return {"rule": "assumption", "need": need}
    b = interval(e, ctx)
    if (
        (need == "positive" and b.positive)
        or (need == "nonnegative" and b.nonnegative)
        or (need == "nonzero" and b.nonzero)
    ):
        return {"rule": "interval", "need": need, "bounds": b.data()}
    if e.op == "pow" and int(e.args[1].value) > 0 and int(e.args[1].value) % 2 == 0:
        if need == "nonnegative":
            return {"rule": "even_power", "need": need}
        nz = sign_proof(e.args[0], "nonzero", ctx)
        if nz:
            return {"rule": "even_power", "need": need, "base": nz}
    try:
        r = normalize(e, ctx)
        if r.d != pc(1):
            return None
        p = r.n
        if need == "nonzero":
            for sign in (1, -1):
                q = {m: sign * c for m, c in p.items()}
                proof = sos_discover(q, ctx)
                if proof and sos_replay(q, proof, ctx)[1]:
                    return {
                        "rule": "polynomial",
                        "need": need,
                        "sign": sign,
                        "sos": proof,
                    }
        else:
            proof = sos_discover(p, ctx)
            if proof:
                ok, positive = sos_replay(p, proof, ctx)
                if ok and (need == "nonnegative" or positive):
                    return {"rule": "polynomial", "need": need, "sign": 1, "sos": proof}
        # Extract a common monomial. Each factor receives its own sign proof,
        # and replay re-multiplies them exactly. This handles e.g. eta*(1-s)
        # under eta>=0 and 0<=s<=1 without sampling a dependent expression.
        if len(p) > 1:
            common = dict(next(iter(p)))
            for mon in p:
                powers = dict(mon)
                common = {
                    name: min(n, powers.get(name, 0))
                    for name, n in common.items()
                    if powers.get(name, 0)
                }
            if common:
                factor = {tuple(sorted(common.items())): F(1)}
                rest = {
                    tuple(
                        (name, n - common.get(name, 0))
                        for name, n in mon
                        if n > common.get(name, 0)
                    ): c
                    for mon, c in p.items()
                }
                for sgn in (1, -1):
                    factors = [
                        {mon: sgn * c for mon, c in f.items()} for f in (factor, rest)
                    ]
                    children = [
                        sign_proof(poly_expr(f, ctx), need, ctx) for f in factors
                    ]
                    if all(children):
                        return {
                            "rule": "product",
                            "need": need,
                            "factors": [pack(f) for f in factors],
                            "children": children,
                        }
    except (Unsupported, NonExact, Undefined):
        pass
    return None


def replay_sign(e, need, proof, ctx):
    if not isinstance(proof, dict) or proof.get("need") != need:
        return False
    rule = proof.get("rule")
    if rule == "assumption":
        return need in assumed_sign(e, ctx)
    if rule == "interval":
        b = interval(e, ctx)
        return proof.get("bounds") == b.data() and (
            (need == "positive" and b.positive)
            or (need == "nonnegative" and b.nonnegative)
            or (need == "nonzero" and b.nonzero)
        )
    if (
        rule == "even_power"
        and e.op == "pow"
        and int(e.args[1].value) > 0
        and int(e.args[1].value) % 2 == 0
    ):
        return need == "nonnegative" or replay_sign(
            e.args[0], "nonzero", proof.get("base"), ctx
        )
    if rule == "polynomial":
        r = normalize(e, ctx)
        sgn = proof.get("sign")
        if (
            r.d != pc(1)
            or type(sgn) is not int
            or sgn not in (-1, 1)
            or (need != "nonzero" and sgn != 1)
        ):
            return False
        ok, pos = sos_replay(
            {m: sgn * c for m, c in r.n.items()}, proof.get("sos"), ctx
        )
        return ok and (need == "nonnegative" or pos)
    if rule == "product":
        factors = proof.get("factors")
        children = proof.get("children")
        if (
            not isinstance(factors, list)
            or len(factors) != 2
            or not isinstance(children, list)
            or len(children) != 2
        ):
            return False
        parts = [unpack(f, ctx.budget) for f in factors]
        r = normalize(e, ctx)
        if r.d != pc(1) or pm(parts[0], parts[1], ctx.budget) != r.n:
            return False
        return all(
            replay_sign(poly_expr(f, ctx), need, c, ctx)
            for f, c in zip(parts, children)
        )
    return False


def domain_requirements(e, differentiable=False):
    out = []
    for a in e.args:
        out += domain_requirements(a, differentiable)
    if e.op == "div":
        out.append((e.args[1], "nonzero"))
    if e.op == "pow" and e.args[1].value < 0:
        out.append((e.args[0], "nonzero"))
    if e.op == "log":
        out.append((e.args[0], "positive"))
    if e.op == "sqrt":
        out.append((e.args[0], "positive" if differentiable else "nonnegative"))
    if e.op in ("gcd", "binomial"):
        raise Unsupported("integer functions require explicit finite enumeration")
    return out


def domain_discover(expressions, ctx, differentiable=False):
    required = {}
    for e in expressions:
        for arg, need in domain_requirements(e, differentiable):
            required[(arg.key, need)] = (arg, need)
    proofs = []
    for arg, need in required.values():
        proof = sign_proof(arg, need, ctx)
        if proof is None:
            return None
        proofs.append({"expression": arg.data(), "need": need, "proof": proof})
    return proofs


def replay_domain(expressions, proofs, ctx, differentiable=False):
    required = {}
    for e in expressions:
        for arg, need in domain_requirements(e, differentiable):
            required[(arg.key, need)] = (arg, need)
    if not isinstance(proofs, list) or len(proofs) != len(required):
        return False
    found = set()
    for item in proofs:
        if not isinstance(item, dict):
            return False
        key = (canonical(item.get("expression")), item.get("need"))
        if key not in required or key in found:
            return False
        arg, need = required[key]
        if not replay_sign(arg, need, item.get("proof"), ctx):
            return False
        found.add(key)
    return True


def simplify_derivative(e, ctx):
    """Local algebra for a derived formula; original definedness stays required."""
    ctx.budget.tick()
    args = tuple(simplify_derivative(a, ctx) for a in e.args)
    e = Expr(e.op, args, e.value)
    if not args:
        return e
    if all(a.op == "const" for a in args):
        try:
            return C(exact(e, {}, ctx.budget))
        except (NonExact, Undefined):
            pass
    zero = lambda a: a.op == "const" and a.value == 0
    one = lambda a: a.op == "const" and a.value == 1
    if e.op == "add":
        if zero(args[0]):
            return args[1]
        if zero(args[1]):
            return args[0]
    if e.op == "sub" and zero(args[1]):
        return args[0]
    if e.op == "mul":
        if any(zero(a) for a in args):
            return C(0)
        if one(args[0]):
            return args[1]
        if one(args[1]):
            return args[0]
    if e.op == "div" and one(args[1]):
        return args[0]
    if e.op == "pow":
        if zero(args[1]):
            return C(1)
        if one(args[1]):
            return args[0]
    return e


def differentiate(e, var, ctx):
    return simplify_derivative(_differentiate(e, var, ctx), ctx)


def _differentiate(e, var, ctx):
    ctx.budget.tick()
    d = lambda a: differentiate(a, var, ctx)
    if e.op == "const" or e.op == "pi":
        return C(0)
    if e.op == "symbol":
        return C(int(e.value == var))
    a = e.args[0]
    if e.op == "neg":
        return E("neg", d(a))
    if e.op in ("add", "sub"):
        return E(e.op, d(a), d(e.args[1]))
    if e.op == "mul":
        return E("add", E("mul", d(a), e.args[1]), E("mul", a, d(e.args[1])))
    if e.op == "div":
        return E(
            "div",
            E("sub", E("mul", d(a), e.args[1]), E("mul", a, d(e.args[1]))),
            E("pow", e.args[1], C(2)),
        )
    if e.op == "pow":
        n = int(e.args[1].value)
        if not n:
            return C(0)
        return E("mul", E("mul", C(n), E("pow", a, C(n - 1))), d(a))
    if e.op == "exp":
        return E("mul", e, d(a))
    if e.op == "log":
        return E("div", d(a), a)
    if e.op == "sqrt":
        return E("div", d(a), E("mul", C(2), E("sqrt", a)))
    if e.op == "sin":
        return E("mul", E("cos", a), d(a))
    if e.op == "cos":
        return E("neg", E("mul", E("sin", a), d(a)))
    if e.op == "abs":
        b = interval(a, ctx)
        if b.positive:
            return d(a)
        if b.negative:
            return E("neg", d(a))
    raise Unsupported("no differentiability proof rule for " + e.op)


def sample_points(ctx):
    names = list(ctx.bounds)
    choices = {}
    for name, b in ctx.bounds.items():
        candidates = [
            F(0),
            F(1),
            F(-1),
            F(1, 2),
            F(-1, 2),
            F(2),
            F(-2),
            F(3),
            F(-3),
            F(10),
            F(-10),
        ]
        if b.lo is not None:
            candidates += [b.lo, b.lo + 1, b.lo + F(1, 2)]
        if b.hi is not None:
            candidates += [b.hi, b.hi - 1, b.hi - F(1, 2)]
        if b.lo is not None and b.hi is not None:
            candidates += [(b.lo + b.hi) / 2]
        candidates = list(
            dict.fromkeys(
                x
                for x in candidates
                if b.contains(x)
                and (ctx.domains[name] != "integer" or x.denominator == 1)
            )
        )
        if not candidates:
            return
        choices[name] = candidates
    seen = set()
    count = 0

    def add(values):
        nonlocal count
        key = tuple(values)
        if key in seen:
            return None
        seen.add(key)
        count += 1
        return dict(zip(names, values))

    if not names:
        yield {}
        return
    bases = [choices[n][0] for n in names]
    p = add(bases)
    if p is not None:
        yield p
    # Multi-coordinate witnesses precede axis-only scans. Matrix identities and
    # interactions such as xy=0 cannot be refuted by changing one zero at a time.
    rng = random.Random(1729)
    for j in range(min(24, ctx.budget.limits.counterexample_trials - 1)):
        values = [
            choices[n][(j + 1) % len(choices[n])] if j < 3 else rng.choice(choices[n])
            for n in names
        ]
        p = add(values)
        if p is not None:
            yield p
        if count >= ctx.budget.limits.counterexample_trials:
            return
    for i, name in enumerate(names):
        for value in choices[name][1:]:
            row = list(bases)
            row[i] = value
            p = add(row)
            if p is not None:
                yield p
            if count >= ctx.budget.limits.counterexample_trials:
                return
    for values in product(*(choices[n] for n in names)):
        p = add(values)
        if p is not None:
            yield p
        if count >= ctx.budget.limits.counterexample_trials:
            return


def finite_points(ctx):
    values = []
    count = 1
    for name, b in ctx.bounds.items():
        if ctx.domains[name] != "integer" or b.lo is None or b.hi is None:
            raise Invalid("finite integer boxes require explicit finite bounds")
        low = math.ceil(b.lo)
        high = math.floor(b.hi)
        if b.lo_open and F(low) == b.lo:
            low += 1
        if b.hi_open and F(high) == b.hi:
            high -= 1
        count *= max(0, high - low + 1)
        if count > ctx.budget.limits.max_enumeration:
            raise Limit("finite enumeration budget exceeded")
        values.append(range(low, high + 1))
    for xs in product(*values):
        ctx.budget.tick()
        p = dict(zip(ctx.bounds, map(F, xs)))
        if ctx.allowed(p):
            yield p


def _scalar_data(claim, budget):
    strict(
        claim,
        ("kind", "variables", "lhs", "op", "rhs", "assumptions"),
        ("kind", "variables", "lhs", "op", "rhs"),
    )
    if claim["op"] not in OPS:
        raise Invalid("unsupported relation")
    ctx = Context(claim["variables"], claim.get("assumptions", []), budget)
    return (
        ctx,
        parse(claim["lhs"], ctx.bounds, budget),
        OPS[claim["op"]],
        parse(claim["rhs"], ctx.bounds, budget),
    )


def scalar_proof(claim, budget, search=True):
    ctx, a, op, b = _scalar_data(claim, budget)
    if ctx.vacant():
        return "VACUOUS", {"method": "empty_domain"}, None
    if claim["kind"] == "finite":
        checked = 0
        values = []
        for point in finite_points(ctx):
            checked += 1
            try:
                ok = relation(exact(a, point, budget), op, exact(b, point, budget))
            except Undefined:
                return (
                    "ILL_DEFINED",
                    {
                        "method": "witness",
                        "point": {k: str(v) for k, v in point.items()},
                        "undefined": True,
                    },
                    point,
                )
            if not ok:
                return (
                    "DISPROVEN",
                    {
                        "method": "witness",
                        "point": {k: str(v) for k, v in point.items()},
                        "undefined": False,
                    },
                    point,
                )
            values.append([str(point[k]) for k in ctx.bounds])
        if not checked:
            return "VACUOUS", {"method": "empty_finite"}, None
        return (
            "VERIFIED_FINITE",
            {"method": "finite", "checked": checked, "point_digest": digest(values)},
            None,
        )
    # Refutation is exact, deterministic, budgeted and never used as proof.
    if search:
        for point in sample_points(ctx):
            budget.tick()
            try:
                if not ctx.allowed(point):
                    continue
            except (NonExact, Undefined):
                continue
            try:
                ok = relation(exact(a, point, budget), op, exact(b, point, budget))
            except Undefined:
                return (
                    "ILL_DEFINED",
                    {
                        "method": "witness",
                        "point": {k: str(v) for k, v in point.items()},
                        "undefined": True,
                    },
                    point,
                )
            except NonExact:
                continue
            if not ok:
                return (
                    "DISPROVEN",
                    {
                        "method": "witness",
                        "point": {k: str(v) for k, v in point.items()},
                        "undefined": False,
                    },
                    point,
                )
    expressions = [a, b] + [x for lhs, _, rhs in ctx.assumptions for x in (lhs, rhs)]
    domains = domain_discover(expressions, ctx)
    if domains is None:
        raise Unsupported("definedness/domain obligations remain unresolved")
    difference = E("sub", b, a) if op in ("<=", "<") else E("sub", a, b)
    r = normalize(difference, ctx)
    if not r.n and op in ("==", "<=", ">="):
        return "VERIFIED", {"method": "identity", "domain": domains}, None
    if op == "==":
        raise Unsupported("identity did not reduce to exact zero; no witness found")
    if op in ("!=",):
        proof = sign_proof(difference, "nonzero", ctx)
        if proof:
            return (
                "VERIFIED",
                {
                    "method": "relation_sign",
                    "domain": domains,
                    "need": "nonzero",
                    "proof": proof,
                },
                None,
            )
        raise Unsupported("nonzero proof unavailable")
    need = "positive" if op in ("<", ">") else "nonnegative"
    proof = sign_proof(difference, need, ctx)
    if proof:
        return (
            "VERIFIED",
            {
                "method": "relation_sign",
                "domain": domains,
                "need": need,
                "proof": proof,
            },
            None,
        )
    # For rational inequalities a certified denominator sign permits reduction
    # to a polynomial numerator. No denominator is cancelled before domain checks.
    denexpr = poly_expr(r.d, ctx)
    for sign in (1, -1):
        dp = sign_proof(denexpr if sign == 1 else E("neg", denexpr), "positive", ctx)
        if dp:
            p = {m: sign * c for m, c in r.n.items()}
            sos = sos_discover(p, ctx)
            if sos:
                ok, pos = sos_replay(p, sos, ctx)
                if ok and (need == "nonnegative" or pos):
                    return (
                        "VERIFIED",
                        {
                            "method": "rational_sos",
                            "domain": domains,
                            "den_sign": sign,
                            "den_proof": dp,
                            "sos": sos,
                        },
                        None,
                    )
    raise Unsupported(
        "no exact inequality certificate found; finite successes are not proof"
    )


# Fixed-dimension matrix expressions reduce to independently replayed scalar
# polynomial obligations; dimensions are part of the claim, never guessed.
def matrix_data(claim, budget):
    strict(
        claim, ("kind", "matrices", "lhs", "rhs"), ("kind", "matrices", "lhs", "rhs")
    )
    if not isinstance(claim["matrices"], dict) or not claim["matrices"]:
        raise Invalid("matrix declarations required")
    matrices = {}
    variables = {}
    for name, shape in claim["matrices"].items():
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in RESERVED:
            raise Invalid("invalid matrix name")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                type(n) is not int or not 1 <= n <= budget.limits.max_matrix_size
                for n in shape
            )
        ):
            raise Limit("matrix shape limit")
        entries = []
        for i in range(shape[0]):
            row = []
            for j in range(shape[1]):
                var = f"{name}_{i}_{j}"
                variables[var] = {}
                row.append(S(var))
            entries.append(row)
        matrices[name] = entries
    if len(variables) > budget.limits.max_variables:
        raise Limit("matrix scalar-variable budget exceeded")

    def matrix(x):
        return isinstance(x, list)

    def shape(x):
        return (len(x), len(x[0]))

    def visit(n, source):
        budget.tick()
        if isinstance(n, ast.Name) and n.id in matrices:
            return matrices[n.id]
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return C(rational(ast.get_source_segment(source, n)))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            a = visit(n.operand, source)
            return (
                [[E("neg", v) for v in row] for row in a] if matrix(a) else E("neg", a)
            )
        if isinstance(n, ast.BinOp):
            a, b = visit(n.left, source), visit(n.right, source)
            if isinstance(n.op, (ast.Add, ast.Sub)):
                op = "add" if isinstance(n.op, ast.Add) else "sub"
                if matrix(a) != matrix(b):
                    raise Invalid("cannot add a matrix and scalar")
                if not matrix(a):
                    return E(op, a, b)
                if shape(a) != shape(b):
                    raise Invalid("matrix addition shape mismatch")
                return [[E(op, x, y) for x, y in zip(ar, br)] for ar, br in zip(a, b)]
            if isinstance(n.op, (ast.Mult, ast.MatMult)):
                if not matrix(a) and not matrix(b):
                    return E("mul", a, b)
                if matrix(a) and matrix(b):
                    if len(a[0]) != len(b):
                        raise Invalid("matrix multiplication shape mismatch")
                    return [
                        [
                            sum_expr(
                                [E("mul", a[i][k], b[k][j]) for k in range(len(b))]
                            )
                            for j in range(len(b[0]))
                        ]
                        for i in range(len(a))
                    ]
                arr, scalar = (a, b) if matrix(a) else (b, a)
                return [[E("mul", scalar, x) for x in row] for row in arr]
            raise Invalid("unsupported matrix operator")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and not n.keywords:
            name = n.func.id
            if (
                name not in ("trace", "det", "transpose", "frobenius2")
                or len(n.args) != 1
            ):
                raise Invalid("unsupported matrix function")
            a = visit(n.args[0], source)
            if not matrix(a):
                raise Invalid("matrix argument required")
            if name == "transpose":
                return transpose(a)
            if name == "frobenius2":
                return sum_expr([E("pow", v, C(2)) for row in a for v in row])
            if len(a) != len(a[0]):
                raise Invalid("trace/determinant require square matrices")
            if name == "trace":
                return sum_expr([a[i][i] for i in range(len(a))])
            terms = []
            for perm in permutations(range(len(a))):
                budget.tick()
                sign = (-1) ** sum(
                    perm[i] > perm[j]
                    for i in range(len(a))
                    for j in range(i + 1, len(a))
                )
                term = C(sign)
                for i, j in enumerate(perm):
                    term = E("mul", term, a[i][j])
                terms.append(term)
            return sum_expr(terms)
        raise Invalid("unsafe or unsupported matrix expression")

    outputs = []
    for key in ("lhs", "rhs"):
        text = claim[key]
        if not isinstance(text, str) or len(text) > budget.limits.max_expression_chars:
            raise Invalid("matrix expression limit")
        try:
            t = ast.parse(text, mode="eval")
        except SyntaxError as e:
            raise Invalid("invalid matrix syntax") from e
        if sum(1 for _ in ast.walk(t)) > budget.limits.max_nodes:
            raise Limit("matrix syntax node limit")
        outputs.append(visit(t.body, text))
    a, b = outputs
    if matrix(a) != matrix(b) or (matrix(a) and shape(a) != shape(b)):
        raise Invalid("matrix equality type/shape mismatch")
    if not matrix(a):
        a, b = [[a]], [[b]]
    pairs = [(x, y) for ar, br in zip(a, b) for x, y in zip(ar, br)]
    return variables, pairs


def sum_expr(items):
    out = C(0)
    for e in items:
        out = E("add", out, e)
    return out


def logic_data(claim, budget):
    strict(claim, ("kind", "formula"), ("kind", "formula"))
    source = claim["formula"]
    if not isinstance(source, str) or len(source) > budget.limits.max_expression_chars:
        raise Invalid("bounded Boolean formula required")
    try:
        t = ast.parse(source, mode="eval")
    except SyntaxError as e:
        raise Invalid("invalid Boolean formula") from e
    if sum(1 for _ in ast.walk(t)) > budget.limits.max_nodes:
        raise Limit("Boolean formula too large")
    names = set()

    def validate(n):
        budget.tick()
        if (
            isinstance(n, ast.Name)
            and NAME.fullmatch(n.id)
            and n.id not in ("implies", "iff")
        ):
            names.add(n.id)
            return
        if isinstance(n, ast.Constant) and type(n.value) is bool:
            return
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            validate(n.operand)
            return
        if isinstance(n, ast.BoolOp) and isinstance(n.op, (ast.And, ast.Or)):
            for v in n.values:
                validate(v)
            return
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in ("implies", "iff")
            and len(n.args) == 2
            and not n.keywords
        ):
            for v in n.args:
                validate(v)
            return
        if (
            isinstance(n, ast.Compare)
            and len(n.ops) == 1
            and isinstance(n.ops[0], (ast.Eq, ast.NotEq))
        ):
            validate(n.left)
            validate(n.comparators[0])
            return
        raise Invalid("unsupported Boolean construct")

    validate(t.body)
    if 2 ** len(names) > budget.limits.max_enumeration:
        raise Limit("truth table budget exceeded")

    def evaluate(n, p):
        if isinstance(n, ast.Name):
            return p[n.id]
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.UnaryOp):
            return not evaluate(n.operand, p)
        if isinstance(n, ast.BoolOp):
            values = [evaluate(v, p) for v in n.values]
            return all(values) if isinstance(n.op, ast.And) else any(values)
        if isinstance(n, ast.Call):
            a, b = (evaluate(v, p) for v in n.args)
            return (not a or b) if n.func.id == "implies" else a == b
        a, b = evaluate(n.left, p), evaluate(n.comparators[0], p)
        return a == b if isinstance(n.ops[0], ast.Eq) else a != b

    return sorted(names), lambda p: evaluate(t.body, p)


def replace_symbol(e, name, value):
    if e.op == "symbol" and e.value == name:
        return value
    return Expr(e.op, tuple(replace_symbol(a, name, value) for a in e.args), e.value)


def derivative_data(claim, budget):
    strict(
        claim,
        ("kind", "variables", "expression", "variable", "proposed", "assumptions"),
        ("kind", "variables", "expression", "variable", "proposed"),
    )
    ctx = Context(claim["variables"], claim.get("assumptions", []), budget)
    var = claim["variable"]
    if var not in ctx.bounds or ctx.domains[var] != "real":
        raise Invalid("derivative variable must be declared real")
    expression = parse(claim["expression"], ctx.bounds, budget)
    proposed = parse(claim["proposed"], ctx.bounds, budget)
    actual = differentiate(expression, var, ctx)
    return ctx, expression, actual, proposed


def expr_text(e):
    # Exact safe source printer, also used to derive child certificate claims.
    # An iterative bounded printer also caps expansion of shared matrix trees.
    pending = [e]
    parts = []
    length = 0
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            length += len(item)
            if length > Limits().max_expression_chars:
                raise Limit("derived expression length budget exceeded")
            parts.append(item)
            continue
        if item.op == "const":
            pending.append(f"({item.value.numerator}/{item.value.denominator})")
        elif item.op == "symbol":
            pending.append(item.value)
        elif item.op == "pi":
            pending.append("pi")
        elif item.op == "neg":
            pending.extend([")", item.args[0], "(-"])
        elif item.op in ("add", "sub", "mul", "div", "pow"):
            op = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**"}[item.op]
            pending.extend([")", item.args[1], op, item.args[0], "("])
        else:
            sequence = [item.op, "("]
            for i, arg in enumerate(item.args):
                if i:
                    sequence.append(",")
                sequence.append(arg)
            sequence.append(")")
            pending.extend(reversed(sequence))
    return "".join(parts)


def child_scalar(variables, a, b, assumptions=()):
    return {
        "kind": "scalar",
        "variables": variables,
        "lhs": expr_text(a),
        "op": "==",
        "rhs": expr_text(b),
        "assumptions": list(assumptions),
    }


def ode_data(claim, budget):
    strict(
        claim,
        ("kind", "variables", "variable", "solution", "rhs", "initial", "assumptions"),
        ("kind", "variables", "variable", "solution", "rhs", "initial"),
    )
    ctx = Context(claim["variables"], claim.get("assumptions", []), budget)
    var = claim["variable"]
    if set(ctx.bounds) != {var} or ctx.domains[var] != "real" or var == "y":
        raise Invalid("one real independent variable (not y) required")
    solution = parse(claim["solution"], ctx.bounds, budget)
    rhs = replace_symbol(
        parse(claim["rhs"], set(ctx.bounds) | {"y"}, budget), "y", solution
    )
    actual = differentiate(solution, var, ctx)
    initial = claim["initial"]
    strict(initial, ("at", "value"), ("at", "value"))
    at = rational(initial["at"])
    value = parse(initial["value"], (), budget)
    if not ctx.allowed({var: at}):
        raise Invalid("initial point is outside the declared domain/assumptions")
    ic = replace_symbol(solution, var, C(at))
    return ctx, solution, actual, rhs, ic, value


def psd_data(claim, budget):
    strict(claim, ("kind", "matrix"), ("kind", "matrix"))
    raw = claim["matrix"]
    if (
        not isinstance(raw, list)
        or not 1 <= len(raw) <= 16
        or any(not isinstance(row, list) or len(row) != len(raw) for row in raw)
    ):
        raise Invalid("bounded square rational matrix required")
    a = [[budget.rational(rational(v)) for v in row] for row in raw]
    if a != transpose(a):
        raise Invalid("PSD certificate API requires a symmetric matrix")
    return a


@dataclass
class Verdict:
    status: str
    scope: str
    reasons: list = field(default_factory=list)
    certificate: dict | None = None
    counterexample: dict | list | None = None
    subclaims: list = field(default_factory=list)
    family: list = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self):
        return asdict(self)

    @property
    def score(self):
        return None  # A proof is not a learned truth probability.


def _certificate(claim, status, payload):
    return {
        "version": VERSION,
        "claim_hash": digest(claim),
        "status": status,
        "payload": payload,
    }


def _prove(claim, budget):
    if not isinstance(claim, dict):
        raise Invalid("typed claim dictionary required")
    kind = claim.get("kind")
    if kind in ("scalar", "finite"):
        status, payload, witness = scalar_proof(claim, budget)
        return (
            status,
            payload,
            None if witness is None else {k: str(v) for k, v in witness.items()},
        )
    if kind == "logic":
        names, evaluate = logic_data(claim, budget)
        bits = []
        for values in product((False, True), repeat=len(names)):
            budget.tick()
            point = dict(zip(names, values))
            ok = evaluate(point)
            if not ok:
                return "DISPROVEN", {"method": "logic_witness", "point": point}, point
            bits.append(int(ok))
        return (
            "VERIFIED_FINITE",
            {"method": "logic", "valuations": len(bits), "table_digest": digest(bits)},
            None,
        )
    if kind == "psd":
        a = psd_data(claim, budget)
        proof, witness = ldl(a, budget)
        if proof:
            return "VERIFIED", {"method": "ldl", "decomposition": proof}, None
        witness = list(map(str, witness))
        return "DISPROVEN", {"method": "negative_quadratic", "vector": witness}, witness
    if kind == "matrix_identity":
        variables, pairs = matrix_data(claim, budget)
        children = []
        for a, b in pairs:
            c = child_scalar(variables, a, b)
            status, payload, witness = _prove(c, budget)
            if status == "DISPROVEN":
                return (
                    "DISPROVEN",
                    {"method": "matrix_witness", "point": witness},
                    witness,
                )
            if status != "VERIFIED":
                raise Unsupported("unresolved matrix entry")
            children.append(_certificate(c, status, payload))
        return "VERIFIED", {"method": "matrix_entries", "children": children}, None
    if kind == "derivative":
        ctx, expression, actual, proposed = derivative_data(claim, budget)
        if ctx.vacant():
            return "VACUOUS", {"method": "derivative_empty"}, None
        original = domain_discover([expression], ctx, differentiable=True)
        if original is None:
            raise Unsupported("original function domain is not certified")
        child = child_scalar(
            claim["variables"], actual, proposed, claim.get("assumptions", [])
        )
        status, payload, witness = _prove(child, budget)
        if status in ("DISPROVEN", "ILL_DEFINED"):
            return (
                status,
                {
                    "method": "derivative_child",
                    "original_domain": original,
                    "child": _certificate(child, status, payload),
                },
                witness,
            )
        original_domain = domain_discover([expression, actual, proposed], ctx)
        if status != "VERIFIED" or original_domain is None:
            raise Unsupported("derivative/domain proof incomplete")
        return (
            "VERIFIED",
            {
                "method": "derivative_child",
                "domain": original_domain,
                "original_domain": original,
                "child": _certificate(child, status, payload),
            },
            None,
        )
    if kind == "ode":
        ctx, solution, actual, rhs, ic, value = ode_data(claim, budget)
        if ctx.vacant():
            return "VACUOUS", {"method": "ode_empty"}, None
        original = domain_discover([solution], ctx, differentiable=True)
        if original is None:
            raise Unsupported("original ODE solution domain is not certified")
        children = []
        for a, b, variables, assumptions in (
            (actual, rhs, claim["variables"], claim.get("assumptions", [])),
            (ic, value, {}, []),
        ):
            child = child_scalar(variables, a, b, assumptions)
            status, payload, witness = _prove(child, budget)
            children.append(_certificate(child, status, payload))
            if status in ("DISPROVEN", "ILL_DEFINED"):
                return (
                    status,
                    {
                        "method": "ode_children",
                        "children": children,
                        "original_domain": original,
                    },
                    witness,
                )
            if status != "VERIFIED":
                raise Unsupported("ODE equality or initial-condition check incomplete")
        dom = domain_discover([solution, actual, rhs], ctx)
        if dom is None:
            raise Unsupported(
                "solution not proved defined/differentiable on entire declared domain"
            )
        return (
            "VERIFIED",
            {
                "method": "ode_children",
                "domain": dom,
                "children": children,
                "original_domain": original,
            },
            None,
        )
    raise Unsupported("no mathematical verification backend for this claim kind")


# Public replay never calls the proof search and never trusts a verdict label.
def verify_certificate(claim, certificate, limits=None):
    try:
        budget = Budget(limits)
        if len(canonical(certificate).encode()) > budget.limits.max_payload_bytes:
            return False
        return _replay(claim, certificate, budget)
    except (
        Invalid,
        Unsupported,
        Limit,
        Undefined,
        NonExact,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        ZeroDivisionError,
        OverflowError,
        RecursionError,
    ):
        return False


def _replay(claim, cert, budget):
    budget.tick()
    strict(
        cert,
        ("version", "claim_hash", "status", "payload"),
        ("version", "claim_hash", "status", "payload"),
    )
    if cert["version"] != VERSION or cert["claim_hash"] != digest(claim):
        return False
    status, p = cert["status"], cert["payload"]
    if status not in (
        "VERIFIED",
        "VERIFIED_FINITE",
        "DISPROVEN",
        "ILL_DEFINED",
        "VACUOUS",
    ) or not isinstance(p, dict):
        return False
    method = p.get("method")
    kind = claim.get("kind")
    if kind in ("scalar", "finite"):
        ctx, a, op, b = _scalar_data(claim, budget)
        if method == "empty_domain":
            return status == "VACUOUS" and ctx.vacant()
        if method in ("finite", "empty_finite"):
            if kind != "finite":
                return False
            points = []
            for point in finite_points(ctx):
                if not relation(exact(a, point, budget), op, exact(b, point, budget)):
                    return False
                points.append([str(point[k]) for k in ctx.bounds])
            if method == "empty_finite":
                return not points and status == "VACUOUS"
            return (
                bool(points)
                and status == "VERIFIED_FINITE"
                and p.get("checked") == len(points)
                and p.get("point_digest") == digest(points)
            )
        if method == "witness":
            raw = p.get("point")
            if not isinstance(raw, dict):
                return False
            point = {k: rational(v) for k, v in raw.items()}
            if not ctx.allowed(point):
                return False
            try:
                ok = relation(exact(a, point, budget), op, exact(b, point, budget))
            except Undefined:
                return status == "ILL_DEFINED" and p.get("undefined") is True
            return not ok and status == "DISPROVEN" and p.get("undefined") is False
        if status != "VERIFIED" or kind != "scalar" or ctx.vacant():
            return False
        expressions = [a, b] + [e for l, _, r in ctx.assumptions for e in (l, r)]
        if not replay_domain(expressions, p.get("domain"), ctx):
            return False
        difference = E("sub", b, a) if op in ("<", "<=") else E("sub", a, b)
        if method == "identity":
            return op in ("==", "<=", ">=") and not normalize(difference, ctx).n
        need = (
            "nonzero"
            if op == "!="
            else "positive" if op in ("<", ">") else "nonnegative"
        )
        if method == "relation_sign":
            return (
                op != "=="
                and p.get("need") == need
                and replay_sign(difference, need, p.get("proof"), ctx)
            )
        if method == "rational_sos":
            if op not in ("<", ">", "<=", ">="):
                return False
            r = normalize(difference, ctx)
            sign = p.get("den_sign")
            if type(sign) is not int or sign not in (-1, 1):
                return False
            de = poly_expr(r.d, ctx)
            if not replay_sign(
                de if sign == 1 else E("neg", de), "positive", p.get("den_proof"), ctx
            ):
                return False
            ok, positive = sos_replay(
                {m: sign * c for m, c in r.n.items()}, p.get("sos"), ctx
            )
            return ok and (need == "nonnegative" or positive)
        return False
    if kind == "logic":
        names, evaluate = logic_data(claim, budget)
        if method == "logic_witness":
            point = p.get("point")
            return (
                status == "DISPROVEN"
                and isinstance(point, dict)
                and set(point) == set(names)
                and all(type(v) is bool for v in point.values())
                and not evaluate(point)
            )
        if method != "logic" or status != "VERIFIED_FINITE":
            return False
        bits = []
        for values in product((False, True), repeat=len(names)):
            budget.tick()
            ok = evaluate(dict(zip(names, values)))
            if not ok:
                return False
            bits.append(int(ok))
        return p.get("valuations") == len(bits) and p.get("table_digest") == digest(
            bits
        )
    if kind == "psd":
        a = psd_data(claim, budget)
        if method == "ldl":
            return status == "VERIFIED" and replay_ldl(
                a, p.get("decomposition"), budget
            )
        if method == "negative_quadratic":
            x = list(map(rational, p.get("vector", [])))
            return (
                status == "DISPROVEN"
                and len(x) == len(a)
                and sum(
                    (
                        x[i] * a[i][j] * x[j]
                        for i in range(len(a))
                        for j in range(len(a))
                    ),
                    F(0),
                )
                < 0
            )
        return False
    if kind == "matrix_identity":
        variables, pairs = matrix_data(claim, budget)
        if method == "matrix_witness":
            point = {k: rational(v) for k, v in p.get("point", {}).items()}
            if status != "DISPROVEN" or set(point) != set(variables):
                return False
            return any(
                exact(a, point, budget) != exact(b, point, budget) for a, b in pairs
            )
        children = p.get("children")
        return (
            method == "matrix_entries"
            and status == "VERIFIED"
            and isinstance(children, list)
            and len(children) == len(pairs)
            and all(
                _replay(child_scalar(variables, a, b), c, budget)
                and c["status"] == "VERIFIED"
                for (a, b), c in zip(pairs, children)
            )
        )
    if kind == "derivative":
        ctx, expression, actual, proposed = derivative_data(claim, budget)
        if method == "derivative_empty":
            return status == "VACUOUS" and ctx.vacant()
        if method != "derivative_child":
            return False
        if not replay_domain(
            [expression], p.get("original_domain"), ctx, differentiable=True
        ):
            return False
        child = child_scalar(
            claim["variables"], actual, proposed, claim.get("assumptions", [])
        )
        c = p.get("child")
        if not _replay(child, c, budget) or c["status"] != status:
            return False
        if status == "VERIFIED":
            return replay_domain([expression, actual, proposed], p.get("domain"), ctx)
        return status in ("DISPROVEN", "ILL_DEFINED")
    if kind == "ode":
        ctx, solution, actual, rhs, ic, value = ode_data(claim, budget)
        if method == "ode_empty":
            return status == "VACUOUS" and ctx.vacant()
        if not replay_domain(
            [solution], p.get("original_domain"), ctx, differentiable=True
        ):
            return False
        children = p.get("children")
        if (
            method != "ode_children"
            or not isinstance(children, list)
            or not 1 <= len(children) <= 2
        ):
            return False
        claims = [
            child_scalar(claim["variables"], actual, rhs, claim.get("assumptions", [])),
            child_scalar({}, ic, value),
        ]
        if not all(_replay(c, proof, budget) for c, proof in zip(claims, children)):
            return False
        if status in ("DISPROVEN", "ILL_DEFINED"):
            return children[-1]["status"] == status and all(
                c["status"] == "VERIFIED" for c in children[:-1]
            )
        return (
            status == "VERIFIED"
            and len(children) == 2
            and all(c["status"] == "VERIFIED" for c in children)
            and replay_domain([solution, actual, rhs], p.get("domain"), ctx)
        )
    return False


# Family routing is diagnostic only. No route can create mathematical evidence.
def classify_families(text):
    t = text.lower()
    scores = []
    for code, title in MATH_FAMILIES.items():
        words = [
            w
            for w in re.findall(r"[a-z]+", title.lower())
            if len(w) >= 5 and w not in ("theory", "general", "other", "systems")
        ]
        hits = sum(bool(re.search(r"\b" + re.escape(w) + r"\b", t)) for w in words)
        if hits:
            scores.append((hits, code))
    return [code for _, code in sorted(scores, reverse=True)[:4]]


def parse_statement(text):
    if not isinstance(text, str) or len(text) > 20000:
        raise Invalid("bounded statement required")
    t = text.strip().rstrip(".").strip()
    pattern = r"(?:For all|forall) (real|integer)(?:s| numbers?)? ([A-Za-z][A-Za-z0-9_]*(?:\s*(?:,|\band\b)\s*[A-Za-z][A-Za-z0-9_]*)*)\s*[:,]\s*(.+)"
    m = re.fullmatch(pattern, t, flags=re.I)
    if not m:
        raise Unsupported(
            "free text is not an executable mathematical claim; provide typed JSON"
        )
    body = m[3]
    relations = list(re.finditer(r"(?<![<>=!])(?:==|!=|<=|>=|=|<|>)(?![=])", body))
    if len(relations) != 1:
        raise Unsupported("one complete relation required; clauses are never discarded")
    r = relations[0]
    names = re.split(r"\s*(?:,|\band\b)\s*", m[2])
    if len(set(names)) != len(names):
        raise Invalid("duplicate quantified variable")
    return {
        "kind": "scalar",
        "variables": {n: {"domain": m[1].lower()} for n in names},
        "lhs": body[: r.start()].strip(),
        "op": r[0],
        "rhs": body[r.end() :].strip(),
    }


def coverage_report():
    return {
        "taxonomy": "MSC2020 top-level sections",
        "source": MSC_SOURCE,
        "recognized_top_level_sections": len(MATH_FAMILIES),
        "families": MATH_FAMILIES,
        "verification_backends": [
            "scalar exact algebra/SOS/interval",
            "symbolic differentiation",
            "bounded integer enumeration",
            "Boolean truth tables",
            "rational PSD/LDL",
            "fixed-dimension symbolic matrices",
            "first-order ODE solution+IC",
            "conditional generator mathematical subclaims",
        ],
        "arbitrary_claim_verification": False,
        "scope_warning": "A taxonomy label is not proof support; no engine here covers arbitrary statements across all mathematics.",
    }


def controller_expression(node, depth=0):
    if depth > 4:
        raise Limit("generator controller depth exceeded")
    strict(node, ("op", "name", "children"), ("op", "name", "children"))
    op = node["op"]
    name = node["name"]
    children = node["children"]
    arity = {
        "signal": 0,
        "half": 0,
        "complement": 1,
        "mean": 2,
        "product": 2,
        "minimum": 2,
        "maximum": 2,
    }
    signals = ("disagreement", "dispersion", "alignment", "progress", "time", "energy")
    if (
        not isinstance(op, str)
        or op not in arity
        or not isinstance(children, list)
        or len(children) != arity[op]
        or not isinstance(name, str)
    ):
        raise Invalid("invalid generator controller")
    if op == "signal":
        if name not in signals:
            raise Invalid("unsupported generator signal")
        return S(name)
    if name:
        raise Invalid("only signal nodes have names")
    if op == "half":
        return C(F(1, 2))
    a = [controller_expression(c, depth + 1) for c in children]
    if op == "complement":
        return E("sub", C(1), a[0])
    if op == "mean":
        return E("div", E("add", *a), C(2))
    return E({"product": "mul", "minimum": "min", "maximum": "max"}[op], *a)


def generator_contract(record):
    allowed = (
        "schema_version",
        "hid",
        "signature",
        "statement",
        "algorithm",
        "task",
        "strategy",
        "parents",
        "seed",
        "round_no",
        "equations",
        "pseudocode",
        "assumptions",
        "risks",
        "falsification",
        "ablations",
        "acceptance_gates",
        "novelty_status",
        "resources",
    )
    strict(
        record,
        allowed,
        ("schema_version", "algorithm", "task", "signature", "statement"),
    )
    if record["schema_version"] != "generator-3.0":
        raise Unsupported("unrecognized generator schema")
    a = record["algorithm"]
    strict(
        a,
        (
            "kind",
            "feature_map",
            "scaling",
            "loss",
            "mechanism",
            "aggregation",
            "preconditioner",
            "controller",
            "coupling",
        ),
        (
            "kind",
            "feature_map",
            "scaling",
            "loss",
            "mechanism",
            "aggregation",
            "preconditioner",
            "controller",
            "coupling",
        ),
    )
    choices = {
        "kind": ("regression", "classification"),
        "feature_map": ("linear", "diagonal_quadratic", "linear_fourier"),
        "scaling": ("standard", "robust"),
        "mechanism": (
            "clip_memory",
            "clip_step",
            "memory_step",
            "dual_memory",
            "penalty_step",
            "variance_penalty",
        ),
        "aggregation": ("mean", "median_of_means"),
        "preconditioner": ("identity", "rms"),
        "coupling": ("coupled", "first_only"),
    }
    if any(a[k] not in vals for k, vals in choices.items()):
        raise Invalid("unknown generator component")
    losses = {
        "regression": ("squared", "huber", "pseudo_huber", "cauchy"),
        "classification": ("cross_entropy", "brier", "focal"),
    }
    if a["loss"] not in losses[a["kind"]] or (
        a["mechanism"] == "variance_penalty" and a["preconditioner"] != "rms"
    ):
        raise Invalid("incompatible generator component")
    if not isinstance(record["signature"], str) or record["signature"] != digest(a):
        raise Invalid("generator definition signature mismatch")
    task = record["task"]
    strict(
        task,
        ("kind", "features", "classes", "goals", "max_parameters", "max_state_vectors"),
        ("kind", "features", "classes"),
    )
    if (
        task["kind"] != a["kind"]
        or type(task["features"]) is not int
        or not 1 <= task["features"] <= 4096
        or type(task["classes"]) is not int
        or not 1 <= task["classes"] <= 256
    ):
        raise Invalid("invalid generator task shape")
    if (a["kind"] == "regression" and task["classes"] != 1) or (
        a["kind"] == "classification" and task["classes"] < 2
    ):
        raise Invalid("invalid class count")
    return a, task, controller_expression(a["controller"])


def generator_audit(record, engine):
    a, task, controller = generator_contract(record)
    sub = []

    def add(name, claim, assumptions_note=""):
        result = engine.check(claim)
        sub.append(
            {
                "obligation": name,
                "claim": claim,
                "assumptions_note": assumptions_note,
                "verdict": result.to_dict(),
            }
        )

    signals = ("disagreement", "dispersion", "alignment", "progress", "time", "energy")
    variables = {s: {"lower": "0", "upper": "1"} for s in signals}
    for op, value, name in (
        (">=", 0, "controller_lower_bound"),
        ("<=", 1, "controller_upper_bound"),
    ):
        add(
            name,
            {
                "kind": "scalar",
                "variables": variables,
                "lhs": expr_text(controller),
                "op": op,
                "rhs": value,
            },
            "Conditional on all six training signals lying in [0,1]; signal formula range proofs are not silently assumed discharged.",
        )
    q = {"q": {"lower": "0", "upper": "1"}}
    q2 = "q" if a["coupling"] == "coupled" else "(1/2)"
    if a["mechanism"] in ("clip_memory", "clip_step"):
        for op, b in ((">=", "1/4"), ("<=", "7/4")):
            add(
                "clip_multiplier_" + op,
                {
                    "kind": "scalar",
                    "variables": q,
                    "lhs": "1/4+3/2*(1-q)",
                    "op": op,
                    "rhs": b,
                },
            )
    for op, b in ((">=", "0"), ("<=", "19/20")):
        expr = (
            f"19/20*(1-{q2})"
            if a["mechanism"] == "clip_memory"
            else (
                "19/20*q"
                if a["mechanism"] == "memory_step"
                else "1/2" if a["mechanism"] == "dual_memory" else "9/10"
            )
        )
        add(
            "momentum_coefficient_" + op,
            {"kind": "scalar", "variables": q, "lhs": expr, "op": op, "rhs": b},
            "q is the controller output; q2=q for coupled and q2=1/2 for first_only. dual_memory fast coefficient is 1/2.",
        )
    step = {
        "clip_memory": "1",
        "clip_step": f"1/4+3/4*{q2}",
        "memory_step": f"1-3/4*{q2}",
        "dual_memory": f"1-1/2*{q2}",
        "penalty_step": f"1/(1+{q2})",
        "variance_penalty": "1",
    }[a["mechanism"]]
    for op, b in ((">", "0"), ("<=", "1")):
        add(
            "mechanism_step_factor_" + op,
            {"kind": "scalar", "variables": q, "lhs": step, "op": op, "rhs": b},
            "Relative to the nonnegative base step, under q in [0,1] and the declared coupling.",
        )
    for op, b in ((">", "0"), ("<=", "1")):
        add(
            "clipping_weight_" + op,
            {
                "kind": "scalar",
                "variables": {
                    "c": {"lower": "0", "lower_open": True},
                    "e_norm": {"lower": "0"},
                },
                "lhs": "min(1,c/(e_norm+1/1000000000000))",
                "op": op,
                "rhs": b,
            },
            "Conditional on a positive finite clipping threshold and nonnegative finite example-gradient norm.",
        )
    if a["kind"] == "regression":
        definitions = {
            "squared": ("r*r/2", "r"),
            "pseudo_huber": ("sqrt(1+r*r)-1", "r/sqrt(1+r*r)"),
            "cauchy": ("log(1+r*r)/2", "r/(1+r*r)"),
        }
        if a["loss"] in definitions:
            f, g = definitions[a["loss"]]
            add(
                "declared_loss_derivative",
                {
                    "kind": "derivative",
                    "variables": {"r": {}},
                    "expression": f,
                    "variable": "r",
                    "proposed": g,
                },
            )
        else:
            for tag, bounds, f, g in (
                ("center", {"lower": "-1", "upper": "1"}, "r*r/2", "r"),
                ("right", {"lower": "1"}, "r-1/2", "1"),
                ("left", {"upper": "-1"}, "-r-1/2", "-1"),
            ):
                add(
                    "huber_" + tag + "_branch_derivative",
                    {
                        "kind": "derivative",
                        "variables": {"r": bounds},
                        "expression": f,
                        "variable": "r",
                        "proposed": g,
                    },
                    "Branchwise derivative; boundary matching and global C1 proof remain a separate obligation.",
                )
    else:
        # General softmax Jacobian reduced to same/distinct index, with u the
        # positive sum of the other exponentials. This is an explicit conditional
        # lemma, never a numerical gradient check or a claim of convergence.
        add(
            "softmax_same_index",
            {
                "kind": "derivative",
                "variables": {"z": {}, "u": {"lower": "0", "lower_open": True}},
                "expression": "exp(z)/(exp(z)+u)",
                "variable": "z",
                "proposed": "exp(z)/(exp(z)+u)*(1-exp(z)/(exp(z)+u))",
            },
            "u is held constant in this derivative and represents the sum of all other positive exponentials.",
        )
        add(
            "softmax_distinct_index",
            {
                "kind": "derivative",
                "variables": {"z": {}, "v": {}, "u": {"lower": "0"}},
                "expression": "exp(v)/(exp(v)+exp(z)+u)",
                "variable": "z",
                "proposed": "-exp(v)*exp(z)/(exp(v)+exp(z)+u)^2",
            },
            "u represents all remaining exponentials; the lemma is conditional on that mathematical representation.",
        )
        if a["loss"] == "cross_entropy":
            add(
                "cross_entropy_probability_derivative",
                {
                    "kind": "derivative",
                    "variables": {
                        "p": {"lower": "0", "lower_open": True, "upper": "1"}
                    },
                    "expression": "-log(p)",
                    "variable": "p",
                    "proposed": "-1/p",
                },
            )
        elif a["loss"] == "focal":
            add(
                "focal_probability_derivative",
                {
                    "kind": "derivative",
                    "variables": {
                        "p": {"lower": "0", "lower_open": True, "upper": "1"}
                    },
                    "expression": "-(1-p)^2*log(p)",
                    "variable": "p",
                    "proposed": "2*(1-p)*log(p)-(1-p)^2/p",
                },
            )
        else:
            add(
                "brier_coordinate_derivative",
                {
                    "kind": "derivative",
                    "variables": {"p": {}, "y": {}},
                    "expression": "(p-y)^2/2",
                    "variable": "p",
                    "proposed": "p-y",
                },
            )
    # These two conditional exact identities establish the advertised length cap
    # once a nonnegative direction norm and positive base step are supplied.
    for branch, bounds, expr in (
        ("small", {"lower": "0", "upper": "1"}, "eta*s"),
        ("large", {"lower": "1"}, "eta"),
    ):
        add(
            "normalized_step_" + branch,
            {
                "kind": "scalar",
                "variables": {"eta": {"lower": "0"}, "s": bounds},
                "lhs": expr,
                "op": "<=",
                "rhs": "eta",
            },
            "Piecewise scalar consequence; binding s to the actual vector norm and correctness of an external implementation are not proved here.",
        )
    return Verdict(
        "UNVERIFIED",
        "partial mathematical obligations only",
        [
            "Whole empirical-improvement/novelty hypothesis is not mathematically verified.",
            "Only the displayed typed subclaims are proved; annotations, all signal ranges, convergence and implementation semantics are not silently certified.",
            "No training, validation or test data or scores were consumed. Scientific evaluation engine is out of scope.",
        ],
        subclaims=sub,
        family=["49", "68"],
    )


class MathCheckerEngine:
    def __init__(self, limits=None):
        self.limits = limits or Limits()
        self._cache = OrderedDict()

    def check(self, claim):
        start = perf_counter()
        families = []
        try:
            if not isinstance(claim, (dict, str)):
                if hasattr(claim, "to_dict"):
                    claim = claim.to_dict()
                elif hasattr(claim, "statement"):
                    claim = claim.statement
                else:
                    raise Invalid("typed claim, statement or hypothesis required")
            if isinstance(claim, str):
                families = classify_families(claim)
                claim = parse_statement(claim)
            encoded = canonical(claim)
            if len(encoded.encode()) > self.limits.max_payload_bytes:
                raise Limit("input payload byte budget exceeded")
            key = digest({"claim": claim, "limits": asdict(self.limits)})
            if key in self._cache:
                result = copy.deepcopy(self._cache[key])
                self._cache.move_to_end(key)
                result.elapsed_ms = (perf_counter() - start) * 1000
                return result
            if (
                isinstance(claim, dict)
                and claim.get("schema_version") == "generator-3.0"
            ):
                result = generator_audit(claim, self)
            else:
                status, payload, witness = _prove(claim, Budget(self.limits))
                cert = _certificate(claim, status, payload)
                if not verify_certificate(claim, cert, self.limits):
                    result = Verdict(
                        "UNVERIFIED",
                        "none",
                        [
                            "Candidate certificate failed replay; no verified verdict emitted."
                        ],
                    )
                else:
                    scope = (
                        "explicit finite domain only"
                        if status == "VERIFIED_FINITE"
                        else (
                            "exact supplied claim under its stated domains/assumptions"
                            if status == "VERIFIED"
                            else (
                                "exact counterexample in stated domain"
                                if status in ("DISPROVEN", "ILL_DEFINED")
                                else "proved empty declared domain"
                            )
                        )
                    )
                    result = Verdict(status, scope, [payload["method"]], cert, witness)
            result.family = result.family or families
            self._cache[key] = copy.deepcopy(result)
            if len(self._cache) > self.limits.cache_entries:
                self._cache.popitem(last=False)
        except Unsupported as e:
            result = Verdict("UNVERIFIED", "none", [str(e)], family=families)
        except (Limit, RecursionError, MemoryError) as e:
            result = Verdict("RESOURCE_LIMIT", "none", [str(e)])
        except (
            Invalid,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            ZeroDivisionError,
            OverflowError,
        ) as e:
            result = Verdict("INVALID", "none", [str(e)])
        except (NonExact, Undefined) as e:
            result = Verdict(
                "UNVERIFIED", "none", ["No replayable exact certificate: " + str(e)]
            )
        result.elapsed_ms = (perf_counter() - start) * 1000
        return result

    def check_many(self, claims):
        return [self.check(c) for c in claims]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument(
        "input", nargs="?", default="-", help="JSON claim file, or - for stdin"
    )
    c.add_argument(
        "--statement", help="restricted whole-statement grammar instead of JSON input"
    )
    v = sub.add_parser("replay")
    v.add_argument("input", help="JSON with claim and certificate")
    sub.add_parser("coverage")
    args = p.parse_args()
    if args.command == "coverage":
        print(json.dumps(coverage_report(), indent=2))
        return
    if args.command == "check" and args.statement is not None:
        claim = args.statement
    else:
        if args.input == "-":
            raw = sys.stdin.buffer.read(Limits().max_payload_bytes + 1)
        else:
            with open(args.input, "rb") as f:
                raw = f.read(Limits().max_payload_bytes + 1)
        if len(raw) > Limits().max_payload_bytes:
            raise SystemExit("input exceeds byte limit")
        claim = json.loads(raw)
    if args.command == "replay":
        strict(claim, ("claim", "certificate"), ("claim", "certificate"))
        print(
            json.dumps(
                {"valid": verify_certificate(claim["claim"], claim["certificate"])}
            )
        )
        return
    print(
        json.dumps(
            MathCheckerEngine().check(claim).to_dict(), indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
