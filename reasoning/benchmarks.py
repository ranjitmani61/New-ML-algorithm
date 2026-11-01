"""Benchmark data for the reasoning lab: answers, grading and exact step checking.

* Gold answers: GSM8K ("#### n") and MATH (last \\boxed{...} of the solution).
* Model answers: last \\boxed{...} of a completion (balanced braces).
* Normalization and equivalence: Hendrycks-style string normalization plus exact
  rational comparison whenever both sides are plain numbers (0.5 == \\frac12 == 1/2).
* Step verifier: finds "expression = value" claims in a chain of thought
  ("16 - 3 - 4 = 9", "\\frac{3}{4} \\times 8 = 6", "<<48/2=24>>"), evaluates both
  sides with exact rational arithmetic, and reports false steps. A decimal written
  with k places is accepted when it is the correct value rounded to k places.
  No tokens are generated; this is CPU-only evidence about a trace.
"""

from __future__ import annotations

import re
from fractions import Fraction

# ---------------------------------------------------------------- extraction


def last_boxed(text):
    """Content of the last \\boxed{...} (or \\fbox{...}), with balanced braces."""
    if not text:
        return None
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    i = text.find("{", start)
    after = text[start:].split(None, 1)
    if i < 0 or (len(after) == 2 and text.find("{", start) > start + len(after[0]) + 1):
        # "\boxed 5" form: the next whitespace-delimited token
        return after[1].split("$")[0].split()[0].rstrip(".") if len(after) == 2 else None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def gsm8k_gold(solution):
    tail = solution.split("####")[-1].strip()
    return tail.replace(",", "").replace("$", "").strip()


def math_gold(solution):
    return last_boxed(solution)


def gold_answer(item):
    return gsm8k_gold(item["solution"]) if item["id"].startswith("gsm8k") else math_gold(item["solution"])


def model_answer(text):
    """Final answer of a completion: last \\boxed{}, else 'answer is X', else None."""
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed
    m = re.findall(r"answer is[:\s]*\$?([-\d.,/\\{}a-zA-Z]+)", text or "")
    return m[-1].rstrip(".") if m else None


# ---------------------------------------------------------------- normalization


def _fix_fracs(s):
    """\\frac12 -> \\frac{1}{2}, \\frac1{2} -> \\frac{1}{2} (Hendrycks MATH convention)."""
    parts = s.split("\\frac")
    out = parts[0]
    for p in parts[1:]:
        out += "\\frac"
        if p.startswith("{"):
            out += p
        elif len(p) < 2:
            return s
        elif p[1] == "{":
            out += "{" + p[0] + "}" + p[1:]
        else:
            out += "{" + p[0] + "}{" + p[1] + "}" + p[2:]
    return out


def _remove_text(s):
    return re.sub(r"\\(?:text|textbf|mathrm|mbox)\{([^{}]*)\}", r"\1", s)


UNIT_WORDS = r"(?:dollars?|cents?|units?|square|feet|foot|inches?|meters?|cm|km|degrees?|hours?|minutes?|seconds?|days?|years?|miles?|mph|pounds?|kg|grams?|liters?|students?|people|marbles?|apples?)"


def normalize(answer):
    """Canonical string for voting and grading (None stays None)."""
    if answer is None:
        return None
    s = str(answer).strip()
    s = s.replace("\\\\", ",")  # LaTeX row/line separator, before single-backslash spacing commands
    s = s.replace("\n", "").replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\displaystyle", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\circ", "")
    s = s.replace("\\%", "").replace("%", "").replace("\\$", "").replace("$", "")
    s = _remove_text(s)
    s = re.sub(r"\s*" + UNIT_WORDS + r"\s*$", "", s)
    s = s.strip().rstrip(".").strip()
    if re.fullmatch(r"[a-zA-Z]\s*=\s*.+", s) and "," not in s:
        s = s.split("=", 1)[1]
    s = s.replace(" ", "")
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    s = _fix_fracs(s)
    s = re.sub(r"\\frac\{(-?\d+)\}\{(\d+)\}", r"\1/\2", s)  # simple fractions inside compound answers
    if s.startswith("0.") is False and re.fullmatch(r"\.\d+", s):
        s = "0" + s
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    s = s.replace("\\{", "{").replace("\\}", "}")
    value = to_fraction(s)
    if value is not None:
        return _canonical_number(value)
    return s


def _canonical_number(v):
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def to_fraction(s):
    """Plain numbers only: integers, decimals, a/b, \\frac{a}{b}, signs."""
    if s is None:
        return None
    t = str(s).replace(" ", "")
    neg = False
    while t.startswith(("-", "+")):
        neg ^= t[0] == "-"
        t = t[1:]
    m = re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", t)
    try:
        if m:
            v = Fraction(m.group(1)) / Fraction(m.group(2))
        elif re.fullmatch(r"\d+(?:\.\d+)?/\d+(?:\.\d+)?", t):
            a, b = t.split("/")
            v = Fraction(a) / Fraction(b)
        elif re.fullmatch(r"\d+(?:\.\d+)?|\.\d+", t):
            v = Fraction(t)
        else:
            return None
    except (ZeroDivisionError, ValueError):
        return None
    return -v if neg else v


def is_equiv(pred, gold):
    if pred is None or gold is None:
        return False
    a, b = normalize(pred), normalize(gold)
    return a == b


# ---------------------------------------------------------------- step verifier

_NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"
_TOKEN = re.compile(r"\s*(\d+(?:,\d{3})*(?:\.\d+)?|\.\d+|\*\*|[-+*/()^])")


def _latex_to_plain(expr):
    e = expr
    e = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", e)
    e = e.replace("\\times", "*").replace("\\cdot", "*").replace("×", "*").replace("÷", "/").replace("\\div", "/")
    e = e.replace("\\left", "").replace("\\right", "").replace("{", "(").replace("}", ")")
    e = e.replace("$", "").replace("\\$", "").replace("−", "-").replace("–", "-")
    e = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", e)  # "3 x 4" multiplication
    return e


class _Parser:
    def __init__(self, text):
        self.tokens = []
        pos = 0
        text = text.strip()
        while pos < len(text):
            m = _TOKEN.match(text, pos)
            if not m:
                raise ValueError("unsupported character")
            self.tokens.append(m.group(1))
            pos = m.end()
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self):
        t = self.peek()
        self.i += 1
        return t

    def parse(self):
        v = self.expr()
        if self.peek() is not None:
            raise ValueError("trailing tokens")
        return v

    def expr(self):
        v = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            w = self.term()
            v = v + w if op == "+" else v - w
        return v

    def term(self):
        v = self.factor()
        while self.peek() in ("*", "/") or (self.peek() == "("):
            op = self.take() if self.peek() in ("*", "/") else "*"
            w = self.factor()
            if op == "*":
                v = v * w
            else:
                if w == 0:
                    raise ZeroDivisionError
                v = v / w
        return v

    def factor(self):
        t = self.peek()
        if t in ("+", "-"):
            self.take()
            v = self.factor()
            return v if t == "+" else -v
        base = self.atom()
        if self.peek() in ("^", "**"):
            self.take()
            exponent = self.factor()
            if exponent.denominator != 1 or abs(exponent) > 12 or abs(base) > 10**6:
                raise ValueError("exponent unsupported")
            return base ** int(exponent)
        return base

    def atom(self):
        t = self.take()
        if t == "(":
            v = self.expr()
            if self.take() != ")":
                raise ValueError("unbalanced")
            return v
        if t is None or not (t[0].isdigit() or t[0] == "."):
            raise ValueError("number expected")
        return Fraction(t.replace(",", ""))


def evaluate(expr):
    """Exact value of a plain numeric expression, or None if it is not one."""
    plain = _latex_to_plain(expr)
    if not re.search(r"\d", plain) or re.search(r"[a-zA-Z\\_]", plain):
        return None
    try:
        return _Parser(plain).parse()
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _rounded_ok(value, written):
    """Accept '3.33' for 10/3: written decimal equals value rounded to its places."""
    w = written.replace(",", "").strip()
    if "." not in w:
        return False
    places = len(w.split(".")[1])
    try:
        return abs(value - Fraction(w)) <= Fraction(1, 2 * 10**places)
    except ValueError:
        return False


_CLAIM = re.compile(
    r"(?<![A-Za-z0-9_^}\)\].])([\d.(\[{\\][\d\s.,+\-*/×÷^()\[\]{}\\a-zA-Z%]*?)\s*=\s*\\?\$?\s*"
    r"(-?(?:" + _NUM + r"|\.\d+)(?:\s*/\s*" + _NUM + r")?)(?![\d.,]*\d|[.,]?\s*\\|\s*[-+/^*×÷!]|\s*\\(?:frac|times|cdot|sqrt|pi|div)|[a-zA-Z(\\{]|\s+(?:and\s+)?\d+\s*/\s*\d)"
)
_WORD = re.compile(r"(?<!\\)\b[a-zA-Z]{2,}\b|%")


def check_steps(text):
    """Exact check of 'numeric expression = number' claims in a completion.

    Returns {"claims", "false", "false_values": [(written, exact)], "last": (ok, written, exact)}.
    Only claims whose left side contains an operator and parses as a plain
    numeric expression are counted; prose, variables and units are skipped.
    """
    claims = false = 0
    false_values = []
    last = None
    seen = set()
    text = (text or "").replace("\\[", " ").replace("\\]", " ")
    text = text.replace("\\!", "").replace("\\,", "").replace("\\;", "")  # LaTeX spacing (e.g. 729,\!000)
    for m in _CLAIM.finditer(text):
        lhs, rhs = m.group(1), m.group(2)
        before = text[:m.start(1)].rstrip(" $")
        if before and before[-1] in "+-*/^×÷−":
            continue  # the match starts inside a longer expression (x^2 + 3^2 = 25)
        if re.search(r"\d\s*\\[dt]?frac", lhs):
            continue  # mixed number such as 33\frac{1}{3}: ambiguous, not judged
        lhs = _WORD.split(lhs.split("=")[-1])[-1]  # keep the mathematical tail of the phrase
        lhs = re.sub(r"^\.(?=\s|$)|^\.\s+", "", lhs).strip()  # a sentence-ending period is not a decimal point
        if not re.search(r"[+\-*/×÷^]|\\times|\\cdot|\\frac|\\div", lhs):
            continue
        key = (lhs.strip(), rhs)
        if key in seen:
            continue
        seen.add(key)
        value = evaluate(lhs)
        written = evaluate(rhs)
        if value is None or written is None:
            continue
        claims += 1
        rhs = rhs.replace(",", "").replace(" ", "")
        ok = value == written or ("/" not in rhs and _rounded_ok(value, rhs))
        if not ok:
            false += 1
            false_values.append((rhs, _canonical_number(value)))
        last = (ok, rhs, _canonical_number(value))
    return {"claims": claims, "false": false, "false_values": false_values, "last": last}


# ---------------------------------------------------------------- attempt summary


def summarize_attempt(text, tokens, finished, conf=None, logprob=None, window=64, tail=256):
    """Compact, analysis-ready record of one attempt (shared by the GPU runner and the lab).

    [raw_answer, tokens, finished, conf_mean, conf_tail, conf_low, logprob_mean,
     checked_claims, false_claims, last_claim]  with last_claim = [ok, written, exact] or None.
    conf_* are DeepConf token confidences (-mean top-20 log-probabilities): the mean,
    the mean over the last `tail` tokens, and the lowest sliding-window (`window`) mean.
    """
    raw = model_answer(text) if finished else None
    steps = check_steps(text) if finished else {"claims": 0, "false": 0, "last": None}
    if conf is not None and len(conf):
        values = [float(v) for v in conf]
        mean = sum(values) / len(values)
        tail_values = values[-tail:]
        tail_mean = sum(tail_values) / len(tail_values)
        if len(values) >= window:
            run = sum(values[:window])
            low = run
            for i in range(window, len(values)):
                run += values[i] - values[i - window]
                low = min(low, run)
            low /= window
        else:
            low = mean
        lp = sum(float(v) for v in logprob) / len(logprob) if logprob is not None and len(logprob) else 0.0
    else:
        mean = tail_mean = low = lp = 0.0
    last = list(steps["last"]) if steps.get("last") else None
    return [raw, int(tokens), bool(finished), round(mean, 4), round(tail_mean, 4), round(low, 4), round(lp, 4),
            int(steps["claims"]), int(steps["false"]), last]
