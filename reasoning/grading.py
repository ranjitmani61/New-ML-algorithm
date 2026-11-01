"""Answer equivalence for grading and for grouping attempts into answer clusters.

String normalisation (benchmarks.normalize) misses many equal answers on hard MATH problems:
``\\frac{\\sqrt{6}}2`` vs ``\\frac{\\sqrt{6}}{2}``, ``(-3/2,3)`` vs ``(-1.5,3)``, ``AB=12+12\\sqrt{3}``
vs ``12\\sqrt{3}+12``. That both mis-grades attempts and splits one answer into several vote
clusters. Here two answers are the same if their normalised strings match or if Math-Verify
(the Hugging Face grader) finds them equal in both directions. Grading uses the standard
direction verify(gold, prediction).

Multiple-choice problems (AMC style, answer options in the problem text) are answered with a
letter by the model while the reference solution boxes the value; letters are mapped to the
option text taken from the problem. Everything used for clustering is label-free (attempts and
problem text only), so every selection method sees the same clusters.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import re
from functools import lru_cache

import benchmarks as bm
import lab

_ASY = re.compile(r"\[asy\].*?\[/asy\]", re.S)
_MARK = re.compile(r"\(([A-E])\)")
_WRAP = re.compile(r"\\(?:qquad|quad|mathrm|textbf|textrm|text|mbox|hspace\{[^{}]*\}|hfill|newline)")
_LETTER = re.compile(r"\(?([A-E])\)?(.*)", re.S)


def _trim_braces(s):
    """Drop braces that are unbalanced (left over from the option markup)."""
    keep, stack = [], []
    for ch in s:
        if ch == "}":
            if not stack:
                continue
            stack.pop()
        elif ch == "{":
            stack.append(len(keep))
        keep.append(ch)
    for i in reversed(stack):
        del keep[i]
    return "".join(keep)


def mc_options(problem):
    """{letter: option text} for problems that list options (A)-(D)[/(E)]; {} otherwise."""
    text = _ASY.sub(" ", problem or "")
    marks = list(_MARK.finditer(text))
    letters = [m.group(1) for m in marks]
    run = None
    for i in range(len(marks)):
        if letters[i:i + 4] == list("ABCD"):
            run = marks[i:i + 5] if letters[i:i + 5] == list("ABCDE") else marks[i:i + 4]
            break
    if run is None:
        return {}
    options = {}
    for k, m in enumerate(run):
        end = run[k + 1].start() if k + 1 < len(run) else len(text)
        v = text[m.end():end]
        v = v.replace("\\\\", " ").replace("$", " ")
        v = _WRAP.sub(" ", v).replace("\\ ", " ").replace("~", " ")
        v = _trim_braces(v).strip().rstrip(".,;").strip()
        if v:
            options[m.group(1)] = v
    return options if len(options) >= 4 else {}


def map_choice(raw, options):
    """Replace a letter answer ('C', '(C)', '\\textbf{(C) }270') by the option's text."""
    if raw is None or not options:
        return raw
    s = bm._remove_text(raw).replace("$", "").strip()
    m = _LETTER.fullmatch(s)
    if m and m.group(1) in options and (s.startswith("(") or len(s) == 1 or not m.group(2).strip()
                                        or m.group(2).lstrip().startswith((")", ":", "\\"))):
        return options[m.group(1)]
    return raw


@lru_cache(maxsize=None)
def _parsed(raw):
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse

    try:
        return parse("$" + raw.replace("$", "") + "$",
                     extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()], parsing_timeout=5)
    except Exception:  # noqa: BLE001 - a parse failure only means "no symbolic match"
        return []


@lru_cache(maxsize=None)
def _verify(gold, pred):
    from math_verify import verify

    a, b = _parsed(gold), _parsed(pred)
    if not a or not b:
        return False
    try:
        return bool(verify(a, b, timeout_seconds=5))
    except Exception:  # noqa: BLE001
        return False


def same_answer(a, b):
    """Symmetric equivalence used for clustering."""
    if a is None or b is None:
        return False
    return bm.normalize(a) == bm.normalize(b) or (_verify(a, b) and _verify(b, a))


def matches_gold(gold, pred):
    if gold is None or pred is None:
        return False
    return bm.normalize(gold) == bm.normalize(pred) or _verify(gold, pred)


def cluster(raw_answers, problem):
    """Canonical key per attempt (None for no answer); equivalent answers share a key.

    Returns (keys, members) where members maps each key to the raw forms it groups.
    """
    options = mc_options(problem)
    mapped = [map_choice(r, options) for r in raw_answers]
    norm = [bm.normalize(m) if m is not None else None for m in mapped]
    raw_of, count = {}, {}
    for k, m in zip(norm, mapped):
        if k is not None and k != "":
            raw_of.setdefault(k, m)
            count[k] = count.get(k, 0) + 1
    distinct = list(raw_of)
    parent = {k: k for k in distinct}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(distinct):
        for b in distinct[i + 1:]:
            ra, rb = find(a), find(b)
            if ra != rb and same_answer(raw_of[a], raw_of[b]):
                parent[rb] = ra
    groups = {}
    for k in distinct:
        groups.setdefault(find(k), []).append(k)
    rep, members = {}, {}
    for ks in groups.values():
        r = max(ks, key=lambda k: (count[k], -distinct.index(k)))
        members[r] = [raw_of[k] for k in ks]
        for k in ks:
            rep[k] = r
    keys = [rep[k] if k in rep else None for k in norm]
    return keys, members


def gold_key(gold_raw, members, problem=""):
    """Key of the cluster that matches the reference answer, else the normalised reference."""
    gold = map_choice(gold_raw, mc_options(problem))
    for key, raws in members.items():
        if any(matches_gold(gold, r) for r in raws):
            return key
    g = bm.normalize(gold)
    return g if g not in members else g + "\u2020"  # never collide with a non-matching cluster


def load_pools(directory, sets=None):
    """lab.Question objects whose trace answers are equivalence-cluster keys and whose gold is
    the key of the matching cluster, with per-attempt mention sets. Attempt features are those of
    lab.load_pools."""
    import selection

    questions = {q.qid: q for q in lab.load_pools(directory, sets)}
    for part in sorted(glob.glob(os.path.join(directory, "**", "part*.jsonl.gz"), recursive=True)):
        for line in gzip.open(part, "rt"):
            item = json.loads(line)
            q = questions.get(item["id"])
            if q is None:
                continue
            raws = [bm.model_answer(s["text"]) if s["finished"] else None for s in item["samples"]]
            keys, members = cluster(raws, item["question"])
            for t, k, s in zip(q.traces, keys, item["samples"]):
                t.answer = k
                t.repaired = None
                t.mentions = frozenset(selection.mentioned_values(s["text"]))
            q.problem_values = frozenset(selection.mentioned_values(item["question"]))
            q.gold = gold_key(bm.gold_answer(item), members, item["question"])
    return sorted(questions.values(), key=lambda q: q.qid)
