"""Answer selection over a pool of attempts: tools for the hard (MATH level 5) setting.

Everything here is zero-token: it reads attempts that were already generated.

Pass-through count. Final-answer voting ignores what the other attempts computed on the way.
For a candidate answer A, L(A) is the number of attempts that do NOT answer A but write A as a
value in their reasoning (values copied from the problem statement are ignored).

The first hypothesis, that L(A) is extra support for A ("latent votes"), was falsified on hard
development data: when the majority answer is wrong it is usually the value that other attempts
pass through and then go beyond (an intermediate result, a partial case count, one of two
roots). Terminal-ratio voting uses L(A) the other way round. Of the attempts that reached the
value A, the fraction that stopped there is

    tau(A) = v(A) / (v(A) + L(A)),

and a value that most attempts reach and then transform further is a waypoint rather than a
destination. The score of A is its (optionally confidence-weighted) vote mass times tau(A)**gamma;
gamma = 0 is ordinary voting.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import re

import benchmarks as bm
import lab

_NUM = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?![\d])")
_FRAC = re.compile(r"-?\\[dt]?frac\{[^{}]+\}\{[^{}]+\}")
_SQRT = re.compile(r"-?\d*\\sqrt\{[^{}]+\}")
_SLASH = re.compile(r"(?<![\w.])-?\d+/\d+(?![\d])")


def mentioned_values(text):
    """Normalised values written anywhere in a text (numbers, fractions, k*sqrt(n), a/b)."""
    out = set()
    for pattern in (_FRAC, _SQRT, _SLASH, _NUM):
        for m in pattern.findall(text or ""):
            v = bm.normalize(m)
            if v:
                out.add(v)
    return out


def load_with_texts(directory, sets=None):
    """lab.Question objects plus, per question, the mention set of each attempt and of the problem."""
    questions = {q.qid: q for q in lab.load_pools(directory, sets)}
    mentions, problem_values = {}, {}
    for part in sorted(glob.glob(os.path.join(directory, "**", "part*.jsonl.gz"), recursive=True)):
        for line in gzip.open(part, "rt"):
            item = json.loads(line)
            if item["id"] not in questions:
                continue
            mentions[item["id"]] = [mentioned_values(s["text"]) for s in item["samples"]]
            problem_values[item["id"]] = mentioned_values(item["question"])
    qs = sorted(questions.values(), key=lambda q: q.qid)
    return qs, mentions, problem_values


def latent_support(answers, mention_sets, problem_values, candidates=None):
    """For each candidate: number of attempts with a different final answer that mention it."""
    candidates = candidates if candidates is not None else {a for a in answers if a is not None}
    support = {}
    for a in candidates:
        if a is None or a in problem_values:
            support[a] = 0
            continue
        support[a] = sum(1 for b, m in zip(answers, mention_sets) if b != a and a in m)
    return support


def latent_vote(traces, mention_sets, problem_values, beta=0.5, weights=None):
    """Score(a) = sum of weights of attempts answering a + beta * latent support of a."""
    answers = [t.answer for t in traces]
    base, first = {}, {}
    for i, a in enumerate(answers):
        if a is None:
            continue
        base[a] = base.get(a, 0.0) + (1.0 if weights is None else weights[i])
        first.setdefault(a, i)
    if not base:
        return None, {}
    latent = latent_support(answers, mention_sets, problem_values, set(base))
    scores = {a: base[a] + beta * latent[a] for a in base}
    return max(scores, key=lambda a: (scores[a], -first[a])), scores


def terminal_ratio_vote(traces, problem_values, gamma=2.0, weights=None):
    """Score(A) = W(A) * (v(A) / (v(A) + L(A)))**gamma; returns (answer, scores).

    W(A) is the vote mass (count, or sum of ``weights``), v(A) the count of attempts answering A,
    L(A) the number of attempts answering something else whose text contains the value A
    (``trace.mentions``). Ties go to the earliest attempt, as in lab.vote.
    """
    answers = [t.answer for t in traces]
    count, mass, first = {}, {}, {}
    for i, a in enumerate(answers):
        if a is None:
            continue
        count[a] = count.get(a, 0) + 1
        mass[a] = mass.get(a, 0.0) + (1.0 if weights is None else float(weights[i]))
        first.setdefault(a, i)
    if not count:
        return None, {}
    scores = {}
    for a in count:
        passed = 0 if a in problem_values else sum(
            1 for b, t in zip(answers, traces) if b != a and t.mentions is not None and a in t.mentions)
        scores[a] = mass[a] * (count[a] / (count[a] + passed)) ** gamma
    return max(scores, key=lambda a: (scores[a], -first[a])), scores
