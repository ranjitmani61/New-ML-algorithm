"""Third testbed: a small-model version of the three-person task (research/llm/multi.py).

Input: three start amounts, then K events (person, gain or loss, 1-9), then "=". Answer: the
largest final amount, two digits, least significant first. As in multi.py, starts are 5-30, each
event changes one person's amount, and no amount ever goes negative. Answers are capped at 99
(two answer digits); multi.py allows up to 999, so this is a small difference at long lengths.

Answer-preserving expansions, built without reading any amount (only the events themselves):
  pairs  P gains a ... (anywhere later) ... P loses a          (multi.cancel_spread)
  split  one event becomes two with the same person, kind and total (gains 7 -> gains 3 and
         gains 4). A gain's second piece moves to a random earlier position and a loss's to a
         random later one, so every intermediate amount is at least what it was: no amount can
         go negative, and every final amount is unchanged.
"""

from __future__ import annotations

import numpy as np

PAD, EQ = 0, 11                 # digits 0-9 are tokens 1-10
PERSON = (12, 13, 14)
GAIN, LOSS = 15, 16
VOCAB = 17
ANS = 2
N = 3


def finals(item):
    starts, events = item
    x = list(starts)
    for who, kind, n in events:
        x[who] += n if kind == 0 else -n
    return x


def valid(item):
    starts, events = item
    x = list(starts)
    if min(x) < 0:
        return False
    for who, kind, n in events:
        x[who] += n if kind == 0 else -n
        if x[who] < 0:
            return False
    return True


def answer(item):
    return max(finals(item))


def length(item):
    return len(item[1])


def sample(rng, k):
    """A valid problem with exactly k events (kind 0 = gain, 1 = loss), as multi.make_multi."""
    while True:
        starts = [int(rng.integers(5, 31)) for _ in range(N)]
        x, events = list(starts), []
        for _ in range(k):
            who = int(rng.integers(0, N))
            kind = int(rng.integers(0, 2))
            if kind == 1 and x[who] == 0:
                kind = 0
            n = int(rng.integers(1, 10)) if kind == 0 else int(rng.integers(1, min(9, x[who]) + 1))
            x[who] += n if kind == 0 else -n
            events.append((who, kind, n))
        item = (starts, events)
        if valid(item) and answer(item) <= 99:
            return item


def encode_input(item):
    starts, events = item
    toks = []
    for p, s in zip(PERSON, starts):
        toks += [p, 1 + s // 10, 1 + s % 10]
    for who, kind, n in events:
        toks += [PERSON[who], GAIN if kind == 0 else LOSS, 1 + n]
    return toks + [EQ]


def encode_answer(v):
    if not 0 <= v <= 99:
        raise ValueError(v)
    return [1 + v % 10, 1 + v // 10]


def decode_answer(toks):
    t = [int(x) for x in toks]
    if any(not 1 <= x <= 10 for x in t):
        return None
    return (t[0] - 1) + 10 * (t[1] - 1)


def batch(items, answers=None, max_in=None):
    enc = [encode_input(x) for x in items]
    max_in = max_in or max(len(e) for e in enc)
    out = np.zeros((len(enc), max_in + ANS), dtype=np.int32)
    for i, e in enumerate(enc):
        out[i, max_in - len(e):max_in] = e
        if answers is not None:
            out[i, max_in:] = encode_answer(answers[i])
    return out


def expand_pairs(rng, item, max_k):
    """multi.cancel_spread: k pairs, each a gain at i and the matching loss anywhere after it."""
    starts, events = item
    ev = list(events)
    for _ in range(int(rng.integers(0, (max_k - len(ev)) // 2 + 1))):
        who, a = int(rng.integers(0, N)), int(rng.integers(1, 10))
        i = int(rng.integers(0, len(ev) + 1))
        ev.insert(i, (who, 0, a))
        j = int(rng.integers(i + 1, len(ev) + 1))
        ev.insert(j, (who, 1, a))
    return (starts, ev)


def expand_split(rng, item, max_k):
    """Up to max_k - k splits; each turns one event of size n >= 2 into two of sizes a, n - a."""
    starts, events = item
    ev = list(events)
    for _ in range(int(rng.integers(0, max_k - len(ev) + 1))):
        big = [i for i, e in enumerate(ev) if e[2] >= 2]
        if not big:
            break
        i = big[int(rng.integers(0, len(big)))]
        who, kind, n = ev[i]
        a = int(rng.integers(1, n))
        ev[i] = (who, kind, a)
        if kind == 0:   # the rest of a gain comes earlier: amounts in between only grow
            ev.insert(int(rng.integers(0, i + 1)), (who, 0, n - a))
        else:           # the rest of a loss comes later: amounts in between only grow
            ev.insert(int(rng.integers(i + 1, len(ev) + 1)), (who, 1, n - a))
    return (starts, ev)
