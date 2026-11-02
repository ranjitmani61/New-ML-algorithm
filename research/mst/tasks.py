"""Signed-sum expressions and the relations used for label-free training.

An expression is a list of signed single-digit terms, written "+3-7+5=" (every term carries its
sign). Its value is the sum. The answer is written as a sign and two digits, least significant
first ("-23" -> "-", 3, 2), so every answer has the same three tokens.

Relations between expressions of the same length (no new terms, no labels needed):
  invariance   permute the terms                      value unchanged
  equivariant  change one term t_i -> t_i + d (0..9)   value + s_i * d
  equivariant  flip every sign                         value negated
"""

from __future__ import annotations

import numpy as np

PAD, PLUS, MINUS, EQ = 0, 11, 12, 13
VOCAB = 14
MAX_TERMS = 9
TERM_MAX = 9
ANS_DIGITS = 2
MAX_IN = (1 + len(str(TERM_MAX))) * MAX_TERMS + 1  # sign + digits per term, then "="
ANS = 1 + ANS_DIGITS  # sign + reversed digits
SEQ = MAX_IN + ANS


def digit(c):
    return 1 + c


def random_terms(rng, n):
    signs = rng.choice([1, -1], size=n)
    values = rng.integers(0, TERM_MAX + 1, size=n)
    return [(int(s), int(v)) for s, v in zip(signs, values)]


def value(terms):
    return sum(s * v for s, v in terms)


def encode_input(terms):
    toks = []
    for s, v in terms:
        toks.append(PLUS if s > 0 else MINUS)
        toks.extend(digit(int(c)) for c in str(v))
    toks.append(EQ)
    return toks


def encode_answer(v):
    if abs(v) >= 10 ** ANS_DIGITS:
        raise ValueError(v)
    digits = [int(c) for c in f"{abs(v):0{ANS_DIGITS}d}"][::-1]
    return [PLUS if v >= 0 else MINUS] + [digit(d) for d in digits]


def decode_answer(toks):
    """Integer from the ANS answer tokens, or None if malformed."""
    toks = [int(t) for t in toks]
    if len(toks) != ANS or toks[0] not in (PLUS, MINUS) or any(not 1 <= t <= 10 for t in toks[1:]):
        return None
    v = sum((t - 1) * 10 ** i for i, t in enumerate(toks[1:]))
    return v if toks[0] == PLUS else -v


def batch(term_lists, answers=None):
    """Left-padded int32 array [B, SEQ]; answers (ints) fill the last ANS slots if given."""
    out = np.zeros((len(term_lists), SEQ), dtype=np.int32)
    for i, terms in enumerate(term_lists):
        x = encode_input(terms)
        out[i, MAX_IN - len(x):MAX_IN] = x
        if answers is not None and answers[i] is not None:
            out[i, MAX_IN:] = encode_answer(answers[i])
    return out


# ---------------------------------------------------------------- relations


def permute(rng, terms):
    order = rng.permutation(len(terms))
    return [terms[j] for j in order], (lambda a: a)  # map an answer of the transform back to the original


def shift_term(rng, terms):
    i = int(rng.integers(len(terms)))
    s, v = terms[i]
    d = int(rng.integers(-v, TERM_MAX - v + 1))
    while d == 0 and TERM_MAX > 0:
        d = int(rng.integers(-v, TERM_MAX - v + 1))
    new = list(terms)
    new[i] = (s, v + d)
    delta = s * d
    return new, (lambda a, delta=delta: a - delta)


def flip_signs(rng, terms):
    return [(-s, v) for s, v in terms], (lambda a: -a)


def shift_term_wrong(rng, terms):
    """Placebo: the same transform as shift_term, mapped back with the wrong law (opposite sign)."""
    new, back = shift_term(rng, terms)
    delta = value(new) - value(terms)
    return new, (lambda a, delta=delta: a + delta)


def shift_all(rng, terms):
    """Equivariant: shift every term by its own d_i (staying in 0..9); value changes by sum s_i * d_i."""
    new, delta = [], 0
    for s, v in terms:
        d = int(rng.integers(-v, TERM_MAX - v + 1))
        new.append((s, v + d))
        delta += s * d
    return new, (lambda a, delta=delta: a - delta)


def shift_all_wrong(rng, terms):
    """Placebo for shift_all: same transform, mapped back with the opposite sign."""
    new, _ = shift_all(rng, terms)
    delta = value(new) - value(terms)
    return new, (lambda a, delta=delta: a + delta)
