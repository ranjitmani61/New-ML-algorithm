"""Second testbed: state tracking in the permutation group S3.

Input: a word g_1 ... g_k of permutations of {0, 1, 2} followed by "="; answer: the composition
g_k o ... o g_1 (one token). Order matters and each step depends on the previous state, so longer
words need genuinely more sequential computation (unlike sums).

Answer-preserving expansions use only local identities, so no answer of a long word is ever
computed to build a training example:
  identity   insert identity elements e                   (analogue of +0)
  pairs      insert adjacent pairs g, g^-1                 (analogue of +a, -a)
  blocks     insert short identity blocks of length 2-3:   a, b, (b o a)^-1 / a, a^-1
"""

from __future__ import annotations

import itertools

import numpy as np

ELEMS = [tuple(p) for p in itertools.permutations(range(3))]  # ELEMS[0] is the identity
INDEX = {g: i for i, g in enumerate(ELEMS)}
PAD, EQ = 0, 7
VOCAB = 8
ANS = 1


def compose(a, b):
    """(a o b)[i] = a[b[i]]: apply b first, then a (indices into ELEMS)."""
    A, B = ELEMS[a], ELEMS[b]
    return INDEX[tuple(A[B[i]] for i in range(3))]


INV = [next(j for j in range(6) if compose(i, j) == 0) for i in range(6)]


def product(word):
    p = 0
    for g in word:
        p = compose(g, p)
    return p


def random_word(rng, k):
    return [int(x) for x in rng.integers(0, 6, size=k)]


def encode_input(word):
    return [1 + g for g in word] + [EQ]


def encode_answer(p):
    return [1 + p]


def decode_answer(toks):
    t = int(toks[0])
    return t - 1 if 1 <= t <= 6 else None


def batch(words, answers=None, max_in=None):
    enc = [encode_input(w) for w in words]
    max_in = max_in or max(len(e) for e in enc)
    out = np.zeros((len(enc), max_in + ANS), dtype=np.int32)
    for i, e in enumerate(enc):
        out[i, max_in - len(e):max_in] = e
        if answers is not None:
            out[i, max_in:] = encode_answer(answers[i])
    return out


def _insert_blocks(rng, word, blocks):
    out = list(word)
    for b in blocks:
        i = int(rng.integers(0, len(out) + 1))
        out[i:i] = b
    return out


def expand_identity(rng, word, max_len):
    m = int(rng.integers(0, max_len - len(word) + 1))
    return _insert_blocks(rng, word, [[0]] * m)


def expand_pairs(rng, word, max_len):
    k = int(rng.integers(0, (max_len - len(word)) // 2 + 1))
    blocks = []
    for _ in range(k):
        g = int(rng.integers(0, 6))
        blocks.append([g, INV[g]])
    return _insert_blocks(rng, word, blocks)


def expand_blocks(rng, word, max_len):
    blocks, room = [], max_len - len(word)
    while room >= 2:
        if room >= 3 and rng.random() < 0.5:
            a, b = int(rng.integers(0, 6)), int(rng.integers(0, 6))
            blocks.append([a, b, INV[compose(b, a)]])
            room -= 3
        else:
            g = int(rng.integers(0, 6))
            blocks.append([g, INV[g]])
            room -= 2
        if rng.random() < 0.3:
            break
    return _insert_blocks(rng, word, blocks)
