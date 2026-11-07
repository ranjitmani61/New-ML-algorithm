"""Three-person word problems whose answer is the largest final amount, for the test where answers
cannot be composed from short problems' answers.

A problem: three people each start with some amount of one item. K events each change one
person's amount (a gain or a loss of 1-9). The question asks for the largest final amount:

    Mia has 12 apples, Leo has 7 apples and Ava has 20 apples. Leo finds 9 apples. Ava sells
    4 apples. How many apples does the person with the most apples have now?   -> 16

The answer is max(final amounts). Appending events to a labeled problem changes it in a way that
the label alone cannot tell (it depends on every person's amount), so chaining by composing
labels, which needs a way to compute the new answer, does not apply.

Expansion (answer unchanged, and built without simulating the problem)
  cancel_spread  person P gains a ... (later, anywhere) ... P loses a. The gain comes first, so no
                 amount can go negative, and every final amount is unchanged. So the answer to any
                 question about the final amounts is unchanged, whatever the question is.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from problems import GAIN, ITEMS_HELDOUT, ITEMS_TRAIN, LOSS, NAMES_HELDOUT, NAMES_TRAIN, singular

QUESTION = "How many {item} does the person with the most {item} have now?"
N_PEOPLE = 3


@dataclass
class Event:
    who: int           # index into Problem.people
    kind: str          # gain or loss
    n: int
    text: str = ""


@dataclass
class Problem:
    people: list
    item: str
    starts: list
    events: list = field(default_factory=list)

    def amounts(self):
        """Every person's amount after each event (used to check validity)."""
        x, out = list(self.starts), []
        for e in self.events:
            x[e.who] += e.n if e.kind == "gain" else -e.n
            out.append(list(x))
        return out

    def finals(self):
        return self.amounts()[-1] if self.events else list(self.starts)

    def answer(self):
        return max(self.finals())

    def valid(self):
        """No amount ever goes negative."""
        return min([min(self.starts)] + [min(a) for a in self.amounts()]) >= 0

    def text(self):
        def has(name, n):
            return f"{name} has {n} {singular(self.item) if n == 1 else self.item}"
        a, b, c = (has(nm, s) for nm, s in zip(self.people, self.starts))
        parts = [f"{a}, {b} and {c}."] + [e.text for e in self.events]
        parts.append(QUESTION.format(item=self.item))
        return " ".join(parts)

    def n_changes(self):
        return len(self.events)


def _render(rng, kind, n, people, who, item, names):
    other = rng.choice([x for x in names if x not in people])  # never one of the three
    t = rng.choice(GAIN if kind == "gain" else LOSS)
    return t.format(name=people[who], n=n, item=singular(item) if n == 1 else item, other=other)


def make_multi(rng, k, heldout=False):
    """A valid problem with exactly k events."""
    names = NAMES_HELDOUT if heldout else NAMES_TRAIN
    items = ITEMS_HELDOUT if heldout else ITEMS_TRAIN
    while True:
        people = rng.sample(names, N_PEOPLE)
        item = rng.choice(items)
        starts = [rng.randint(5, 30) for _ in range(N_PEOPLE)]
        p = Problem(people, item, starts)
        x = list(starts)
        for _ in range(k):
            who = rng.randrange(N_PEOPLE)
            kind = rng.choice(["gain", "loss"])
            if kind == "loss" and x[who] == 0:
                kind = "gain"
            n = rng.randint(1, 9) if kind == "gain" else rng.randint(1, min(9, x[who]))
            x[who] += n if kind == "gain" else -n
            p.events.append(Event(who, kind, n, _render(rng, kind, n, people, who, item, names)))
        if p.valid() and p.answer() <= 999:
            return p


def _copy(p):
    return Problem(list(p.people), p.item, list(p.starts), list(p.events))


def cancel_spread(rng, p, max_changes):
    """Insert pairs: person P gains a, and P loses a anywhere later. Uses no amounts at all."""
    q = _copy(p)
    k = rng.randint(0, (max_changes - p.n_changes()) // 2)
    names = NAMES_HELDOUT if p.people[0] in NAMES_HELDOUT else NAMES_TRAIN
    for _ in range(k):
        who = rng.randrange(N_PEOPLE)
        a = rng.randint(1, 9)
        i = rng.randint(0, len(q.events))
        q.events.insert(i, Event(who, "gain", a, _render(rng, "gain", a, q.people, who, q.item, names)))
        j = rng.randint(i + 1, len(q.events))
        q.events.insert(j, Event(who, "loss", a, _render(rng, "loss", a, q.people, who, q.item, names)))
    return q


def key(p):
    return hashlib.sha256(p.text().encode()).hexdigest()


def demo(seed=0):
    rng = random.Random(seed)
    p = make_multi(rng, 3)
    return {"base": p.text(), "answer": p.answer(), "cancel_spread": cancel_spread(rng, p, 9).text()}
