"""Word-problem generator for the language-model test of cancelling-event insertion.

A problem: a person starts with some amount of an item, then K events change the amount, then a
question asks for the final amount. The model must answer with the number only (no worked
steps), which is the faithful analogue of the toy result: an inserted cancelling pair needs no
new label.

Families
  additive  events add or remove 1-9 items (the amount never goes negative)
  mixed     additive events plus "doubles" / "loses half" (order-dependent, like S3)

Expansions (answer unchanged, checked by tests)
  cancel_spread    gain a ... (later, anywhere) ... loss a          additive family only
  cancel_adjacent  gain a immediately followed by loss a            (mixed: doubles then halves)
                   "adjacent" means adjacent when inserted: a later-inserted pair can land between
                   them, which keeps the answer exact because every inserted block is an identity
  inactive         events that change nothing ("counts them")       length control
  irrelevant       numeric sentences about something else           PROBLEMATHIC/NoOp-style control
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

NAMES_TRAIN = ["Mia", "Leo", "Ava", "Noah", "Zoe", "Liam", "Emma", "Omar", "Lucy", "Ravi",
               "Sara", "Kai", "Nina", "Theo", "Ivy", "Sam", "Maya", "Eli", "Lena", "Arjun"]
NAMES_HELDOUT = ["Priya", "Hugo", "Wren", "Tariq", "Elsa", "Jonah", "Mei", "Felix", "Anya", "Rohan"]
ITEMS_TRAIN = ["apples", "marbles", "stickers", "coins", "pencils", "shells", "cards", "stamps",
               "beads", "cookies"]
ITEMS_HELDOUT = ["buttons", "crayons", "ribbons", "tokens", "acorns"]
OTHER_THINGS = ["chairs", "windows", "books", "lamps", "trees", "bikes", "cups", "doors"]

GAIN = ["{name} buys {n} more {item}.", "{other} gives {name} {n} {item}.",
        "{name} finds {n} {item}.", "{name} wins {n} {item}."]
LOSS = ["{name} gives {n} {item} to {other}.", "{name} loses {n} {item}.",
        "{name} sells {n} {item}.", "{name} uses {n} {item}."]
INACTIVE = ["{name} counts the {item}.", "{name} puts the {item} in a box.",
            "{name} shows the {item} to {other}.", "{name} looks at the {item}."]
IRRELEVANT = ["{other} has {n} {thing}.", "The room has {n} {thing}.",
              "{other} sees {n} {thing}.", "There are {n} {thing} outside."]
DOUBLE = "{name}'s {item} double."
HALVE = "{name} loses half of the {item}."


@dataclass
class Event:
    kind: str          # gain, loss, double, halve, inactive, irrelevant
    n: int = 0
    text: str = ""


@dataclass
class Problem:
    name: str
    item: str
    start: int
    events: list = field(default_factory=list)

    def answer(self):
        x = self.start
        for e in self.events:
            if e.kind == "gain":
                x += e.n
            elif e.kind == "loss":
                x -= e.n
            elif e.kind == "double":
                x *= 2
            elif e.kind == "halve":
                x //= 2
        return x

    def amounts(self):
        """Running amount after each event (used to check validity)."""
        x, out = self.start, []
        for e in self.events:
            if e.kind == "gain":
                x += e.n
            elif e.kind == "loss":
                x -= e.n
            elif e.kind == "double":
                x *= 2
            elif e.kind == "halve":
                x //= 2
            out.append(x)
        return out

    def valid(self):
        """Amounts stay non-negative and every halving is exact (even amount)."""
        x = self.start
        for e in self.events:
            if e.kind == "halve" and x % 2:
                return False
            if e.kind == "loss" and e.n > x:
                return False
            x = {"gain": x + e.n, "loss": x - e.n, "double": 2 * x, "halve": x // 2}.get(e.kind, x)
        return True

    def text(self):
        parts = [f"{self.name} has {self.start} {self.item}."] + [e.text for e in self.events]
        parts.append(f"How many {self.item} does {self.name} have now?")
        return " ".join(parts)

    def n_changes(self):
        return sum(e.kind in ("gain", "loss", "double", "halve") for e in self.events)


def singular(item):
    """apples -> apple, cookies -> cookie (all item and thing names are regular plurals)."""
    return item[:-1]


def _other(rng, name, names):
    return rng.choice([x for x in names if x != name])


def _render(rng, kind, n, name, item, names):
    other = _other(rng, name, names)
    if kind == "gain":
        t = rng.choice(GAIN)
    elif kind == "loss":
        t = rng.choice(LOSS)
    elif kind == "double":
        t = DOUBLE
    elif kind == "halve":
        t = HALVE
    elif kind == "inactive":
        t = rng.choice(INACTIVE)
    else:
        t = rng.choice(IRRELEVANT)
    thing = rng.choice(OTHER_THINGS)
    return t.format(name=name, n=n, item=singular(item) if n == 1 else item, other=other,
                    thing=singular(thing) if n == 1 else thing)


def make_problem(rng, k, family="additive", heldout=False):
    """A valid problem with exactly k state-changing events."""
    names = NAMES_HELDOUT if heldout else NAMES_TRAIN
    items = ITEMS_HELDOUT if heldout else ITEMS_TRAIN
    while True:
        name, item = rng.choice(names), rng.choice(items)
        start = rng.randint(5, 30)
        p = Problem(name, item, start)
        x = start
        ok = True
        for _ in range(k):
            kinds = ["gain", "loss"] + (["double", "halve"] if family == "mixed" else [])
            kind = rng.choice(kinds)
            if kind == "loss" and x == 0:
                kind = "gain"
            if kind == "halve" and x % 2:
                kind = "gain"
            if kind == "double" and x > 60:
                kind = "loss"
            n = 0
            if kind == "gain":
                n = rng.randint(1, 9)
                x += n
            elif kind == "loss":
                n = rng.randint(1, min(9, x))
                x -= n
            elif kind == "double":
                x *= 2
            else:
                x //= 2
            p.events.append(Event(kind, n, _render(rng, kind, n, name, item, names)))
        if p.valid() and p.answer() <= 999:
            return p


# ---------------------------------------------------------------- expansions (answer unchanged)


def _copy(p):
    return Problem(p.name, p.item, p.start, list(p.events))


def cancel_spread(rng, p, max_changes, family="additive"):
    """Insert pairs gain a ... loss a, the loss anywhere after its gain (additive family only)."""
    if family != "additive":
        raise ValueError("spread pairs do not cancel across doubling or halving; additive family only")
    q = _copy(p)
    k = rng.randint(0, (max_changes - p.n_changes()) // 2)
    names = NAMES_HELDOUT if p.name in NAMES_HELDOUT else NAMES_TRAIN
    for _ in range(k):
        a = rng.randint(1, 9)
        i = rng.randint(0, len(q.events))
        q.events.insert(i, Event("gain", a, _render(rng, "gain", a, p.name, p.item, names)))
        j = rng.randint(i + 1, len(q.events))
        q.events.insert(j, Event("loss", a, _render(rng, "loss", a, p.name, p.item, names)))
    return q


def cancel_adjacent(rng, p, max_changes, family="additive"):
    """Insert adjacent cancelling pairs: gain a then loss a; in the mixed family, half of the pairs
    are doubles-then-loses-half instead (both cancel exactly because they are adjacent)."""
    q = _copy(p)
    k = rng.randint(0, (max_changes - p.n_changes()) // 2)
    names = NAMES_HELDOUT if p.name in NAMES_HELDOUT else NAMES_TRAIN
    for _ in range(k):
        i = rng.randint(0, len(q.events))
        if family == "mixed" and rng.random() < 0.5:
            pair = [Event("double", 0, _render(rng, "double", 0, p.name, p.item, names)),
                    Event("halve", 0, _render(rng, "halve", 0, p.name, p.item, names))]
        else:
            a = rng.randint(1, 9)
            pair = [Event("gain", a, _render(rng, "gain", a, p.name, p.item, names)),
                    Event("loss", a, _render(rng, "loss", a, p.name, p.item, names))]
        q.events[i:i] = pair
    return q


def _insert_sentences(rng, p, max_changes, kind, family="additive"):
    q = _copy(p)
    m = 2 * rng.randint(0, (max_changes - p.n_changes()) // 2)  # same count as a pair expansion
    names = NAMES_HELDOUT if p.name in NAMES_HELDOUT else NAMES_TRAIN
    for _ in range(m):
        n = rng.randint(1, 9)
        q.events.insert(rng.randint(0, len(q.events)), Event(kind, n, _render(rng, kind, n, p.name, p.item, names)))
    return q


def inactive(rng, p, max_changes, family="additive"):
    return _insert_sentences(rng, p, max_changes, "inactive")


def irrelevant(rng, p, max_changes, family="additive"):
    return _insert_sentences(rng, p, max_changes, "irrelevant")


EXPANSIONS = {"cancel_spread": cancel_spread, "cancel_adjacent": cancel_adjacent,
              "inactive": inactive, "irrelevant": irrelevant}


def key(p):
    return hashlib.sha256(p.text().encode()).hexdigest()


def prompt_and_target(p):
    return p.text(), str(p.answer())


def demo(seed=0):
    rng = random.Random(seed)
    p = make_problem(rng, 3)
    return {"base": p.text(), "answer": p.answer(),
            **{name: f(rng, p, 9).text() for name, f in EXPANSIONS.items()}}  # additive family
