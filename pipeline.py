"""Generator -> exact math checker pipeline (metadata only; no data is read).

Each generated hypothesis is audited by math_checker. A hypothesis passes the
mathematical-obligation gate only when every extracted subclaim is VERIFIED
and every certificate replays. Passing means "the declared bounded-coefficient
obligations hold"; it does NOT verify novelty or empirical improvement, so the
whole-hypothesis status stays UNVERIFIED and the ten scientific gates are not
modified (the generator rejects forged gates).
"""
from __future__ import annotations
import argparse, json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ('generator_only', 'math_verifier'):
    sys.path.insert(0, os.path.join(_HERE, _d))

import hypothesis_generator as hg
from math_checker import MathCheckerEngine, verify_certificate


def audit(hypothesis, engine=None):
    """Return a sidecar audit dict for one Hypothesis or its to_dict() record."""
    engine = engine or MathCheckerEngine()
    record = hypothesis.to_dict() if hasattr(hypothesis, 'to_dict') else hypothesis
    if not hasattr(hypothesis, 'to_dict'):
        # math_checker validates only the definition signature, so forged gates or
        # equations in an otherwise valid record would pass it. The generator's
        # strict round-trip import catches those.
        try:
            hypothesis = hg.Hypothesis.from_dict(record)
        except Exception as e:
            return {'hid': record.get('hid') if isinstance(record, dict) else None,
                    'signature': None, 'whole_status': 'INVALID', 'subclaims': 0,
                    'math_gate_passed': False, 'failed_obligations': [],
                    'replay_failures': [], 'scope': f'generator import rejected record: {e}'}
    res = engine.check(hypothesis)
    subs = res.subclaims or []
    bad, replay_fail = [], []
    for s in subs:
        v = s['verdict']
        if v['status'] != 'VERIFIED':
            bad.append({'obligation': s['obligation'], 'status': v['status']})
        elif not verify_certificate(s['claim'], v['certificate']):
            replay_fail.append(s['obligation'])
    ok = bool(subs) and not bad and not replay_fail and res.status != 'INVALID'
    return {
        'hid': record.get('hid'), 'signature': record.get('signature'),
        'whole_status': res.status, 'subclaims': len(subs),
        'math_gate_passed': ok, 'failed_obligations': bad,
        'replay_failures': replay_fail,
        'scope': 'conditional bounded-coefficient obligations only; novelty and empirical claims remain UNVERIFIED',
    }


def run(kind, features, classes, goals, count, seed):
    task = hg.TaskSpec(kind=kind, features=features, classes=classes, goals=tuple(goals))
    engine = MathCheckerEngine()
    for h in hg.HypothesisGenerator(seed=seed).generate(task, count):
        yield h, audit(h, engine)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--kind', default='classification', choices=['classification', 'regression'])
    p.add_argument('--features', type=int, default=8)
    p.add_argument('--classes', type=int, default=3)
    p.add_argument('--goal', action='append', default=[])
    p.add_argument('--count', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--emit-passed', action='store_true', help='print passing hypothesis records + audit as JSONL')
    a = p.parse_args(argv)
    passed = total = 0
    for h, au in run(a.kind, a.features, a.classes, a.goal, a.count, a.seed):
        total += 1; passed += au['math_gate_passed']
        if a.emit_passed and au['math_gate_passed']:
            print(json.dumps({'hypothesis': h.to_dict(), 'math_audit': au}))
    print(json.dumps({'total': total, 'math_gate_passed': passed}), file=sys.stderr)
    return 0 if passed == total else 1


if __name__ == '__main__':
    raise SystemExit(main())
