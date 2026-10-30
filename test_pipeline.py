import copy, unittest
import pipeline
from pipeline import hg, MathCheckerEngine


class PipelineContract(unittest.TestCase):
    def test_generated_hypotheses_pass_math_gate(self):
        for kind, ncls in (('classification', 3), ('regression', 1)):
            for h, au in pipeline.run(kind, 6, ncls, ['outliers'], 15, 7):
                self.assertTrue(au['math_gate_passed'], au)
                self.assertEqual(au['whole_status'], 'UNVERIFIED')  # never promoted
                self.assertGreater(au['subclaims'], 0)

    def test_tampered_record_is_rejected(self):
        h = next(iter(pipeline.run('classification', 6, 3, [], 1, 1)))[0]
        d = copy.deepcopy(h.to_dict())
        d['algorithm']['mechanism'] = 'memory_step' if d['algorithm']['mechanism'] != 'memory_step' else 'clip_step'
        au = pipeline.audit(d)
        self.assertFalse(au['math_gate_passed'])

    def test_forged_gate_is_rejected(self):
        h = next(iter(pipeline.run('regression', 6, 1, [], 1, 2)))[0]
        d = copy.deepcopy(h.to_dict())
        d['acceptance_gates']['mathematical_foundation'] = 'PASS'
        self.assertFalse(pipeline.audit(d)['math_gate_passed'])

    def test_checker_can_actually_refute(self):
        # the gate logic must be able to fail: a false obligation is not VERIFIED
        claim = {'kind': 'scalar', 'variables': {'q': {'lower': '0', 'upper': '1'}},
                 'lhs': 'q', 'op': '<=', 'rhs': '1/2'}
        self.assertEqual(MathCheckerEngine().check(claim).status, 'DISPROVEN')


if __name__ == '__main__':
    unittest.main()
