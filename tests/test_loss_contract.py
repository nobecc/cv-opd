"""OPD math and serialized-cache adversarial checks (real CPU autograd)."""
import copy
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from finance_opd.advantages import compute_advantages
from finance_opd.rollout import samples_from_records


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
class LossContractTests(unittest.TestCase):
    def setUp(self):
        import torch
        from upstream_cpu import load_core
        self.torch = torch
        self.core = load_core()
        self.args = SimpleNamespace(use_opd=False, normalize_advantages=False,
                                   use_rollout_logprobs=True, opd_kl_coef=0.7)

    def test_nonuniform_policy_gradient_equals_local_reverse_kl_at_sampling_policy(self):
        t = self.torch
        logits = t.tensor([-2., 0.4, 1.1], dtype=t.float64, requires_grad=True)
        teacher = t.tensor([0.7, 0.2, 0.1], dtype=t.float64).log()
        old = logits.log_softmax(-1).detach()
        batch = dict(rollout_log_probs=[old], teacher_log_probs=[teacher], rewards=[0.6])
        compute_advantages(self.args, batch)
        current = logits.log_softmax(-1)
        terms, _ = self.core.compute_policy_loss(old-current, batch['advantages'][0], 0.2, 0.2)
        # Enumerate actions with their actual behavior probabilities, not a
        # uniform average (which only proves correctness for a uniform policy).
        surrogate = (old.exp()*terms).sum()
        exact = 0.7*0.6*(current.exp()*(current-teacher)).sum()
        g1 = t.autograd.grad(surrogate, logits, retain_graph=True)[0]
        g2 = t.autograd.grad(exact, logits)[0]
        t.testing.assert_close(g1, g2, atol=1e-12, rtol=1e-12)
        self.assertFalse(batch['advantages'][0].requires_grad)

    def test_invalid_tensor_and_coefficient_fail_even_for_zero_weight(self):
        t = self.torch
        for bad in [float('nan'), float('inf'), -float('inf')]:
            for column in ['rollout_log_probs', 'teacher_log_probs']:
                batch = dict(rollout_log_probs=[t.tensor([-1.])], teacher_log_probs=[t.tensor([-2.])], rewards=[0])
                batch[column] = [t.tensor([bad])]
                with self.subTest(column=column, bad=bad), self.assertRaisesRegex(ValueError, 'finite'):
                    compute_advantages(self.args, batch)
        for coefficient in [-1, 0, True, float('nan'), float('inf')]:
            self.args.opd_kl_coef = coefficient
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                compute_advantages(self.args, dict(rollout_log_probs=[t.tensor([-1.])],
                    teacher_log_probs=[t.tensor([-2.])], rewards=[1]))

    def test_low_precision_inputs_promoted_and_empty_cp_slice_supported(self):
        t = self.torch
        for dtype in [t.float16, t.bfloat16]:
            batch = dict(rollout_log_probs=[t.tensor([-2.], dtype=dtype), t.empty(0, dtype=dtype)],
                         teacher_log_probs=[t.tensor([-1.], dtype=dtype), t.empty(0, dtype=dtype)], rewards=[1, 1])
            compute_advantages(self.args, batch)
            self.assertEqual(batch['advantages'][0].dtype, t.float32)
            t.testing.assert_close(batch['advantages'][0], t.tensor([0.7]))
            self.assertEqual(batch['advantages'][1].numel(), 0)

    def test_cache_rejects_corruption_and_preserves_original_values(self):
        parent = self.core.Sample(index=1, group_index=1, prompt='task', session_id='cache-test')
        row = dict(turn=0, prompt_ids=[1, 2], response_ids=[3, 4], rollout_log_probs=[-1., -2.],
                   teacher_log_probs=[-0.5, -3.], loss_mask=[0, 1], response='fixture',
                   finish_reason='stop', weight_version='v1', opd_weight=1., student_prefix_sha256='fixture')
        result = samples_from_records(parent, [row], {}, Path('fixture'), self.core.Sample)[0]
        self.assertEqual(result.tokens, [1, 2, 3, 4])
        self.assertEqual(result.rollout_log_probs, row['rollout_log_probs'])
        self.assertEqual(result.teacher_log_probs, row['teacher_log_probs'])
        self.assertEqual(result.loss_mask, [0, 1])
        self.assertEqual(result.weight_versions, ['v1'])
        for key, value in [('response_ids', [True, 4]), ('prompt_ids', []),
                           ('rollout_log_probs', [float('nan'), -2.]),
                           ('teacher_log_probs', [-1.]), ('loss_mask', [1, 2]),
                           ('opd_weight', -1)]:
            bad = copy.deepcopy(row); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                samples_from_records(parent, [bad], {}, Path('fixture'), self.core.Sample)
        other = dict(row, weight_version='v2')
        with self.assertRaisesRegex(ValueError, 'weights changed'):
            samples_from_records(parent, [row, other], {}, Path('fixture'), self.core.Sample)
