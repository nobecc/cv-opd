import importlib.util
import unittest
from types import SimpleNamespace

from finance_opd.advantages import compute_advantages, post_process_weights


class AdvantageTests(unittest.TestCase):
    def test_weight_transport_supports_upstream_numeric_metrics(self):
        samples = [SimpleNamespace(reward={"opd_weight": w}) for w in [0, 0.5, 2]]
        raw, rewards = post_process_weights(None, samples)
        self.assertEqual(sum(raw), 2.5)
        self.assertEqual(rewards, [0, 0.5, 2])
        self.assertEqual(samples[1].reward, {"opd_weight": 0.5})
        for bad in [True, -1, float("nan"), 11, 0]:
            with self.assertRaises(ValueError):
                post_process_weights(None, [SimpleNamespace(reward={"opd_weight": bad})])
        with self.assertRaisesRegex(ValueError, "no positive masked supervision"):
            post_process_weights(None, [SimpleNamespace(reward={"opd_weight": 1}, loss_mask=[0, 0])])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "tensor advantage test requires torch")
    def test_positive_negative_and_zero_weight_advantages_are_detached(self):
        import torch
        student = torch.tensor([-1.0, -0.2], requires_grad=True)
        teacher = torch.tensor([-0.4, -1.4])
        args = SimpleNamespace(use_opd=False, normalize_advantages=False,
                               use_rollout_logprobs=True, opd_kl_coef=1.0)
        batch = {"rollout_log_probs": [student, student], "teacher_log_probs": [teacher, teacher],
                 "rewards": [0.5, 0]}
        compute_advantages(args, batch)
        torch.testing.assert_close(batch["advantages"][0], torch.tensor([0.3, -0.6]))
        torch.testing.assert_close(batch["advantages"][1], torch.zeros(2))
        self.assertFalse(batch["advantages"][0].requires_grad)
