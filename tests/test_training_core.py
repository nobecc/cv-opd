import copy
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from finance_opd.advantages import post_process_weights
from finance_opd.rollout import samples_from_records


@unittest.skipUnless(importlib.util.find_spec("torch"), "training math requires CPU or GPU PyTorch")
class TrainingCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from upstream_cpu import load_core
        torch.set_num_threads(1)
        cls.torch = torch
        cls.core = load_core()

    def setUp(self):
        self.core.ranks.size = 1
        self.core.ranks.rank = 0
        self.args = SimpleNamespace(use_opd=False, normalize_advantages=False, use_rollout_logprobs=True,
                                    opd_kl_coef=1.0, kl_coef=0, qkv_format="thd",
                                    custom_advantage_function_path="finance_opd.advantages.compute_advantages")

    def convert(self, samples):
        manager = SimpleNamespace(custom_convert_samples_to_train_data_func=None, args=self.args,
                                  _post_process_rewards=lambda values: post_process_weights(self.args, values))
        batch = self.core._convert_samples_to_train_data(manager, samples)
        batch["total_lengths"] = [len(s.tokens) for s in samples]
        batch["loss_masks"] = [self.torch.tensor(mask, dtype=self.torch.float64) for mask in batch["loss_masks"]]
        batch["group_mask_sums"] = [self.torch.tensor(n, dtype=self.torch.float64) for n in batch["group_mask_sums"]]
        for key in ["teacher_log_probs", "rollout_log_probs"]:
            batch[key] = [self.torch.tensor(lp, dtype=self.torch.float64) for lp in batch[key]]
        return batch

    def make_sample(self, index, action, old, teacher, weight=1.0, length=1):
        record = {"turn": 0, "prompt_ids": [17, 18, 19], "response_ids": [action] * length,
                  "rollout_log_probs": [old] * length, "teacher_log_probs": [teacher] * length,
                  "response": "fixture", "finish_reason": "stop", "weight_version": "cpu-1",
                  "opd_weight": weight, "student_prefix_sha256": "fixture"}
        parent = self.core.Sample(index=index, group_index=index, prompt="task", session_id=f"test-{index}")
        return samples_from_records(parent, [record], {"finished": True}, Path("cpu-fixture"), self.core.Sample)[0]

    def test_upstream_conversion_advantage_and_optimizer_follow_reverse_kl(self):
        torch = self.torch
        student = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        teacher = torch.tensor([0.8, 0.2], dtype=torch.float64).log()
        old = student.log_softmax(-1).detach()
        samples = [self.make_sample(i, i, old[i].item(), teacher[i].item(), weight=0.7) for i in range(2)]
        batch = self.convert(samples)
        self.core.compute_advantages_and_returns(self.args, batch)
        advantages = torch.cat(batch["advantages"])
        self.assertGreater(advantages[0].item(), 0)
        self.assertLess(advantages[1].item(), 0)
        self.assertFalse(advantages.requires_grad)
        loss_terms, _ = self.core.compute_policy_loss(old - student.log_softmax(-1), advantages, 0.2, 0.2)
        reducer = self.core.get_sum_of_sample_mean(batch["total_lengths"], batch["response_lengths"],
                                                  batch["loss_masks"], batch["group_mask_sums"])
        surrogate = reducer(loss_terms) / 2
        gradient = torch.autograd.grad(surrogate, student, retain_graph=True)[0]
        logp = student.log_softmax(-1)
        exact_kl = 0.7 * (logp.exp() * (logp - teacher)).sum()
        exact_gradient = torch.autograd.grad(exact_kl, student)[0]
        torch.testing.assert_close(gradient, exact_gradient, atol=1e-10, rtol=1e-10)
        before = student.detach().clone()
        optimizer = torch.optim.SGD([student], lr=0.1)
        surrogate.backward(); optimizer.step()
        after_kl = (student.log_softmax(-1).exp() * (student.log_softmax(-1) - teacher)).sum()
        self.assertLess(after_kl.item(), exact_kl.item() / 0.7)
        self.assertFalse(torch.equal(before, student))

    def test_zero_weight_blocks_gradient_and_ppo_clips_both_signs(self):
        torch = self.torch
        old = torch.tensor([-0.4, -0.7], dtype=torch.float64)
        batch = {"rollout_log_probs": [old], "teacher_log_probs": [old - 1], "rewards": [0]}
        self.core.compute_advantages_and_returns(self.args, batch)
        value = torch.nn.Parameter(old.clone())
        losses, _ = self.core.compute_policy_loss(old - value, batch["advantages"][0], 0.2, 0.2)
        losses.sum().backward()
        torch.testing.assert_close(value.grad, torch.zeros_like(value))
        # A positive token above 1.2 and a negative token below .8 both clip.
        ratio = torch.tensor([2.0, 0.5], dtype=torch.float64, requires_grad=True)
        loss, _ = self.core.compute_policy_loss(-ratio.log(), torch.tensor([1., -1.]), 0.2, 0.2)
        loss.sum().backward()
        torch.testing.assert_close(ratio.grad, torch.zeros_like(ratio))

    def test_unequal_episode_lengths_and_microbatches_do_not_change_episode_weight(self):
        torch = self.torch
        samples = [self.make_sample(0, 1, -1, -0.5, length=1),
                   self.make_sample(0, 1, -1, -0.5, length=3),
                   self.make_sample(1, 1, -1, -0.5, length=8)]
        batch = self.convert(samples)
        self.assertEqual([n.item() for n in batch["group_mask_sums"]], [4, 4, 8])
        values = [torch.full((1,), 2.), torch.full((3,), 2.), torch.full((8,), 10.)]
        total = 0
        # Each turn in a different microbatch must retain its episode denominator.
        for i in range(3):
            reduce = self.core.get_sum_of_sample_mean(batch["total_lengths"][i:i+1], batch["response_lengths"][i:i+1],
                                                       batch["loss_masks"][i:i+1], batch["group_mask_sums"][i:i+1])
            total += reduce(values[i])
        self.assertEqual(total.item() / 2, 6.0)
        self.assertNotAlmostEqual(total.item() / 2, torch.cat(values).mean().item())

    def test_cv_mining_and_context_masks_do_not_dilute_retry_gradient(self):
        torch = self.torch
        mining = self.make_sample(0, 1, -1, -0.5, weight=0, length=7)
        mining.loss_mask = [0] * 7
        retry = self.make_sample(0, 1, -1, -0.5, weight=0.18, length=2)
        batch = self.convert([mining, retry])
        self.assertEqual([n.item() for n in batch["group_mask_sums"]], [2, 2])
        first = torch.nn.Parameter(torch.full((7,), 100.0, dtype=torch.float64))
        second = torch.nn.Parameter(torch.full((2,), 2.0, dtype=torch.float64))
        reducer = self.core.get_sum_of_sample_mean(batch["total_lengths"], batch["response_lengths"],
            batch["loss_masks"], batch["group_mask_sums"])
        loss = reducer(torch.cat([first, second]))
        self.assertEqual(loss.item(), 2)
        loss.backward()
        torch.testing.assert_close(first.grad, torch.zeros(7, dtype=torch.float64))
        torch.testing.assert_close(second.grad, torch.full((2,), 0.5, dtype=torch.float64))

    def test_virtual_cp_1_2_4_8_preserves_loss_and_gradients_including_empty_ranks(self):
        torch = self.torch
        samples = [self.make_sample(0, 1, -1, -0.5, length=1),
                   self.make_sample(0, 1, -1, -1.3, length=17),
                   self.make_sample(1, 1, -1, -0.7, weight=0.5, length=3)]
        batch = self.convert(samples)
        expected = None
        for size in [1, 2, 4, 8]:
            self.core.ranks.size = size
            values = [torch.nn.Parameter(lp.clone()) for lp in batch["rollout_log_probs"]]
            loss = 0
            for rank in range(size):
                self.core.ranks.rank = rank
                local = copy.copy(batch)
                def sliced(column):
                    return [self.core.slice_log_prob_with_cp(lp, total, response)
                            for lp, total, response in zip(column, batch["total_lengths"], batch["response_lengths"])]
                local["rollout_log_probs"] = sliced(batch["rollout_log_probs"])
                local["teacher_log_probs"] = sliced(batch["teacher_log_probs"])
                self.core.compute_advantages_and_returns(self.args, local)
                ppo_kl = torch.cat(local["rollout_log_probs"]) - torch.cat(sliced(values))
                terms, _ = self.core.compute_policy_loss(ppo_kl, torch.cat(local["advantages"]), 0.2, 0.2)
                reduce = self.core.get_sum_of_sample_mean(batch["total_lengths"], batch["response_lengths"],
                                                          batch["loss_masks"], batch["group_mask_sums"])
                loss += reduce(terms) / 2
            loss.backward()
            result = (loss.detach(), torch.cat([v.grad for v in values]))
            if expected is None: expected = result
            else:
                torch.testing.assert_close(result[0], expected[0])
                torch.testing.assert_close(result[1], expected[1])
