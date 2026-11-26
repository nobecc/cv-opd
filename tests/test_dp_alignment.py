import copy
import unittest
from types import SimpleNamespace

from finance_opd.rollout import align_static_samples


class DPAlignmentTests(unittest.TestCase):
    def sample(self, group, length=2):
        return SimpleNamespace(tokens=[1] * (length + 3), response_length=length,
                               loss_mask=[1] * length, reward={"opd_weight": 0.4},
                               group_id=group, metadata={"trainable_tokens": length})

    def config(self, gpus=8):
        return {"training": dict(gpus=gpus, tensor_parallel_size=4,
                                 pipeline_parallel_size=1, context_parallel_size=1,
                                 qkv_format="bshd")}

    def test_variable_turn_episodes_preserve_group_and_supervision(self):
        # Actual failure was 21 turns across two episodes under TP4/DP2.
        a, b = [self.sample(0) for _ in range(9)], [self.sample(1) for _ in range(12)]
        original = copy.deepcopy(a + b)
        result = align_static_samples(a, self.config()) + align_static_samples(b, self.config())
        self.assertEqual(len(result), 22)
        for group in [0, 1]:
            before = sum(sum(s.loss_mask) for s in original if s.group_id == group)
            after = sum(sum(s.loss_mask) for s in result if s.group_id == group)
            self.assertEqual(before, after)
        padding = [s for s in result if s.metadata.get("dp_alignment_padding")]
        self.assertEqual(len(padding), 1)
        self.assertEqual(padding[0].reward, {"opd_weight": 0.0})
        self.assertEqual(a[0].loss_mask, [1, 1])
        self.assertFalse(a[0].metadata.get("dp_alignment_padding", False))

    def test_no_admission_episode_remains_zero_loss(self):
        sample = self.sample(4)
        sample.loss_mask = [0, 0]
        sample.reward = {"opd_weight": 0.0}
        result = align_static_samples([sample], self.config())
        self.assertEqual(len(result), 2)
        self.assertEqual(sum(sum(s.loss_mask) for s in result), 0)

    def test_upstream_static_schedule_accepts_previous_failure(self):
        # Exercise the existing Slime scheduler; no scheduler replacement.
        from finance_opd.config import add_dependency_paths, read_config
        add_dependency_paths(read_config())
        from slime.utils.dp_schedule import build_dp_schedule
        samples = (align_static_samples([self.sample(0) for _ in range(9)], self.config())
                   + align_static_samples([self.sample(1) for _ in range(12)], self.config()))
        args = SimpleNamespace(use_dynamic_batch_size=False, micro_batch_size=1,
                               balance_data=False)
        parallel = dict(dp_size=2, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1)
        parts, indices, counts, sizes = build_dp_schedule(args, parallel,
            [len(s.tokens) for s in samples], global_batch_size=2,
            group_indices=[s.group_id for s in samples])
        self.assertEqual(counts, [11])
        self.assertEqual(sizes, [2])
        self.assertEqual([len(p) for p in parts], [11, 11])
        self.assertEqual(sorted(parts[0] + parts[1]), list(range(22)))

    def test_dp1_does_not_pad(self):
        samples = [self.sample(1)]
        self.assertIs(align_static_samples(samples, self.config(4)), samples)
        self.assertEqual(len(samples), 1)
