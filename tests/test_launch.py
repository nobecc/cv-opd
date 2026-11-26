import unittest
from types import SimpleNamespace
from unittest.mock import patch

from finance_opd.config import read_config
from finance_opd.launch import training_arguments, doctor


class LaunchTests(unittest.TestCase):
    def test_offline_doctor_never_claims_live_training_readiness(self):
        with patch('finance_opd.launch.add_dependency_paths'), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('finance_opd.launch.importlib.util.find_spec', return_value=True), \
             patch('finance_opd.launch.assert_compatible_tokenizers'), \
             patch('finance_opd.teacher_transport.validate_teacher_config'), \
             patch('rl.examples.pi_finance.qwen_protocol.validate_native_parser', return_value=True), \
             patch('finance_opd.launch.subprocess.run', return_value=SimpleNamespace(returncode=0)):
            result = doctor(read_config(), offline=True)
        self.assertTrue(result['checks_passed'])
        self.assertFalse(result['training_ready'])
        self.assertFalse(result['teacher_checked'])

    def test_invalid_opd_coefficient_rejected_before_launch(self):
        for coefficient in [-1, 0, True, float("nan"), float("inf")]:
            config = read_config()
            config["training"]["opd_coef"] = coefficient
            with self.subTest(coefficient=coefficient), self.assertRaisesRegex(ValueError, "opd coefficient"):
                training_arguments(config)

    def test_cv_training_filters_empty_supervision_before_optimizer(self):
        config = read_config()
        config["continuation"]["enabled"] = True
        args = training_arguments(config)
        self.assertEqual(args[args.index("--dynamic-sampling-filter-path") + 1],
                         "finance_opd.training_filter.require_cv_signal")
        config["continuation"]["enabled"] = False
        self.assertNotIn("--dynamic-sampling-filter-path", training_arguments(config))

    def test_cv_can_disable_decay_on_zero_loss_placeholders(self):
        config = read_config()
        config["training"]["weight_decay"] = 0.0
        args = training_arguments(config)
        self.assertEqual(args[args.index("--weight-decay") + 1], "0.0")
        config["training"]["weight_decay"] = float("nan")
        with self.assertRaisesRegex(ValueError, "weight_decay"):
            training_arguments(config)

    def test_eight_gpu_data_parallel_preserves_tp4_checkpoint_shape(self):
        config = read_config()
        config["training"].update(gpus=8, tensor_parallel_size=4,
            context_parallel_size=1, pipeline_parallel_size=1, rollout_batch_size=2,
            rollout_tensor_parallel_size=4)
        args = training_arguments(config)
        self.assertEqual(args[args.index("--actor-num-gpus-per-node") + 1], "8")
        self.assertEqual(args[args.index("--tensor-model-parallel-size") + 1], "4")
        self.assertEqual(args[args.index("--expert-tensor-parallel-size") + 1], "4")
        self.assertEqual(args[args.index("--global-batch-size") + 1], "2")
        config["training"]["rollout_batch_size"] = 1
        with self.assertRaisesRegex(ValueError, "data parallel"):
            training_arguments(config)

    def test_27b_tensor_parallel_topology(self):
        config = read_config()
        config["training"].update(gpus=4, tensor_parallel_size=4,
                                  context_parallel_size=1, pipeline_parallel_size=1,
                                  model_args="scripts/models/qwen3.8-27B.sh")
        args = training_arguments(config)
        self.assertEqual(args[args.index("--tensor-model-parallel-size") + 1], "4")
        self.assertIn("--sequence-parallel", args)
        self.assertEqual(args[args.index("--expert-tensor-parallel-size") + 1], "4")
        config["training"]["context_parallel_size"] = 2
        with self.assertRaisesRegex(ValueError, "TP"):
            training_arguments(config)

    def test_bshd_uses_static_single_sequence_batches(self):
        config = read_config()
        config["training"]["qkv_format"] = "bshd"
        args = training_arguments(config)
        self.assertEqual(args[args.index("--qkv-format") + 1], "bshd")
        self.assertEqual(args[args.index("--micro-batch-size") + 1], "1")
        self.assertNotIn("--use-dynamic-batch-size", args)
        self.assertNotIn("--max-tokens-per-gpu", args)

    def test_partial_optimizer_offload_preserves_gpu_states_between_rollouts(self):
        config = read_config()
        config["training"].update(optimizer_offload_fraction=0.5, offload_train=False,
                                  rollout_tensor_parallel_size=4, rollout_memory_fraction=0.25, save_optimizer=False, checkpoint_format="torch")
        args = training_arguments(config)
        self.assertEqual(args[args.index("--optimizer-offload-fraction") + 1], "0.5")
        self.assertIn("--no-offload-train", args)
        self.assertIn("--offload-rollout", args)
        self.assertIn("--no-save-optim", args)
        self.assertEqual(args[args.index("--ckpt-format") + 1], "torch")
        self.assertIn("--use-precision-aware-optimizer", args)
        for bad in [0, -0.1, 1.1, float("nan"), float("inf")]:
            config["training"]["optimizer_offload_fraction"] = bad
            with self.assertRaisesRegex(ValueError, "optimizer_offload_fraction"):
                training_arguments(config)

    def test_episode_weighting_is_not_replaced_by_global_token_average(self):
        args = training_arguments(read_config())
        self.assertNotIn("--calculate-per-token-loss", args,
                         "Slime bypasses group_mask_sums when this flag is enabled")
        self.assertNotIn("--normalize-advantages", args)
        self.assertNotIn("--use-opd", args)
        self.assertIn("--use-rollout-logprobs", args)

    def test_native_chunking_and_weights_only_resume_preserve_iteration(self):
        config = read_config()
        config["training"].update(log_probs_chunk_size=512, recompute_loss_function=True,
                                  load_checkpoint="/fixture/checkpoints", reset_optimizer=True)
        args = training_arguments(config)
        self.assertEqual(args[args.index("--log-probs-chunk-size") + 1], "512")
        self.assertIn("--recompute-loss-function", args)
        self.assertIn("--no-load-optim", args)
        self.assertIn("--no-load-rng", args)
        self.assertNotIn("--finetune", args)
        self.assertEqual(args[args.index("--load") + 1], "/fixture/checkpoints")
        for value in [0, -1, True, 0.5]:
            config["training"]["log_probs_chunk_size"] = value
            with self.assertRaisesRegex(ValueError, "chunk_size"):
                training_arguments(config)
