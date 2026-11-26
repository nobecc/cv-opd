import importlib.util
import json
import enum
import types
from unittest.mock import patch
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires CPU PyTorch")
class CheckpointVerificationTests(unittest.TestCase):
    def test_real_tensor_files_detect_changes_and_reject_identical_weights(self):
        import torch
        with tempfile.TemporaryDirectory(prefix="verification-fixture-", dir=ROOT / "runs") as temporary:
            directory = Path(temporary)
            output = directory / "training"
            (directory / "config.json").write_text(json.dumps({"output": str(output)}))
            files = [output / f"checkpoints/iter_{i:07d}/mp_rank_00/model_optim_rng.pt" for i in (0, 1)]
            for file in files:
                file.parent.mkdir(parents=True)
            # Metadata references a GPU-only Megatron enum unavailable to the
            # separate verification interpreter; tensor loading must still work.
            package = types.ModuleType("megatron.core.enums")
            package.ModelType = enum.Enum("ModelType", {"encoder_or_decoder": 1}, module=package.__name__)
            modules = {"megatron": types.ModuleType("megatron"),
                       "megatron.core": types.ModuleType("megatron.core"), package.__name__: package}
            with patch.dict(sys.modules, modules):
                torch.save({"model": {"layer.weight": torch.zeros(4, 4)},
                            "args": types.SimpleNamespace(model_type=package.ModelType.encoder_or_decoder)}, files[0])
            torch.save({"model": {"layer.weight": torch.ones(4, 4)}}, files[1])
            (output / "checkpoints/latest_checkpointed_iteration.txt").write_text("1")
            command = [sys.executable, "-B", str(ROOT / "scripts/verify_checkpoints.py"), str(directory)]
            changed = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(changed.returncode, 0, changed.stderr)
            report = json.loads((directory / "PARAMETER_VERIFICATION.json").read_text())
            self.assertEqual(report["changed_sampled_elements"], 16)
            self.assertTrue(report["sampled_only"])
            torch.save({"model": {"layer.weight": torch.zeros(4, 4)}}, files[1])
            identical = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(identical.returncode, 0)
            self.assertFalse(json.loads((directory / "PARAMETER_VERIFICATION.json").read_text())["passed"])
