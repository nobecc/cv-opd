import importlib.util
import tempfile
import unittest
from pathlib import Path

from finance_opd.config import ROOT, read_config


@unittest.skipUnless(importlib.util.find_spec("transformers"), "native smoke requires transformers and local tokenizer")
class NativeTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_pi_sqlite_and_exact_cached_tokens(self):
        config = read_config()
        if not (Path(config["student_model"]) / "tokenizer.json").is_file():
            self.skipTest("local tokenizer unavailable")
        from native_smoke import run
        (ROOT / "validation").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "validation") as directory:
            report = await run(Path(directory) / "smoke", config)
            self.assertTrue(report["passed"])
            self.assertFalse(report["gpu_training_executed"])

    async def test_cpu_queue_real_native_tools_and_exact_tokens(self):
        config = read_config()
        if not (Path(config["student_model"]) / "tokenizer.json").is_file():
            self.skipTest("local tokenizer unavailable")
        from native_smoke import run
        (ROOT / "validation").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "validation") as directory:
            report = await run(Path(directory) / "queue-smoke", config, queued=True)
            self.assertTrue(report["passed"])
            self.assertTrue(report["cpu_episode_queue"])
            self.assertEqual(report["generated_turns"],2)
