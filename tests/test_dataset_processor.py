"""Regression for Slime Dataset -> finance task text with a loaded processor."""
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from finance_opd.config import read_config
from finance_opd.launch import training_arguments
from finance_opd.rollout import task_text
from upstream_cpu import load_definitions


class DatasetProcessorTests(unittest.TestCase):
    def test_upstream_dataset_processor_keeps_original_native_task(self):
        config = read_config()
        source = Path(config["slime"]) / "slime/utils/data.py"
        objective = "Analyze company with CIK 6201\nKeep this original task."
        metadata = {"id": "ddr_10k:6201__entity", "benchmark": "ddr_10k"}
        namespace = {"Sample": SimpleNamespace, "np": SimpleNamespace(ndarray=type(None)),
                     "logger": logging.getLogger(__name__),
                     "read_file": lambda _: [{"task": objective, "metadata": metadata.copy()}]}
        load_definitions(source, ["_build_messages"], namespace)
        load_definitions(source, ["__init__"], namespace, class_name="Dataset")
        visits = []
        def process_vision_info(messages, processor):
            visits.append(messages)
            return {"images": None, "videos": None}
        with patch.dict(sys.modules, {"slime.utils.processing_utils": SimpleNamespace(
                process_vision_info=process_vision_info)}):
            with self.assertRaisesRegex(AssertionError, "prompt must be a list"):
                namespace["__init__"](SimpleNamespace(), "tasks.jsonl", object(), object(),
                                      None, prompt_key="task")
            args = training_arguments(config)
            keys = json.loads(args[args.index("--multimodal-keys") + 1])
            self.assertNotIn("--apply-chat-template", args)
            dataset = SimpleNamespace()
            namespace["__init__"](dataset, "tasks.jsonl", object(), object(), None,
                                  prompt_key="task", multimodal_keys=keys)
        self.assertEqual(visits, [[{"role": "user", "content": objective}]])
        sample = dataset.samples[0]
        self.assertEqual(task_text(sample.prompt), objective)
        self.assertEqual(sample.metadata, metadata)

    def test_does_not_flatten_history_or_multimodal_content_into_task(self):
        self.assertEqual(task_text("original task"), "original task")
        for prompt in [[], [{"role": "system", "content": "task"}],
                       [{"role": "user", "content": [{"type": "text", "text": "task"}]}],
                       [{"role": "user", "content": "task"}, {"role": "assistant", "content": "answer"}]]:
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                task_text(prompt)


if __name__ == "__main__":
    unittest.main()
