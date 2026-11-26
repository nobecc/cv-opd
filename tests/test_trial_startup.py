import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from finance_opd.cluster_worker import teacher_model_support
from finance_opd.multinode_driver import add_master_no_proxy
from scripts.submit_trial import check_teacher_ready, worker_resources


class TrialStartupTests(unittest.TestCase):
    def test_two_eight_gpu_workers_obey_cluster_resource_policy(self):
        config = {"teacher": {"tensor_parallel_size": 8}, "training": {"gpus": 8}}
        student = worker_resources(config, "student")
        teacher = worker_resources(config, "teacher")
        self.assertEqual(student["gpus"] + teacher["gpus"], 16)
        self.assertEqual(student["cpus"], 64)
        self.assertEqual(student["memory_mib"], 819200)
        self.assertEqual(teacher["memory_mib"], 524288)
        config["training"]["gpus"] = 16
        with self.assertRaisesRegex(ValueError, "one node"):
            worker_resources(config, "student")

    def test_master_no_proxy_is_ordered_and_deduplicated(self):
        env = {"NO_PROXY": "proxy", "no_proxy": "proxy"}
        with patch.dict("os.environ", {"MASTER_ADDR": "10.0.0.1"}):
            add_master_no_proxy(env)
        self.assertEqual(env["NO_PROXY"], "proxy,localhost,127.0.0.1,10.0.0.1")
        self.assertEqual(env["no_proxy"], "proxy,localhost,127.0.0.1,10.0.0.1")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.config = {"teacher": {"model": str(self.model)}}
        (self.root / "config.json").write_text(json.dumps(self.config))
        (self.model / "config.json").write_text(json.dumps({
            "architectures": ["Qwen4ExpForConditionalGeneration"]}))

    def test_model_implementation_is_required_instead_of_version_threshold(self):
        registry = Mock()
        registry.is_arch_supported.return_value = False
        with self.assertRaisesRegex(RuntimeError, "no native implementation"):
            teacher_model_support(self.config, registry)
        registry.is_arch_supported.return_value = True
        registry.resolve_model_cls.return_value = (type("NativeModel", (), {}),
                                                   "Qwen4ExpForConditionalGeneration")
        with patch("importlib.metadata.version", return_value="0.5.21"):
            support = teacher_model_support(self.config, registry)
        self.assertEqual(support["sglang_version"], "0.5.21")
        registry.resolve_model_cls.return_value = (object, "TransformersForCausalLM")
        with self.assertRaisesRegex(RuntimeError, "unexpected fallback"):
            teacher_model_support(self.config, registry)

    def test_student_requires_verified_live_teacher(self):
        with self.assertRaisesRegex(ValueError, "not ready"):
            check_teacher_ready(self.root)
        ready = {"model": str(self.model), "scoring_verified": True,
                 "url": "http://127.0.0.1:31051"}
        (self.root / "teacher_ready.json").write_text(json.dumps(ready))
        opener = Mock()
        opener.open.return_value = io.BytesIO(json.dumps({"model_path": str(self.model)}).encode())
        with patch("urllib.request.build_opener", return_value=opener):
            check_teacher_ready(self.root)
        (self.root / "teacher_exit.json").write_text('{"exit_code": 0}')
        with self.assertRaisesRegex(ValueError, "already exited"):
            check_teacher_ready(self.root)

    def test_cv_student_requires_current_real_native_scoring_audit(self):
        ready = {"model": str(self.model), "scoring_verified": True,
                 "url": "http://127.0.0.1:31051"}
        (self.root / "teacher_ready.json").write_text(json.dumps(ready))
        paths = {key: self.root / key for key in ["evaluator_config", "references_path", "dataset"]}
        for p in paths.values():
            p.write_text("fixture")
        self.config.update(continuation={"enabled": True,
            "evaluator_config": str(paths["evaluator_config"]),
            "references_path": str(paths["references_path"])}, dataset=str(paths["dataset"]))
        (self.root / "config.json").write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "judge preflight is not ready"):
            check_teacher_ready(self.root)
        audit = {"passed": True, "real_native_scoring": True}
        for name, key in [("evaluator_sha256", "evaluator_config"),
                          ("references_sha256", "references_path"), ("dataset_sha256", "dataset")]:
            audit[name] = hashlib.sha256(paths[key].read_bytes()).hexdigest()
        (self.root / "cv_judge_preflight.json").write_text(json.dumps(audit))
        opener = Mock()
        opener.open.return_value = io.BytesIO(json.dumps({"model_path": str(self.model)}).encode())
        with patch("urllib.request.build_opener", return_value=opener):
            check_teacher_ready(self.root)
        paths["evaluator_config"].write_text("changed endpoint")
        with self.assertRaisesRegex(ValueError, "stale"):
            check_teacher_ready(self.root)


if __name__ == "__main__":
    unittest.main()
