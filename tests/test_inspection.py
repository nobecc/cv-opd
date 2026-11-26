import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


watch = load("watch_trial")
inspect = load("inspect_trial")


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.save("submission.json", {"jobs": {"student": {"name": "student"}}})
        self.save("config.json", {"output": str(self.directory / "training"), "training": {"updates": 2}})
        job = SimpleNamespace(metadata=SimpleNamespace(name="student"),
                              status=SimpleNamespace(current=SimpleNamespace(name="Running")))
        self.client = patch.object(watch, "RJobClient")
        self.client.start().return_value.list.return_value = [job]
        self.addCleanup(self.client.stop)

    def save(self, name, value):
        (self.directory / name).write_text(json.dumps(value))

    def result(self):
        self.save("student_result.json", {"process_exit_code": 0, "latest_checkpoint_iteration": 1})
        checkpoint = self.directory / "training/checkpoints"
        checkpoint.mkdir(parents=True)
        (checkpoint / "latest_checkpointed_iteration.txt").write_text("1")
        (checkpoint / "iter_0000001").mkdir()
        return checkpoint

    def test_checkpoint_marker_alone_is_not_completion(self):
        checkpoint = self.result()
        self.assertEqual(watch.update(self.directory)["phase"], "student_exited_incomplete_or_unverified")
        (checkpoint / "iter_0000001/weights.distcp").write_bytes(b"fixture tensor artifact")
        status = watch.update(self.directory)
        self.assertEqual(status["phase"], "training_process_completed_checkpoint_saved")
        self.assertFalse(status["parameter_change_independently_verified"])

    def test_common_metadata_is_not_a_tensor_checkpoint(self):
        checkpoint = self.result()
        (checkpoint / "iter_0000001/common.pt").write_bytes(b"metadata only")
        status = watch.update(self.directory)
        self.assertEqual(status["evidence"]["checkpoint_tensor_files"], 0)
        self.assertEqual(status["phase"], "student_exited_incomplete_or_unverified")

    def test_zero_is_first_completed_rollout_not_missing_checkpoint(self):
        checkpoint = self.result()
        self.save("student_result.json", {"process_exit_code": 0, "latest_checkpoint_iteration": 0})
        (checkpoint / "latest_checkpointed_iteration.txt").write_text("0")
        (checkpoint / "iter_0000000").mkdir()
        (checkpoint / "iter_0000000/weights.distcp").write_bytes(b"fixture")
        status = watch.update(self.directory)
        self.assertEqual(status["evidence"]["checkpointed_rollouts"], 1)
        self.assertEqual(status["phase"], "student_exited_incomplete_or_unverified")
        self.save("config.json", {"output": str(self.directory / "training"), "training": {"updates": 1}})
        self.assertEqual(watch.update(self.directory)["phase"], "training_process_completed_checkpoint_saved")
        (checkpoint / "latest_checkpointed_iteration.txt").unlink()
        self.assertEqual(watch.update(self.directory)["evidence"]["checkpointed_rollouts"], 0)

    def test_parameter_verification_rejects_stale_file_metadata(self):
        checkpoint = self.result()
        files = [checkpoint / "iter_0000001/weights.pt", checkpoint / "previous.pt"]
        for file in files:
            file.write_bytes(b"fixture")
        report = {"passed": True, "method": "sampled_checkpoint_model_parameters",
                  "run_dir": str(self.directory.resolve()), "after_iteration": 1,
                  "changed_sampled_elements": 1, "all_sampled_values_finite": True,
                  "files": [{"path": str(f), "size": f.stat().st_size, "mtime_ns": f.stat().st_mtime_ns} for f in files]}
        self.save("PARAMETER_VERIFICATION.json", report)
        self.assertTrue(watch.update(self.directory)["parameter_change_independently_verified"])
        files[0].write_bytes(b"changed fixture")
        self.assertFalse(watch.update(self.directory)["parameter_change_independently_verified"])

    def test_all_tensor_parallel_shards_are_required_for_legacy_format(self):
        checkpoint = self.result()
        self.save("config.json", {"output": str(self.directory / "training"),
                  "training": {"updates": 2, "checkpoint_format": "torch", "tensor_parallel_size": 4}})
        for rank in range(4):
            d = checkpoint / f"iter_0000001/mp_rank_{rank:02d}"
            d.mkdir()
            (d / "model_optim_rng.pt").write_bytes(b"fixture tensor")
            status = watch.update(self.directory)
            self.assertEqual(status["evidence"]["checkpoint_shards_complete"], rank == 3)
        self.assertEqual(status["phase"], "training_process_completed_checkpoint_saved")

    def test_failure_overrides_success_file(self):
        checkpoint = self.result()
        (checkpoint / "iter_0000001/weights.pt").write_bytes(b"fixture")
        self.save("student_error.json", {"error": "optimizer failed"})
        self.assertEqual(watch.update(self.directory)["phase"], "failed_or_stopped")

    def test_nonzero_exit_without_error_file_is_failure(self):
        self.save("student_exit.json", {"exit_code": 1})
        self.assertEqual(watch.update(self.directory)["phase"], "failed_or_stopped")

    def test_teacher_probe_requires_finite_aligned_scores(self):
        self.save("teacher_probe.json", {"real_model": True, "output": [1], "teacher_log_probs": [float('nan')]})
        self.assertFalse(watch.update(self.directory)["teacher_scoring_verified"])
        self.save("teacher_probe.json", {"real_model": True, "output": [1], "teacher_log_probs": [-2.0]})
        self.assertTrue(watch.update(self.directory)["teacher_scoring_verified"])

    def test_ray_host_oom_is_distinguished_from_cuda_oom(self):
        message = "ray.exceptions.OutOfMemoryError: 1 worker(s) were killed due to the node running low on memory."
        diagnosis = inspect.diagnose({"phase": "failed_or_stopped"}, message)
        self.assertEqual(diagnosis["cause"], "ray_host_out_of_memory")
        self.assertIsNone(inspect.diagnose({"phase": "teacher_loading"}, message))

    def test_checkpoint_host_oom_is_identified_separately(self):
        message = "workers were killed due to the node running low on memory; ray::MegatronTrainRayActor.save_model"
        diagnosis = inspect.diagnose({"phase": "failed_or_stopped"}, message)
        self.assertEqual(diagnosis["cause"], "checkpoint_host_out_of_memory")

    def test_warning_text_cannot_create_failure(self):
        message = "GDN does not support packed sequence for now."
        self.assertIsNone(inspect.diagnose({"phase": "teacher_loading"}, message))
        self.assertEqual(inspect.diagnose({"phase": "failed_or_stopped"}, message)["cause"],
                         "gdn_packed_sequence_unsupported")

    def test_engine_context_limit_is_diagnosed(self):
        message = "Input length (32766 tokens) exceeds the maximum allowed length (32762 tokens)."
        self.assertEqual(inspect.diagnose({"phase": "failed_or_stopped"}, message)["cause"],
                         "engine_context_limit")
        self.assertIsNone(inspect.diagnose({"phase": "teacher_loading"}, message))

    def test_failed_query_labels_previous_status_stale(self):
        self.save("STATUS.json", {"phase": "queued_waiting_for_resources", "time": "old"})
        with patch.object(inspect, "RUNS", self.directory), \
             patch.object(inspect.subprocess, "run", side_effect=OSError("network down")):
            record = inspect.inspect(self.directory)
        self.assertFalse(record["query_ok"])
        self.assertTrue(record["status_is_stale"])
        self.assertEqual(record["last_known_status"]["time"], "old")

    def test_failure_archive_is_not_duplicated_on_each_poll(self):
        status = {"phase": "failed_or_stopped", "jobs": {"student": {"name": "student", "state": "Failed"}}}
        self.save("STATUS.json", status)
        (self.directory / "student.log").write_text("NotImplementedError: GDN does not support packed sequence for now.")
        with patch.object(inspect, "RUNS", self.directory), \
             patch.object(inspect.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
             patch.object(inspect, "snapshot_log") as archive:
            one = inspect.inspect(self.directory)
            two = inspect.inspect(self.directory)
        self.assertEqual(one["incident_dir"], two["incident_dir"])
        archive.assert_called_once()
        self.assertTrue(two["attention_required"])
