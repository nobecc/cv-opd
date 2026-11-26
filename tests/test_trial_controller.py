import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


class TrialControllerTests(unittest.TestCase):
    def test_submits_student_once_after_scoring_and_stops_on_completion(self):
        spec = importlib.util.spec_from_file_location("trial_controller", ROOT / "scripts/run_submitted_trial.py")
        controller = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"watch_trial": Mock()}):
            spec.loader.exec_module(controller)
        teacher = {"teacher": {"name": "teacher", "state": "Running"}}
        both = dict(teacher, student={"name": "student", "state": "Running"})
        states = [
            {"phase": "teacher_loading", "jobs": teacher},
            {"phase": "teacher_scoring_verified_waiting_for_student", "jobs": teacher},
            {"phase": "teacher_scoring_verified_waiting_for_student", "jobs": both},
            {"phase": "training_process_completed_checkpoint_saved", "jobs": both},
        ]
        with tempfile.TemporaryDirectory(prefix="controller-test-", dir=ROOT / "runs") as temporary:
            directory = Path(temporary)
            (directory / "submission.json").write_text(json.dumps({"jobs": teacher}))
            (directory / "config.json").write_text('{"continuation": {"enabled": false}}')
            with patch.object(controller, "update", side_effect=states), \
                 patch.object(controller, "save_logs") as save, \
                 patch.object(controller.subprocess, "run") as submit, \
                 patch.object(controller.time, "sleep"), \
                 patch.object(sys, "argv", ["controller", temporary]), \
                 patch.object(sys, "stdout", io.StringIO()):
                controller.main()
            submit.assert_called_once()
            self.assertEqual(submit.call_args.args[0][-2:], ["--role", "student"])
            save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
