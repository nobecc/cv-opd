import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from finance_opd.training_filter import require_cv_signal, supervised_tokens


def sample(mask, weight=1.0):
    return SimpleNamespace(loss_mask=mask, response_length=len(mask), reward={"opd_weight": weight})


class TrainingFilterTests(unittest.TestCase):
    def test_masks_and_weights_both_required_without_mutating_samples(self):
        group = [[sample([0, 0]), sample([1], 0)], [sample([1, 0, 1], .2)]]
        original = copy.deepcopy(group)
        self.assertEqual(supervised_tokens(group), 2)
        self.assertEqual(group, original)

    def test_invalid_mask_and_nonfinite_weight_fail_closed(self):
        for value in [sample([2]), sample([True]), sample([1], float("nan"))]:
            with self.assertRaises(ValueError):
                supervised_tokens([[value]])

    def test_rejections_retry_but_are_bounded_and_positive_signal_resets(self):
        with tempfile.TemporaryDirectory() as temp:
            config = {"continuation": {"enabled": True}, "training": {"max_zero_signal_rollouts": 2}, "output": temp}
            with patch("finance_opd.training_filter.read_config", return_value=config):
                self.assertFalse(require_cv_signal(None, [[sample([0])]]).keep)
                self.assertTrue(require_cv_signal(None, [[sample([1], .1)]]).keep)
                self.assertFalse(require_cv_signal(None, [[sample([0])]]).keep)
                with self.assertRaisesRegex(RuntimeError, "optimizer not advanced"):
                    require_cv_signal(None, [[sample([0])]])
            audits = list(map(json.loads, (Path(temp) / "cv_signal_filter.jsonl").read_text().splitlines()))
            self.assertEqual([x["consecutive_zero_signal_rollouts"] for x in audits], [1, 0, 1, 2])


if __name__ == "__main__":
    unittest.main()
