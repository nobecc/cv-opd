import copy
import unittest

from finance_opd.screening import screening_readiness


class ScreeningReadinessTests(unittest.TestCase):
    def outcome(self, **updates):
        status = dict(native_status="ok", exit_code=0, generated_turns=3,
                      native_tool_successes=2, native_finished=True)
        status.update(updates)
        return {"id": "fixture", "native_status": status}

    def test_model_no_finish_does_not_become_runtime_failure_or_success(self):
        item = self.outcome(native_status="no_finish", exit_code=1, native_finished=False)
        original = copy.deepcopy(item)
        result = screening_readiness([item])
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["model_task_failures"]), 1)
        self.assertEqual(item, original)
        self.assertFalse(item["native_status"]["native_finished"])

    def test_proxy_failure_stops_expansion_even_after_successful_tools(self):
        self.assertFalse(screening_readiness([self.outcome(proxy_error="token/logprob mismatch")])["passed"])

    def test_runtime_exception_stops_expansion(self):
        self.assertFalse(screening_readiness([{"id": "fixture", "error": "worker failed"}])["passed"])

    def test_a_real_generation_and_tool_cycle_are_required(self):
        self.assertFalse(screening_readiness([self.outcome(generated_turns=0)])["passed"])
        self.assertFalse(screening_readiness([self.outcome(native_tool_successes=0)])["passed"])

    def test_budget_exhaustion_is_a_task_outcome(self):
        for update in [dict(timed_out=True), dict(context_truncated=True)]:
            self.assertTrue(screening_readiness([self.outcome(**update)])["passed"])

    def test_unknown_native_failure_is_not_silently_ignored(self):
        self.assertFalse(screening_readiness([self.outcome(native_status="error", exit_code=2)])["passed"])


if __name__ == "__main__":
    unittest.main()
