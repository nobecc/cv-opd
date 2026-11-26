"""Continuation-value contracts with deterministic HTTP models, never a GPU."""
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import ClientSession, web

from finance_opd.value import (LLM_VERIFIER_HOOK, evaluate_continuation_value, extract_tool_evidence,
                              native_report_score, parse_verification, validate_value_config, visible_output,
                              validate_dataset_references)


def observation(call_id, value, *, error=False):
    return {"role": "user", "content": json.dumps({"tool_result": {
        "id": call_id, "name": "query", "is_error": error, "content": json.dumps(value)}})}


class Builder:
    def __init__(self):
        self.prompts, self.cleared = [], []

    def prepare_generate_input(self, session, messages):
        self.prompts.append(copy.deepcopy(messages))
        return SimpleNamespace(input_ids=[len(self.prompts), 123])

    def clear_session(self, session):
        self.cleared.append(session)


class ValueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.builder = Builder()
        self.requests = []
        self.finish = "stop"
        self.verifier_text = json.dumps({"evidence_valid": True, "safety_valid": True,
            "guidance": "Check the acquisition disclosure and comparable reporting scope.", "reason": "Fixture audit"})
        self.module = types.ModuleType("fixture_value_hooks")
        self.score_requests, self.verifier_requests = [], []

        def score(request):
            self.score_requests.append(copy.deepcopy(request))
            return int(request["report"].rsplit("=", 1)[1]) * 0.2

        def verify(request):
            self.verifier_requests.append(copy.deepcopy(request))
            return json.loads(self.verifier_text)

        self.module.score, self.module.verify = score, verify
        self.patcher = patch.dict(sys.modules, {self.module.__name__: self.module})
        self.patcher.start()
        app = web.Application()
        app.router.add_post("/generate", self.model)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self.config = {"teacher": {"url": self.url, "context_length": 4096},
            "runtime": {"context_length": 4096}, "continuation": {
                "score_hook": "fixture_value_hooks.score", "verifier_hook": "fixture_value_hooks.verify",
                "writer_max_new_tokens": 32, "verifier_max_new_tokens": 64, "value_timeout_seconds": 5}}
        self.task = {"id": "fixture:1", "benchmark": "ddr_10k", "native_case_id": "1",
                     "task": "Explain revenue growth.", "evaluation": {"gold": "PRIVATE_GOLD_SENTINEL"}}
        self.before = [{"role": "user", "content": "task"}, observation("random-a", {"revenue_growth": "20%"}),
                       {"role": "assistant", "content": "An unsupported assistant-only assertion."}]
        self.continuation = {"new_observations": [observation("random-b", {"growth_source": "acquisition"})],
            "actions": ['<tool_call>{"name":"query","arguments":{}}</tool_call>'],
            "report": "FINISH: The new observation changes the explanation."}

    async def asyncTearDown(self):
        await self.runner.cleanup()
        self.patcher.stop()
        self.directory.cleanup()

    async def model(self, request):
        body = await request.json()
        self.requests.append(body)
        messages = self.builder.prompts[body["input_ids"][0] - 1]
        text = ("FINISH: evidence_count=" + str(len(json.loads(messages[1]["content"])["evidence"]))
                if "Write a financial" in messages[0]["content"] else self.verifier_text)
        return web.json_response({"text": text, "meta_info": {"finish_reason": {"type": self.finish}}})

    async def evaluate(self):
        async with ClientSession() as client:
            return await evaluate_continuation_value(client=client, builder=self.builder, config=self.config,
                task=self.task, before_messages=self.before, continuation=self.continuation, output=self.root / "value")

    async def test_same_fixed_writer_before_after_and_no_private_or_assistant_evidence(self):
        result = await self.evaluate()
        self.assertAlmostEqual(result["gain"], 0.2)
        self.assertEqual((result["before_score"], result["after_score"]), (0.2, 0.4))
        self.assertEqual(self.requests[0]["sampling_params"], self.requests[1]["sampling_params"])
        self.assertEqual(self.requests[0]["sampling_params"]["sampling_seed"], 42)
        self.assertEqual(self.builder.prompts[0][0], self.builder.prompts[1][0])
        before = json.loads(self.builder.prompts[0][1]["content"])
        after = json.loads(self.builder.prompts[1][1]["content"])
        self.assertEqual(len(before["evidence"]), 1)
        self.assertEqual(len(after["evidence"]), 2)
        public = json.dumps(self.builder.prompts + self.verifier_requests + self.score_requests)
        self.assertNotIn("PRIVATE_GOLD_SENTINEL", public)
        self.assertNotIn("unsupported assistant-only", json.dumps(self.builder.prompts))
        self.assertEqual(set(self.score_requests[0]), {"task", "report"})
        self.assertEqual(len(self.builder.cleared), 2)
        self.assertTrue(result["audit"]["references_visible_to_models"] is False)

    async def test_repeated_observations_do_not_manufacture_gain_or_use_teacher_claims(self):
        self.continuation["new_observations"] = [observation("different-call-id", {"revenue_growth": "20%"}),
                                                  observation("failed", {"future_fact": 999}, error=True)]
        self.continuation["report"] = "FINISH: Unsupported teacher claims do not become evidence."
        result = await self.evaluate()
        self.assertEqual(result["gain"], 0)
        self.assertEqual(result["audit"]["before_evidence_sha256"], result["audit"]["after_evidence_sha256"])
        self.assertNotIn("Unsupported teacher claims", json.dumps(self.builder.prompts))

    async def test_failed_or_malformed_code_output_is_not_successful_evidence(self):
        def code(value):
            return {"role": "user", "content": json.dumps({"tool_result": {
                "id": "code", "name": "execute_code", "is_error": False, "content": value}})}
        for value in [json.dumps({"returncode": 1, "stdout": "partial result 999"}), "malformed",
                      json.dumps({"stdout": "missing status"}), json.dumps({"returncode": False})]:
            self.assertEqual(extract_tool_evidence([code(value)]), [])
        evidence = extract_tool_evidence([code(json.dumps({"returncode": 0, "stdout": "computed 120"}))])
        self.assertEqual(len(evidence), 1)
        self.assertIn("computed 120", evidence[0]["content"])

    async def test_verification_is_explicit_and_rejected_or_invalid_is_not_auto_passed(self):
        self.config["continuation"].pop("verifier_hook")
        with self.assertRaisesRegex(ValueError, "explicitly"):
            await self.evaluate()
        self.assertEqual(self.requests, [])
        self.config["continuation"]["verifier_hook"] = LLM_VERIFIER_HOOK
        self.verifier_text = json.dumps({"evidence_valid": False, "safety_valid": False,
            "guidance": "Recheck the source before continuing.", "reason": "unsupported"})
        result = await self.evaluate()
        self.assertFalse(result["evidence_valid"])
        self.assertFalse(result["safety_valid"])
        self.assertEqual(len(self.requests), 3)
        self.assertNotIn("PRIVATE_GOLD_SENTINEL", json.dumps(self.builder.prompts))
        self.verifier_text = '{"evidence_valid":"true","safety_valid":true,"guidance":"g"}'
        with self.assertRaisesRegex(ValueError, "boolean"):
            await self.evaluate()

    async def test_nonfinite_scores_and_incomplete_or_tool_call_writer_fail_closed(self):
        for value in [True, float("nan"), float("inf")]:
            self.module.score = lambda request, result=value: result
            with self.subTest(score=value), self.assertRaisesRegex(ValueError, "finite"):
                await self.evaluate()
        self.finish = "length"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            await self.evaluate()

    async def test_writer_context_overflow_does_not_silently_shrink_budget(self):
        self.config["teacher"]["context_length"] = 20
        with self.assertRaisesRegex(ValueError, "unequal writer budgets"):
            await self.evaluate()
        self.assertEqual(self.requests, [])

    async def test_native_references_checked_before_model_requests(self):
        self.config["continuation"].pop("score_hook")
        self.config["continuation"].update(references_path=str(self.root / "missing.jsonl"),
            evaluator_config=str(Path(__file__).resolve().parents[2] / "eval_finance/config.yaml"))
        with self.assertRaisesRegex(ValueError, "reference file"):
            await self.evaluate()
        self.assertEqual(self.requests, [])
        refs = self.root / "references.jsonl"
        refs.write_text(json.dumps({"id": "another-case", "benchmark": "ddr_10k", "evaluation": {"row": {"gold": "private"}}}) + "\n")
        self.config["continuation"]["references_path"] = str(refs)
        with self.assertRaisesRegex(ValueError, "no private"):
            await self.evaluate()
        self.assertEqual(self.requests, [])


class ValueContractTests(unittest.TestCase):
    def test_strict_verifier_json_and_boolean_types(self):
        for bad in ['```json\n{}\n```', '[]', '{}',
                    '{"evidence_valid":1,"safety_valid":true,"guidance":"g"}',
                    '{"evidence_valid":true,"evidence_valid":false,"safety_valid":true,"guidance":"g"}',
                    '{"evidence_valid":true,"safety_valid":true,"guidance":"g","audit":{"n":NaN}}',
                    '{"evidence_valid":true,"safety_valid":true,"guidance":""}']:
            with self.subTest(value=bad), self.assertRaises(ValueError):
                parse_verification(bad)

    def test_only_explicit_complete_thinking_framing_is_removed(self):
        self.assertEqual(visible_output("<think>reasoning</think> FINISH: answer"), "FINISH: answer")
        self.assertEqual(visible_output('prefilled reasoning</think> {"evidence_valid":true}'), '{"evidence_valid":true}')
        with self.assertRaisesRegex(ValueError, "incomplete"):
            visible_output("<think>unfinished")

    def test_actual_tool_arguments_retained_but_random_call_ids_are_not_evidence(self):
        call = {"role": "assistant", "content": '<tool_call>{"name":"query","arguments":{"sql":"select revenue"}}</tool_call>'}
        evidence = extract_tool_evidence([call, observation("random-id", {"value": 1})])
        self.assertEqual(evidence[0]["arguments"], {"sql": "select revenue"})
        self.assertNotIn("random-id", json.dumps(evidence))

    def test_native_scorer_uses_only_answer_quality_and_preserves_reference_boundary(self):
        # Real eval_finance config/schema, mocked evaluator computation: no judge request.
        workspace = Path(__file__).resolve().parents[2]
        if str(workspace) not in sys.path:
            sys.path.insert(0, str(workspace))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = root / "private.jsonl"
            references.write_text(json.dumps({"id": "fixture:1", "benchmark": "ddr_10k", "native_case_id": "1",
                "evaluation": {"row": {"reward_rubric": {"secret": "PRIVATE_GOLD_SENTINEL"}}}}) + "\n")
            request = {"task": {"id": "fixture:1", "benchmark": "ddr_10k", "native_case_id": "1", "task": "Q"},
                       "report": "FINISH: Report", "references_path": str(references),
                       "evaluator_config": str(workspace / "eval_finance/config.yaml"), "output": str(root)}
            def evaluate(calculator, case):
                import yaml
                config = yaml.safe_load(Path(calculator.config_path).read_text())
                self.assertEqual(config["reward"]["weights"]["answer_quality"], 1)
                self.assertTrue(all(value == 0 for name, value in config["reward"]["weights"].items() if name != "answer_quality"))
                self.assertEqual(config["reward"]["max_report_chars"], 0)
                self.assertEqual(case["row"]["reward_rubric"]["secret"], "PRIVATE_GOLD_SENTINEL")
                self.assertEqual(case["final_report"], request["report"])
                return {"status": "succeeded", "raw_score": 9, "components": {
                    "answer_quality": {"status": "scored", "score": -0.25}}, "judge": {"failed": 0}}
            with patch("eval_finance.reward_calculator.FreeTextRewardCalculator.evaluate_case", new=evaluate):
                self.assertEqual(native_report_score(request), -0.25)
            self.assertEqual(list(root.glob("answer-only-*")), [])

    def test_dataset_references_preflight_checks_coverage_and_matching_ids_without_model_calls(self):
        workspace = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, references = root / "train.jsonl", root / "refs.jsonl"
            task = {"id": "fixture:1", "benchmark": "ddr_10k", "native_case_id": "1"}
            dataset.write_text(json.dumps({"task": "Q", "metadata": task}) + "\n")
            reference = {**task, "evaluation": {"row": {"reference": "private"}}}
            references.write_text(json.dumps(reference) + "\n")
            config = {"dataset": str(dataset), "continuation": {
                "verifier_hook": LLM_VERIFIER_HOOK, "references_path": str(references),
                "evaluator_config": str(workspace / "eval_finance/config.yaml")}}
            validate_dataset_references(config)
            dataset.write_text(json.dumps({"task": "Q", "metadata": {**task, "id": "unreferenced-company"}}) + "\n")
            with self.assertRaisesRegex(ValueError, "unreferenced-company"):
                validate_dataset_references(config)
            dataset.write_text(json.dumps({"task": "Q", "metadata": {**task, "native_case_id": "wrong"}}) + "\n")
            with self.assertRaisesRegex(ValueError, "native_case_id"):
                validate_dataset_references(config)
            # An explicit custom scorer owns its references and need not open native files.
            config["continuation"]["score_hook"] = "builtins.len"
            config["dataset"] = str(root / "missing")
            validate_dataset_references(config)
