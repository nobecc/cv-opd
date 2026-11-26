import asyncio
import copy
import gzip
import json
import tempfile
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import ClientSession, web

from finance_opd.episode import EpisodeProxy, output_budget, native_settings
from finance_opd.config import ROOT
from finance_opd.rollout import samples_from_records


class FixtureBuilder:
    """HTTP lifecycle fixture, not a replacement for the real training mask builder."""
    def prepare_generate_input(self, session, messages):
        return SimpleNamespace(input_ids=[1, 2, 3])
    def record_generation(self, *args):
        pass
    def clear_session(self, session):
        pass


class EpisodeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runners = []
        self.teacher_requests = []
        self.version = "fixture-1"
        self.misalign = False
        self.teacher_status = 200
        self.teacher_delay = 0
        self.student_requests = []
        self.fill_budget = False
        self.teacher_input_limit = None
        self.student_url = await self.serve(self.student)
        self.teacher_url = await self.serve(self.teacher)

    async def asyncTearDown(self):
        for runner in reversed(self.runners):
            await runner.cleanup()

    async def serve(self, handler):
        app = web.Application(); app.router.add_post("/generate", handler)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
        self.runners.append(runner)
        return f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    async def student(self, request):
        body = await request.json()
        self.student_requests.append(body)
        self.assertEqual(body["sampling_params"]["temperature"], 1)
        ids = [8, 9] if not self.fill_budget else [
            8 + i % 2 for i in range(body["sampling_params"]["max_new_tokens"])]
        return web.json_response({"text": "FINISH: fixture", "meta_info": {
            "output_token_logprobs": [[-0.2, token] for token in ids], "weight_version": self.version,
            "finish_reason": {"type": "stop"}}})

    async def teacher(self, request):
        body = await request.json(); self.teacher_requests.append(body)
        if self.teacher_input_limit is not None and len(body["input_ids"]) > self.teacher_input_limit:
            return web.json_response({"error": "Input length exceeds the maximum allowed length"}, status=400)
        if self.teacher_delay:
            await asyncio.sleep(self.teacher_delay)
        if self.teacher_status != 200:
            return web.json_response({"error": "fixture teacher unavailable"}, status=self.teacher_status)
        ids = list(body["input_ids"])
        if self.misalign:
            ids[-1] += 1
        return web.json_response({"meta_info": {"input_token_logprobs": [[None, ids[0]]] + [[-1.5, t] for t in ids[1:]]}})

    def config(self):
        return {"teacher": {"url": self.teacher_url}, "runtime": {"tool_protocol": "legacy_json", "context_length": 32}, "context_hook": None}

    def test_teacher_prefix_can_be_the_tighter_generation_budget(self):
        config = self.config()
        config["runtime"].update(context_length=100, context_safety_margin=8)
        config["teacher"]["context_length"] = 90
        self.assertEqual(output_budget(config, 20, 80, 50), 2)

    def test_native_compaction_window_is_separate_from_engine_context(self):
        config = self.config()
        config["safactory"] = str(ROOT.parent / "SAfactory")
        config["runtime"].update(context_length=32768, native_context_window=16384)
        settings = native_settings(config, "ddr_10k", base_url="http://localhost:1", run_dir=ROOT)
        self.assertEqual(settings["model"]["context_window"], 16384)
        self.assertEqual(config["runtime"]["context_length"], 32768)

    async def test_cross_context_keeps_student_prefix_and_freezes_weight_before_sampling(self):
        module = types.ModuleType("fixture_context")
        holder = {}
        def hook(request):
            self.assertEqual(len(self.student_requests), 0)
            self.assertNotIn("evaluation", request["task"])
            request["messages"][0]["content"] = "teacher verified context"
            holder["view"] = {"messages": request["messages"], "weight": 0.25, "audit": {"evidence": "fixture"}}
            return holder["view"]
        module.hook = hook
        class Builder(FixtureBuilder):
            def prepare_generate_input(self, session, messages):
                teacher = messages[0]["content"] == "teacher verified context"
                return SimpleNamespace(input_ids=[11 if teacher else 1, 2, 3])
            def record_generation(self, *args):
                # Simulate external mutation after student sampling.
                holder["view"]["weight"] = 9
                holder["view"]["messages"][0]["content"] = "future leaked text"
        config = self.config(); config["context_hook"] = "fixture_context.hook"
        messages = [{"role": "user", "content": "original task"}]
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {"fixture_context": module}):
            async with EpisodeProxy(builder=Builder(), task={"task": "task"}, student_url=self.student_url,
                                    config=config, output=Path(root), session_id="branch", sampling_params={}) as proxy:
                await proxy.complete({"model": "student", "messages": messages})
                self.assertEqual(self.student_requests[0]["input_ids"], [1, 2, 3])
                self.assertEqual(self.teacher_requests[0]["input_ids"], [11, 2, 3, 8, 9])
                self.assertEqual(proxy.records[0]["opd_weight"], 0.25)
                self.assertEqual(messages[0]["content"], "original task")

    async def test_teacher_outage_and_timeout_produce_no_trainable_records(self):
        for status, delay in [(503, 0), (200, 0.1)]:
            self.teacher_status, self.teacher_delay = status, delay
            config = self.config(); config["runtime"]["request_timeout_seconds"] = 0.03
            with self.subTest(status=status), tempfile.TemporaryDirectory() as root:
                async with EpisodeProxy(builder=FixtureBuilder(), task={}, student_url=self.student_url,
                                        config=config, output=Path(root), session_id="failure", sampling_params={}) as proxy:
                    async with ClientSession() as client:
                        async with client.post(proxy.base_url + "/chat/completions", json={"model": "student",
                                "messages": [{"role": "user", "content": "task"}]}) as response:
                            self.assertEqual(response.status, 502)
                    self.assertTrue(proxy.failure)
                    self.assertEqual(proxy.records, [])
                    self.assertTrue((Path(root) / "proxy_error.json").is_file())

    async def test_context_overflow_fails_before_requesting_that_engine(self):
        for side in ["student", "teacher"]:
            config = self.config()
            config["runtime" if side == "student" else "teacher"]["context_length"] = 2
            with self.subTest(side=side), tempfile.TemporaryDirectory() as root:
                self.student_requests.clear(); self.teacher_requests.clear()
                async with EpisodeProxy(builder=FixtureBuilder(), task={}, student_url=self.student_url,
                                        config=config, output=Path(root), session_id="overflow", sampling_params={}) as proxy:
                    with self.assertRaisesRegex(ValueError, "context"):
                        await proxy.complete({"model": "student", "messages": [{"role": "user", "content": "task"}]})
                    self.assertEqual(len(self.student_requests), 0)
                    self.assertEqual(self.teacher_requests, [])
                    self.assertEqual(proxy.records, [])

    async def test_real_32k_teacher_boundary_is_reserved_before_sampling(self):
        config = self.config()
        config["runtime"].update(context_length=32768, context_safety_margin=64)
        config["teacher"]["context_length"] = 32768
        self.teacher_input_limit = 32762
        self.fill_budget = True
        class Builder(FixtureBuilder):
            def prepare_generate_input(self, session, messages):
                return SimpleNamespace(input_ids=[1] * 32700)
        with tempfile.TemporaryDirectory() as root:
            async with EpisodeProxy(builder=Builder(), task={}, student_url=self.student_url,
                                    config=config, output=Path(root), session_id="boundary", sampling_params={}) as proxy:
                await proxy.complete({"model": "student", "messages": [{"role": "user", "content": "task"}],
                                      "max_tokens": 4096})
                self.assertEqual(self.student_requests[0]["sampling_params"]["max_new_tokens"], 4)
                self.assertEqual(len(self.teacher_requests[0]["input_ids"]), 32704)
                self.assertEqual(proxy.records[0]["response_ids"], [8, 9, 8, 9])
                self.assertEqual(self.teacher_requests[0]["input_ids"][-4:], [8, 9, 8, 9])

    async def test_context_exhaustion_preserves_scored_turns_without_proxy_failure(self):
        class Builder(FixtureBuilder):
            length = 3
            def prepare_generate_input(self, session, messages):
                return SimpleNamespace(input_ids=[1] * self.length)
        builder = Builder()
        with tempfile.TemporaryDirectory() as root:
            async with EpisodeProxy(builder=builder, task={}, student_url=self.student_url,
                                    config=self.config(), output=Path(root), session_id="limit", sampling_params={}) as proxy:
                body = {"model": "student", "messages": [{"role": "user", "content": "task"}]}
                await proxy.complete(body)
                builder.length = 32
                async with ClientSession() as client:
                    async with client.post(proxy.base_url + "/chat/completions", json=body) as response:
                        self.assertEqual(response.status, 400)
                self.assertIsNotNone(proxy.context_exhausted)
                self.assertIsNone(proxy.failure)
                self.assertEqual(len(proxy.records), 1)
                self.assertEqual(len(self.student_requests), 1)
                self.assertEqual(len(self.teacher_requests), 1)
                self.assertTrue((Path(root) / "context_budget.json").is_file())

    async def test_proxy_to_teacher_and_sample_grouping(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            async with EpisodeProxy(builder=FixtureBuilder(), task={"task": "question"}, student_url=self.student_url,
                                    config=self.config(), output=directory, session_id="test", sampling_params={}) as proxy:
                async with ClientSession() as client:
                    async with client.post(proxy.base_url + "/chat/completions", json={
                        "model": "student", "messages": [{"role": "user", "content": "question"}], "max_tokens": 4}) as response:
                        self.assertEqual(response.status, 200)
                        value = await response.json()
                self.assertEqual(value["metadata"]["weight_version"], "fixture-1")
                self.assertEqual(self.teacher_requests[0]["input_ids"], [1, 2, 3, 8, 9])
                self.assertEqual(proxy.records[0]["teacher_log_probs"], [-1.5, -1.5])
                class Sample(SimpleNamespace):
                    Status = SimpleNamespace(COMPLETED="completed", TRUNCATED="truncated")
                parent = SimpleNamespace(index=7, group_index=2, prompt="question", session_id="test", metadata={})
                samples = samples_from_records(parent, proxy.records * 2, {"finished": False}, directory, Sample)
                self.assertEqual([s.group_id for s in samples], [7, 7])
                self.assertEqual(samples[0].tokens, [1, 2, 3, 8, 9])
                self.assertEqual(samples[0].loss_mask, [1, 1])
                self.assertEqual(samples[0].reward, {"opd_weight": 1.0})
                self.assertIsNone(samples[0].metadata["native_task_reward"])
            with gzip.open(directory / "turns.jsonl.gz", "rt") as stream:
                self.assertEqual(len(list(stream)), 1)

    async def test_version_change_and_teacher_misalignment_fail(self):
        with tempfile.TemporaryDirectory() as root:
            async with EpisodeProxy(builder=FixtureBuilder(), task={}, student_url=self.student_url,
                                    config=self.config(), output=Path(root), session_id="test", sampling_params={}) as proxy:
                body = {"model": "student", "messages": [{"role": "user", "content": "question"}]}
                await proxy.complete(body)
                self.version = "fixture-2"
                with self.assertRaisesRegex(ValueError, "weights changed"):
                    await proxy.complete(body)
                self.version = "mixed"
                with self.assertRaisesRegex(ValueError, "mixed weight_version"):
                    await proxy.complete(body)
                self.version = "fixture-1"; self.misalign = True
                with self.assertRaisesRegex(ValueError, "not aligned"):
                    await proxy.complete(body)


if __name__ == "__main__":
    unittest.main()
