import asyncio
import copy
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import ClientSession, web

from finance_opd.config import ROOT
from finance_opd.continuation import NativeTools, run_teacher_continuation, snapshot_workspace


class Builder:
    def __init__(self):
        self.contexts = []
        self.generations = []
        self.cleared = []

    def prepare_generate_input(self, session, messages):
        self.contexts.append(copy.deepcopy(messages))
        return SimpleNamespace(input_ids=[1, 2, 3])

    def record_generation(self, prepared, ids, logps, text, finish):
        self.generations.append((ids, text))

    def clear_session(self, session):
        self.cleared.append(session)


class FakeTools:
    def __init__(self, *args):
        self.steps = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def execute(self, text, identifier, remaining, timeout):
        if text.startswith("FINISH:"):
            return {"ok": True, "finish": True, "tool_calls": 0, "observations": []}
        if remaining < 1:
            return {"ok": False, "reason": "tool_budget", "tool_calls": 0, "observations": []}
        return {"ok": True, "finish": False, "tool_calls": 1, "observations": [
            {"role": "user", "content": json.dumps({"tool_result": {"id": identifier,
             "name": "execute_query", "is_error": False, "content": "{\"revenue\": 120}"}})}]}


class ContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.native = self.root / "native"
        (self.native / "raw/finance/input").mkdir(parents=True)
        (self.native / "raw/finance/input/notes.txt").write_text("cash flow 50")
        self.messages = [{"role": "system", "content": "Research the company."},
                         {"role": "user", "content": "Begin."}]
        self.outputs = ["query", "FINISH: verified"]
        self.finish = "stop"
        app = web.Application()
        app.router.add_post("/generate", self.teacher)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self.config = {"teacher": {"url": self.url, "context_length": 1000},
                       "runtime": {}, "continuation": {"max_teacher_turns": 4,
                       "max_teacher_tool_calls": 2, "max_teacher_seconds": 20},
                       "pi_finance": str(ROOT.parent / "pi-finance")}

    async def asyncTearDown(self):
        await self.runner.cleanup()
        self.temp.cleanup()

    async def teacher(self, request):
        payload = await request.json()
        self.assertEqual(payload["input_ids"], [1, 2, 3])
        self.assertTrue(payload["return_logprob"])
        return web.json_response({"text": self.outputs.pop(0), "meta_info": {
            "output_token_logprobs": [[-0.2, 8], [-0.5, 9]],
            "finish_reason": {"type": self.finish}}})

    async def run_probe(self, *, calls=5):
        builder = Builder()
        before = copy.deepcopy(self.messages)
        with patch("finance_opd.continuation.NativeTools", FakeTools):
            async with ClientSession() as client:
                result = await run_teacher_continuation(builder=builder, client=client, config=self.config,
                    task={"task": "research"}, messages=self.messages, native_dir=self.native,
                    output=self.root / "probe", session_id="root", remaining_tool_calls=calls,
                    remaining_seconds=20)
        self.assertEqual(self.messages, before)
        return result, builder

    async def test_teacher_only_context_real_observations_and_separate_cache(self):
        result, builder = await self.run_probe()
        self.assertTrue(result["completed"])
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["messages"][:2], self.messages)
        self.assertEqual(len(result["new_observations"]), 1)
        self.assertNotIn("Training-only", json.dumps(result["messages"]))
        self.assertIn("Training-only", builder.contexts[0][0]["content"])
        self.assertIn("revenue", json.dumps(builder.contexts[1]))
        self.assertFalse(result["audit"]["teacher_tokens_are_training_targets"])
        self.assertEqual(builder.cleared, ["root-continuation"])

    async def test_teacher_may_finish_without_new_evidence_but_not_a_stop_label(self):
        self.outputs = ["FINISH: no apparent gap"]
        result, _ = await self.run_probe(calls=0)
        self.assertTrue(result["completed"])
        self.assertEqual(result["new_observations"], [])
        self.assertFalse(result["audit"]["empty_probe_is_stop_label"])

    async def test_truncated_teacher_generation_never_executes_tools(self):
        self.finish = "length"
        result, _ = await self.run_probe()
        self.assertFalse(result["completed"])
        self.assertEqual(result["audit"]["reason"], "generation_truncated")
        self.assertEqual(result["tool_calls"], 0)

    async def test_original_remaining_budget_caps_teacher(self):
        self.outputs = ["query", "query"]
        result, _ = await self.run_probe(calls=1)
        self.assertFalse(result["completed"])
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["audit"]["reason"], "tool_budget")

    async def test_prior_web_cache_cannot_be_recreated_as_same_state(self):
        self.messages.append({"role": "user", "content": json.dumps({"tool_result": {
            "id": "web-1", "name": "web_search", "is_error": False, "content": "responseId: old-process"}})})
        result, builder = await self.run_probe()
        self.assertFalse(result["completed"])
        self.assertEqual(result["audit"]["reason"], "unsupported_process_local_web_state")
        self.assertEqual(builder.contexts, [])
        self.assertFalse((self.root / "probe/workspace").exists())

    async def test_materialized_web_history_reaches_the_teacher_probe(self):
        self.messages.append({"role":"tool","name":"fetch_content","tool_call_id":"web-1","content":"Prior fetched revenue 120"})
        state={"type":"custom","customType":"web-search-results","data":{"id":"prior-web","type":"fetch","timestamp":int(time.time()*1000),"urls":[{"url":"https://example.org/report","title":"Report","content":"Revenue 120","error":None}]}}
        (self.native/"web-state.jsonl").write_text(json.dumps(state)+"\n")
        result,builder=await self.run_probe()
        self.assertTrue(result["completed"])
        self.assertTrue(builder.contexts)
        self.assertEqual((self.root/"probe/workspace/web-state.jsonl").read_text(),(self.native/"web-state.jsonl").read_text())



class SnapshotTests(unittest.TestCase):
    def test_links_versions_and_artifacts_are_preserved_without_cache(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            native = root / "native"
            (native / "raw/finance/input").mkdir(parents=True)
            (native / "tables").mkdir()
            (native / "raw/finance/input/result.csv").write_text("x\n1\n")
            for name in ("report.md", "result.json", "events.jsonl", "archive.gz", "debug.log"):
                (native / "raw/finance/input" / name).write_text("original financial material")
            (native / "raw/finance/input/logs").mkdir()
            (native / "raw/finance/input/logs/evidence.txt").write_text("source evidence")
            (native / "tables/internal").symlink_to(native / "raw/finance/input/result.csv")
            corpus = root / "source.db"
            corpus.write_bytes(b"source bytes")
            (native / "tables/database").symlink_to(corpus)
            (native / "pi-state").mkdir()
            (native / "pi-state/cache").write_text("do not copy")
            (native / "events.jsonl").write_text("do not copy")
            snapshot = snapshot_workspace(native, root / "copy")
            self.assertEqual((root / "copy/tables/internal").resolve(), root / "copy/raw/finance/input/result.csv")
            self.assertEqual((root / "copy/tables/database").resolve(), corpus)
            self.assertFalse((root / "copy/pi-state").exists())
            self.assertFalse((root / "copy/events.jsonl").exists())
            self.assertIn(str(corpus), snapshot["protected"])
            self.assertEqual(len(snapshot["snapshot_sha256"]), 64)
            self.assertEqual(snapshot["external_source_versions"][0]["size"], len(b"source bytes"))
            for name in ("report.md", "result.json", "events.jsonl", "archive.gz", "debug.log", "logs/evidence.txt"):
                self.assertEqual((root / "copy/raw/finance/input" / name).read_text(),
                                 (native / "raw/finance/input" / name).read_text())


@unittest.skipUnless((ROOT / ".runtime/node24").is_file() and
                     (ROOT.parent / "pi-finance/node_modules/jiti/lib/jiti-cli.mjs").is_file(),
                     "real native tool RPC requires the existing Pi Node/jiti installation")
class RealNativeToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_unmarked_teacher_report_ends_without_a_synthetic_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native"
            (native / "raw/finance/input").mkdir(parents=True)
            (native / "runtime_config.json").write_text(json.dumps({"expectedTools": ["list_files"]}))
            output = root / "probe"
            output.mkdir()
            snapshot = snapshot_workspace(native, output / "workspace")
            config = {"pi_finance": str(ROOT.parent / "pi-finance"), "runtime": {"pi_python": os.sys.executable},
                      "continuation": {"node_bin": str(ROOT / ".runtime/node24")}}
            async with NativeTools(config, {}, native, snapshot, output, 3, 30) as tools:
                for text, kind in [("Summary first.\nFINISH: Revenue is 120.", "unmarked_no_tool_stop"),
                                   ("The evidence is sufficient.", "unmarked_no_tool_stop"),
                                   ("FINISH: Revenue is 120.", "explicit_finish")]:
                    result = await tools.execute(text, "terminal", 3, 10)
                    self.assertTrue(result["ok"], result)
                    self.assertTrue(result["finish"])
                    self.assertEqual(result["finish_kind"], kind)
                    self.assertEqual(result["observations"], [])
                    self.assertEqual(result["tool_calls"], 0)
                empty = await tools.execute("<think>Still reasoning</think>", "empty", 3, 10)
                self.assertFalse(empty["ok"])
                self.assertFalse(empty["finish"])

    async def test_copied_web_state_preserves_prior_response_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native"
            native.mkdir()
            (native / "runtime_config.json").write_text(json.dumps({"expectedTools": ["get_search_content"]}))
            state = {"type": "custom", "customType": "web-search-results", "data": {
                "id": "prior-fetch", "type": "fetch", "timestamp": int(time.time() * 1000),
                "urls": [{"url": "https://example.org/filing", "title": "Fixture filing",
                          "content": "Verified revenue 137 million USD.", "error": None}]}}
            original = json.dumps(state) + "\n"
            (native / "web-state.jsonl").write_text(original)
            output = root / "probe"
            output.mkdir()
            snapshot = snapshot_workspace(native, output / "workspace")
            config = {"pi_finance": str(ROOT.parent / "pi-finance"), "runtime": {"pi_python": os.sys.executable},
                      "continuation": {"node_bin": str(ROOT / ".runtime/node24")}}
            async with NativeTools(config, {}, native, snapshot, output, 3, 30) as tools:
                text = ("<tool_call>\n<function=get_search_content>\n"
                        "<parameter=responseId>prior-fetch</parameter>\n"
                        "<parameter=urlIndex>0</parameter>\n</function>\n</tool_call>")
                result = await tools.execute(text, "read-prior", 2, 15)
                self.assertTrue(result["ok"], result)
                self.assertIn("Verified revenue 137 million USD", result["observations"][0]["content"])
            self.assertEqual((native / "web-state.jsonl").read_text(), original)

    async def test_sql_files_code_and_helper_refusal_use_actual_pi_tools(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            native = root / "native"
            (native / "raw/finance/input").mkdir(parents=True)
            (native / "tables/finance").mkdir(parents=True)
            (native / "raw/finance/input/data.csv").write_text("revenue\n120\n")
            source = root / "source.db"
            with sqlite3.connect(source) as conn:
                conn.execute("CREATE TABLE facts(revenue INTEGER)")
                conn.execute("INSERT INTO facts VALUES (120)")
            (native / "tables/finance/finance.db").symlink_to(source)
            names = ["get_database_info", "describe_table", "execute_query", "execute_code", "list_files",
                     "get_field_description", "retrieve_information"]
            (native / "runtime_config.json").write_text(json.dumps({"expectedTools": names}))
            output = root / "probe"
            output.mkdir()
            snapshot = snapshot_workspace(native, output / "workspace")
            config = {"pi_finance": str(ROOT.parent / "pi-finance"), "runtime": {"pi_python": os.sys.executable},
                      "continuation": {"node_bin": str(ROOT / ".runtime/node24"),
                                       "references_path": str(root / "gold.json")}}
            (root / "gold.json").write_text("PRIVATE_GOLD_SENTINEL")
            async with NativeTools(config, {}, native, snapshot, output, 5, 30) as tools:
                async def execute(name, args, remaining=4):
                    text = "<tool_call>\n<function=" + name + ">\n" + "".join(
                        "<parameter=" + key + ">\n" + (value if isinstance(value, str) else json.dumps(value))
                        + "\n</parameter>\n" for key, value in args.items()) + "</function>\n</tool_call>"
                    return await tools.execute(text, name, remaining, 15)
                sql = await execute("execute_query", {"query": "SELECT revenue FROM facts"})
                self.assertTrue(sql["ok"], sql)
                self.assertEqual(sql["observations"][0]["role"], "tool")
                self.assertEqual(sql["observations"][0]["arguments"], {"query": "SELECT revenue FROM facts"})
                self.assertIn("120", sql["observations"][0]["content"])
                files = await execute("list_files", {})
                self.assertIn("data.csv", files["observations"][0]["content"])
                code = await execute("execute_code", {"code": "from pathlib import Path; print(Path('data.csv').read_text())"})
                self.assertIn("120", code["observations"][0]["content"])
                write = await execute("execute_code", {"code": f"open({str(source)!r}, 'w').write('bad')"})
                self.assertIn("read-only enforcement", write["observations"][0]["content"])
                with sqlite3.connect(source) as conn:
                    self.assertEqual(conn.execute("SELECT revenue FROM facts").fetchone()[0], 120)
                private = await execute("execute_code", {"code": f"print(open({str(root / 'gold.json')!r}).read())"})
                self.assertIn("private supervision file access denied", private["observations"][0]["content"])
                self.assertNotIn("PRIVATE_GOLD_SENTINEL", private["observations"][0]["content"])
                helper = await execute("retrieve_information", {"prompt": "read {{report}}"})
                self.assertFalse(helper["ok"])
                self.assertIn("unsupported_llm_helper", helper["reason"])
                self.assertEqual(helper["tool_calls"], 0)


if __name__ == "__main__":
    unittest.main()
