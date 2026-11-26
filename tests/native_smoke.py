"""Real Pi/SQLite/token cache, deterministic model HTTP fixtures; no GPU training."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sqlite3
import sys
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from aiohttp import web

from finance_opd.config import ROOT, add_dependency_paths, read_config
from finance_opd.episode import EpisodeProxy, run_native, write_json


def text_builder(config):
    from transformers import AutoTokenizer
    add_dependency_paths(config)
    tokenizer = AutoTokenizer.from_pretrained(config["student_model"], local_files_only=True)
    from rl.examples.pi_finance.qwen_mask_builder import QwenFinanceMaskBuilder
    return tokenizer, QwenFinanceMaskBuilder(tokenizer, processor=None)


async def run(output: Path, config: dict, *, profile_task=None, shared_text_backend=None, protocol_failure=False, protocol_recovery=False, schema_failure=False, queued=False):
    output.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(config)
    if schema_failure:
        assert protocol_recovery
        protocol_failure = True
    if protocol_recovery:
        assert protocol_failure
        config["runtime"]["protocol_recovery_turns"] = 2
    if shared_text_backend is None:
        tokenizer, builder = text_builder(config)
    else:
        tokenizer, builder_type = shared_text_backend
        builder = builder_type(tokenizer, processor=None)
    if profile_task is None:
        database = output / "facts.db"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE facts (value INTEGER)")
            connection.execute("INSERT INTO facts VALUES (137)")
        task = {"task": "Read the value from the facts table and report it.", "benchmark": "ddr_10k",
                "id": "native-smoke", "native_case_id": "fixture", "source": {"db_path": str(database)}}
        action = {"name": "execute_query", "arguments": {"query": "SELECT value FROM facts;"}}
    else:
        task = copy.deepcopy(profile_task)
        action = None
        if task["source"].get("db_path"):
            action = {"name": "execute_query", "arguments": {"query": "SELECT 137 AS opd_smoke;"}}
        elif task["source"].get("data_path"):
            action = {"name": "list_files", "arguments": {"limit": 5, "recursive": False}}
    expected_turns = 2 if action else 1
    if protocol_recovery:
        assert action is not None
        expected_turns = 3
    calls, teacher_calls = [], []

    async def student(request):
        body = await request.json()
        calls.append(body)
        answer = ("<tool_call>\n<function=" + action["name"] + ">\n" + "".join(
            "<parameter=" + key + ">\n" + (json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value)
            + "\n</parameter>\n" for key, value in action["arguments"].items())
            + "</function>\n</tool_call>" if action and len(calls) == (2 if protocol_recovery else 1) else "FINISH: Native adapter validation completed.")
        if protocol_failure and len(calls) == 1:
            answer = ('<tool_call><function=execute_query><parameter=query>SELECT value FROM facts;</parameter>'
                      '<parameter=limit>101</parameter></function></tool_call>' if schema_failure else
                      '<tool_call><function=not_a_registered_tool></function></tool_call>')
        ids = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        return web.json_response({"text": answer, "meta_info": {
            "output_token_logprobs": [[-0.7, token] for token in ids],
            "weight_version": "fixture-1", "finish_reason": {"type": "stop"}}})

    async def teacher(request):
        body = await request.json()
        teacher_calls.append(body)
        ids = body["input_ids"]
        return web.json_response({"meta_info": {"input_token_logprobs":
            [[None, ids[0]]] + [[-0.4, token] for token in ids[1:]]}})

    runners = []
    async def serve(handler):
        app = web.Application(); app.router.add_post("/generate", handler)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        runners.append(runner)
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        return f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    try:
        student_url = await serve(student)
        config["teacher"]["url"] = await serve(teacher)
        config["runtime"].update(context_length=config["runtime"].get("context_length",16384), max_new_tokens=512, max_tool_calls=3,
                                 max_runtime_seconds=180, pi_python=sys.executable,
                                 request_timeout_seconds=90)
        if queued:
            import time
            from finance_opd.episode_queue import EpisodeWorker, queued_episode
            config['runtime'].setdefault('thinking','off')
            config['runtime']['episode_worker_queue']=str(output/'queue')
            config['job']={'deadline_epoch':time.time()+180}
            config['dataset']=str(output/'tasks.jsonl')
            Path(config['dataset']).write_text(json.dumps({'metadata':{k:v for k,v in task.items() if k!='task'},'task':task['task']})+'\n')
            worker=EpisodeWorker(config,output)
            worker.tokenizer=tokenizer
            serving=asyncio.create_task(worker.serve())
            try:
                status,records=await queued_episode(config,task,student_url,output/'episode','episode',{})
            finally:
                serving.cancel()
                await asyncio.gather(serving,return_exceptions=True)
        else:
            async with EpisodeProxy(builder=builder, task=task, student_url=student_url, config=config,
                                    output=output / "episode", session_id="native-smoke", sampling_params={}) as proxy:
                status = await run_native(config, task, proxy)
                records = list(proxy.records)
        result = json.loads((output / "episode/native/result.json").read_text())
        if status["exit_code"] != 0 or not status["finished"] or len(records) != expected_turns:
            raise AssertionError(f"native smoke did not finish {expected_turns} model calls: {status}")
        if len(teacher_calls) != expected_turns:
            raise AssertionError("teacher did not score each native call")
        tool_calls = result.get("tool_calls", {})
        if tool_calls.get("total") != (0 if protocol_failure and not protocol_recovery else int(action is not None)) or tool_calls.get("errors") != 0:
            raise AssertionError(f"unexpected tool execution: {tool_calls}")
        if action and (not protocol_failure or protocol_recovery) and tool_calls.get("by_tool", {}).get(action["name"]) != 1:
            raise AssertionError(f"expected native tool did not execute: {tool_calls}")
        # The second REAL model prompt must contain both the exact generated action
        # and a REAL SQLite observation. This detects cache/history reconstruction loss.
        for first, following in zip(records, records[1:]):
            if following["prompt_ids"][:len(first["prompt_ids"]) + len(first["response_ids"])] != first["prompt_ids"] + first["response_ids"]:
                raise AssertionError("following turn lost exact sampled-token prefix")
        if action and (not protocol_failure or protocol_recovery) and action["name"] == "execute_query" and "137" not in tokenizer.decode(records[-1]["prompt_ids"]):
            raise AssertionError("real SQLite observation missing from next prompt")
        for record, teacher_body in zip(records, teacher_calls):
            if teacher_body["input_ids"] != record["teacher_prompt_ids"] + record["response_ids"]:
                raise AssertionError("teacher scored a different sampled sequence")
        if protocol_failure and not schema_failure and not records[0].get("model_protocol_error"):
            raise AssertionError("injected malformed action was not recorded as a model protocol error")
        if schema_failure and records[0].get("model_protocol_error"):
            raise AssertionError("native schema fixture failed before native argument validation")
        if protocol_recovery:
            events = [json.loads(line).get("event", {}) for line in (output / "episode/native/events.jsonl").read_text().splitlines()]
            if sum(e.get("type") == "protocol_recovery_turn" for e in events) != 1 or any(e.get("type") == "finish_repair_turn" for e in events):
                raise AssertionError("protocol repair failed to preserve research before natural completion")
        report = {"passed": True, "protocol_failure_injected": protocol_failure, "benchmark": task["benchmark"], "task_id": task["id"],
                  "protocol_recovery_enabled": protocol_recovery, "native_schema_failure_injected": schema_failure,
                  "real_native_tools": action is not None, "real_tokenizer_and_cache": True,
                  "model_services": "deterministic HTTP fixtures", "vision_helpers": "text-only test stubs",
                  "gpu_training_executed": False, "cpu_episode_queue": queued, "generated_turns": len(records),
                  "native_result": result, "episode_status": status}
        write_json(output / "VALIDATION.json", report)
        return report
    finally:
        for runner in reversed(runners):
            await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen35_9b.json")
    parser.add_argument("--output", type=Path, default=ROOT / "validation" / datetime.now().strftime("native-%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("smoke artifacts must remain in cv-opd")
    report = asyncio.run(run(output, read_config(args.config)))
    print(json.dumps({k: v for k, v in report.items() if k != "native_result"}, indent=2))
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
