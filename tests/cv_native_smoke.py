"""Real Pi + SQLite + tokenizer + CV controller; deterministic model/score fixtures."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sqlite3
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from aiohttp import web

from finance_opd.config import ROOT, read_config
from finance_opd.cv import GUIDANCE_MARKER
from finance_opd.episode import EpisodeProxy, run_native, write_json
from native_smoke import text_builder


async def run(output: Path, config: dict):
    output.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(config)
    tokenizer, builder = text_builder(config)
    database = output / "facts.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE facts(value INTEGER)")
        connection.execute("INSERT INTO facts VALUES (137)")
    task = {"task": "Read the value from the facts table and report it.", "benchmark": "ddr_10k",
            "id": "cv-native-fixture", "native_case_id": "fixture", "source": {"db_path": str(database)}}
    action = "<tool_call>\n<function=execute_query>\n<parameter=query>\nSELECT value FROM facts;\n</parameter>\n</function>\n</tool_call>"
    student_calls, teacher_scores, teacher_calls = [], [], []
    student_texts = ["FINISH: Initial report before checking the facts table.", action, "FINISH: The value is 137."]
    probe_calls = 0
    fixture = types.ModuleType("cv_native_score_fixture")
    fixture.score = lambda request: float("137" in request["report"])

    async def student(request):
        body = await request.json(); student_calls.append(body)
        text = student_texts.pop(0)
        ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
        return web.json_response({"text": text, "meta_info": {"output_token_logprobs": [[-0.7, token] for token in ids],
            "weight_version": "deterministic-fixture", "finish_reason": {"type": "stop"}}})

    async def teacher(request):
        nonlocal probe_calls
        body = await request.json()
        prefix = tokenizer.decode(body["input_ids"])
        if body["sampling_params"]["max_new_tokens"] == 0:
            teacher_scores.append(body)
            ids = body["input_ids"]
            return web.json_response({"meta_info": {"input_token_logprobs":
                [[None, ids[0]]] + [[-0.4, token] for token in ids[1:]]}})
        teacher_calls.append(body)
        if "Training-only continuation probe." in prefix:
            probe_calls += 1
            text = (action if probe_calls % 2 else "FINISH: The executed query establishes that the value is 137.")
            text = "Check the evidence.\n</think>\n" + text
        elif "Write a financial research answer" in prefix:
            text = "</think>\nFINISH: " + ("The value is 137." if "137" in prefix else "The evidence is insufficient.")
        elif "Audit a financial research continuation" in prefix:
            text = "</think>\n" + json.dumps({"evidence_valid": True, "safety_valid": True,
                "guidance": "Read the facts table with execute_query, then report the observed value with support.",
                "reason": "Deterministic verifier fixture; SQL observation is real."})
        else:
            raise AssertionError("unrecognized teacher request")
        ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
        return web.json_response({"text": text, "meta_info": {"output_token_logprobs": [[-0.4, token] for token in ids],
            "finish_reason": {"type": "stop"}, "prompt_tokens": len(body["input_ids"]), "completion_tokens": len(ids)}})

    runners = []
    async def serve(handler):
        app = web.Application(); app.router.add_post("/generate", handler)
        runner = web.AppRunner(app, access_log=None); await runner.setup(); runners.append(runner)
        site = web.TCPSite(runner, "127.0.0.1", 0); await site.start()
        return f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    try:
        student_url = await serve(student)
        config["teacher"].update(url=await serve(teacher), context_length=32768)
        config["runtime"].update(context_length=32768, native_context_window=16384,
            max_new_tokens=512, max_tool_calls=5, max_runtime_seconds=180, pi_python=sys.executable,
            request_timeout_seconds=90)
        config["continuation"] = {"enabled": True, "max_probes_per_episode": 2,
            "max_teacher_turns": 3, "max_teacher_tool_calls": 2, "max_teacher_seconds": 45,
            "max_probe_seconds": 65, "min_retry_seconds": 10,
            "max_teacher_new_tokens": 512, "writer_max_new_tokens": 512, "verifier_max_new_tokens": 512,
            "score_hook": "cv_native_score_fixture.score", "verifier_hook": "finance_opd.value.llm_evidence_verifier",
            "node_bin": str(ROOT / ".runtime/node24")}
        with patch.dict(sys.modules, {fixture.__name__: fixture}), patch.dict(os.environ, {"PI_NODE": str(ROOT / ".runtime/node24")}):
            async with EpisodeProxy(builder=builder, task=task, student_url=student_url, config=config,
                    output=output / "episode", session_id="cv-native-smoke", sampling_params={}) as proxy:
                status = await run_native(config, task, proxy)
                records = copy.deepcopy(proxy.records)
        if status["exit_code"] != 0 or not status["finished"]:
            raise AssertionError(f"real Pi did not finish: {status}")
        if len(student_calls) != 3 or student_calls[0]["input_ids"] != student_calls[1]["input_ids"]:
            raise AssertionError("student retry did not use exactly the original pre-FINISH input IDs")
        if any(GUIDANCE_MARKER in tokenizer.decode(call["input_ids"]) for call in student_calls):
            raise AssertionError("private teacher guidance leaked into a student prompt")
        trainable = [record for record in records if any(record["loss_mask"])]
        if len(trainable) != 1 or trainable[0]["supervision_kind"] != "cv_student_retry":
            raise AssertionError("mining or zero-gain turns entered CV training")
        if abs(trainable[0]["opd_weight"] - 0.98) > 1e-9:
            raise AssertionError("matched before/after gain did not produce the expected CV weight")
        if GUIDANCE_MARKER not in tokenizer.decode(trainable[0]["teacher_prompt_ids"]):
            raise AssertionError("successful continuation reference did not reach teacher scoring")
        for record, request in zip(records, teacher_scores):
            if request["input_ids"] != record["teacher_prompt_ids"] + record["response_ids"]:
                raise AssertionError("teacher did not score the exact student sampled IDs")
        result = json.loads((output / "episode/native/result.json").read_text())
        if result["tool_calls"]["total"] != 1 or result["tool_calls"]["errors"]:
            raise AssertionError("student did not execute the native SQLite query independently")
        values = [json.loads(path.read_text()) for path in sorted((output / "episode/continuation").glob("probe-*/value_audit.json"))]
        if len(values) != 2 or values[0]["status"] != "accepted" or values[1].get("weight") != 0:
            raise AssertionError("new stopping state did not trigger a fresh, zero-gain diagnostic")
        report = {"passed": True, "episode_status": status, "real_tokenizer_and_cache": True,
            "real_native_tools": True, "student_retry_same_input_ids": True,
            "teacher_guidance_private": True, "teacher_scores_exact_student_tokens": True,
            "positive_gain_weight": trainable[0]["opd_weight"], "new_stop_rechecked": True,
            "model_services": "deterministic HTTP fixtures", "task_quality_evaluator": "deterministic fixture, not a benchmark judge",
            "evidence_verifier": "deterministic fixture, not proof of verifier accuracy", "gpu_training_executed": False}
        write_json(output / "VALIDATION.json", report)
        return report
    finally:
        for runner in reversed(runners): await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cv_opd_qwen38_27b.json")
    parser.add_argument("--output", type=Path, default=ROOT / "validation" /
                        datetime.now(timezone.utc).strftime("cv-native-%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        raise ValueError("CV smoke artifacts must remain in cv-opd")
    print(json.dumps(asyncio.run(run(args.output.resolve(), read_config(args.config))), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
