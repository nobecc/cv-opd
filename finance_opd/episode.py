"""Native Pi episode + ephemeral loopback proxy. No Gateway/Buffer fork."""
from __future__ import annotations

import asyncio
import copy
import gzip
import json
import os
import shlex
import signal
import sys
import time
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web
import yaml

from .protocol import freeze_teacher_view, sampled_output
PROTOCOL = "qwen3_xml_v1"
from .teacher_transport import score_teacher
from .cv import ContinuationController, PROTOCOL_MARKER, message_hash, natural_finish, public_tool_calls, visible_answer


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


async def post_json(client, url: str, payload: dict, *, headers=None) -> dict:
    async with client.post(url, json=payload, headers=headers) as response:
        if response.status >= 400:
            detail = (await response.text())[:2000]
            raise aiohttp.ClientResponseError(response.request_info, response.history,
                status=response.status, message=detail, headers=response.headers)
        result = await response.json()
    if not isinstance(result, dict):
        raise ValueError("engine must return one JSON object")
    return result


class ContextBudgetExceeded(ValueError):
    """One trajectory reached its token budget; no unscored action was sampled."""


def output_budget(config, student_prefix_length, teacher_prefix_length, requested):
    runtime = config["runtime"]
    student_limit = int(runtime.get("context_length", 262144))
    teacher_limit = int(config["teacher"].get("context_length", student_limit))
    margin = runtime.get("context_safety_margin", 8)
    if type(margin) is not int or margin < 1:
        raise ValueError("context_safety_margin must be a positive integer")
    available = min(student_limit - margin - student_prefix_length,
                    teacher_limit - margin - teacher_prefix_length)
    if available <= 0:
        raise ContextBudgetExceeded(
            f"context budget exhausted before sampling: student_prefix={student_prefix_length}, "
            f"teacher_prefix={teacher_prefix_length}, student_limit={student_limit}, "
            f"teacher_limit={teacher_limit}, safety_margin={margin}")
    if int(requested) <= 0:
        raise ValueError("requested generation tokens must be positive")
    return min(int(requested), available)


class EpisodeProxy:
    def __init__(self, *, builder, task: dict, student_url: str, config: dict,
                 output: Path, session_id: str, sampling_params: dict):
        self.builder, self.task = builder, copy.deepcopy(task)
        self.student_url, self.config = student_url.rstrip("/"), config
        self.output, self.session_id = output, session_id
        self.sampling_params = dict(sampling_params)
        self.skip_context_teacher_scoring = config["runtime"].get("skip_context_teacher_scoring", False)
        if type(self.skip_context_teacher_scoring) is not bool:
            raise ValueError("skip_context_teacher_scoring must be boolean")
        if self.skip_context_teacher_scoring and not config.get("continuation", {}).get("enabled", False):
            raise ValueError("context-only teacher skip requires explicit CV screening")
        self.helper_sessions = set()
        self.records = []
        self.returned_turns = 0
        self.failure = None
        self.request_stage = "initializing"
        self.request_endpoint = None
        self.context_exhausted = None
        self.lock = asyncio.Lock()
        self.client = None
        self.started = None
        self.continuation = (ContinuationController(config, task, output)
                             if config.get("continuation", {}).get("enabled", False) else None)

    async def __aenter__(self):
        self.started = time.monotonic()
        self.client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(
            total=self.config["runtime"].get("request_timeout_seconds", 600)), trust_env=False,
            connector=aiohttp.TCPConnector(force_close=True))
        app = web.Application(client_max_size=64 * 1024**2)
        app.router.add_post("/v1/sessions/{session}/chat/completions", self.handle)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}/v1/sessions/{self.session_id}"
        self.output.mkdir(parents=True, exist_ok=True)
        self.audit = gzip.open(self.output / "turns.jsonl.gz", "xt", encoding="utf-8")
        return self

    async def __aexit__(self, *exc):
        await self.runner.cleanup()
        await self.client.close()
        self.audit.close()
        self.builder.clear_session(self.session_id)
        for helper_session in self.helper_sessions:
            self.builder.clear_session(helper_session)

    async def handle(self, request):
        if request.match_info["session"] != self.session_id:
            raise web.HTTPNotFound()
        async with self.lock:
            try:
                return web.json_response(await self.complete(await request.json()))
            except ContextBudgetExceeded as error:
                self.context_exhausted = str(error)
                write_json(self.output / "context_budget.json", {
                    "reason": self.context_exhausted, "scored_turns": len(self.records),
                    "student_sampled_on_this_request": False})
                return web.json_response({"error": {"type": "context_length_exceeded",
                    "message": self.context_exhausted}}, status=400)
            except Exception as error:
                self.failure = f"{type(error).__name__}: {error}"
                write_json(self.output / "proxy_error.json", {"error": self.failure,
                    "stage": self.request_stage, "endpoint": self.request_endpoint,
                    "recorded_turns": len(self.records), "time": time.time()})
                raise web.HTTPBadGateway(text=self.failure) from error

    async def teacher_prefix(self, messages, prefix, view):
        if view["messages"] == messages:
            return list(prefix)
        teacher_session = f"{self.session_id}-teacher-{uuid.uuid4().hex}"
        if hasattr(self.builder, "fork_session"):
            self.builder.fork_session(self.session_id, teacher_session)
        try:
            prepared = await asyncio.to_thread(self.builder.prepare_generate_input, teacher_session, view["messages"])
            return list(prepared.input_ids)
        finally:
            self.builder.clear_session(teacher_session)

    def remaining_research_budget(self, messages):
        runtime = self.config["runtime"]
        settings_file = self.output / "native_config.yaml"
        settings = yaml.safe_load(settings_file.read_text()) if settings_file.is_file() else {}
        analysis = settings.get("analysis", {})
        total_calls = int(runtime.get("max_tool_calls", analysis.get("max_tool_calls", 0)))
        total_seconds = float(runtime.get("max_runtime_seconds", analysis.get("max_runtime_seconds", 0)))
        total_seconds -= float(settings.get("run", {}).get("finish_grace_seconds", 0))
        return max(0, total_calls - public_tool_calls(messages)), max(0, total_seconds - (time.monotonic() - self.started))

    async def sample_student(self, payload, prefix, teacher_prefix):
        params = dict(self.sampling_params)
        params.update(temperature=1.0, top_p=1.0, top_k=-1,
                      max_new_tokens=output_budget(self.config, len(prefix), len(teacher_prefix),
                                                   payload.get("max_tokens") or 8192))
        # Preserve protocol/reasoning/EOS tokens in the authoritative wire transcript.
        params["skip_special_tokens"] = False
        params["no_stop_trim"] = True
        request_body = {"input_ids": prefix, "sampling_params": params, "return_logprob": True, "stream": False}
        self.request_stage, self.request_endpoint = "student_generation", self.student_url + "/generate"
        response = await post_json(self.client, self.student_url + "/generate", request_body,
                                   headers={"X-SMG-Routing-Key": self.session_id})
        # Persist the exact successful engine response before parsing or teacher I/O.
        # This journal is not a training record and has no inferred probabilities/mask.
        sample_id = uuid.uuid4().hex
        journal = self.output / "sampled.private"
        journal.mkdir(exist_ok=True)
        sample_path = journal / f"{sample_id}.json.gz"
        with sample_path.open("xb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb") as stream:
                stream.write(json.dumps({"student_request": request_body, "response": response,
                    "teacher_prompt_ids": list(teacher_prefix), "time": time.time(),
                    "status": "sampled_before_teacher_scoring"}, ensure_ascii=False, allow_nan=False).encode())
            raw.flush()
            os.fsync(raw.fileno())
        self.request_stage = "student_response_validation"
        ids, logps, text, finish, version = sampled_output(response)
        reasoning_prefilled = False
        if hasattr(self.builder, "configure_session"):
            text = self.builder.tokenizer.decode(ids, skip_special_tokens=False)
            reasoning_prefilled = self.builder.tokenizer.decode(
                prefix[-16:], skip_special_tokens=False).rstrip().endswith("<think>")
        if len(ids) > params["max_new_tokens"]:
            raise ValueError("student returned more tokens than the shared scoring budget")
        if version is not None and (type(version) not in {int, str} or str(version).lower() in {"", "mixed", "unknown"}):
            raise ValueError("student engine reported an invalid or mixed weight_version")
        if version is None and self.config["runtime"].get("require_weight_version", True):
            raise ValueError("student engine did not report its weight_version")
        if self.records and version != self.records[0]["weight_version"]:
            raise ValueError("student weights changed inside one native episode")
        parsed_message, protocol_error = {"role": "assistant", "content": text}, None
        if (payload.get("metadata") or {}).get("tool_protocol") == PROTOCOL:
            from rl.examples.pi_finance.qwen_protocol import parse_response
            try:
                parsed_message = parse_response(text, payload.get("tools") or [],
                    identifier=f"opd-{self.session_id}-{len(self.records)}", truncated=finish == "length")
            except ValueError as exc:
                protocol_error = str(exc)
        return {"ids": ids, "logps": logps, "text": text, "finish": finish, "version": version,
                "sample_journal": str(sample_path),
                "meta": response["meta_info"], "params": params,
                "parsed_message": parsed_message, "model_protocol_error": protocol_error,
                "reasoning_prefilled": reasoning_prefilled}

    async def score_and_record(self, prefix, teacher_prefix, view, generation, *, train=True, kind="plain_opd"):
        ids, logps = generation["ids"], generation["logps"]
        text, finish, version = generation["text"], generation["finish"], generation["version"]
        teacher_limit = self.config["teacher"].get("context_length", self.config["runtime"].get("context_length", 262144))
        if len(teacher_prefix) + len(ids) > teacher_limit:
            raise ValueError("teacher context exceeds its configured limit; refusing silent truncation")
        teacher_url = self.config["teacher"]["url"].rstrip("/")
        if teacher_url == self.student_url:
            raise ValueError("teacher must be a separate frozen service")
        teacher_scored = train or not self.skip_context_teacher_scoring
        self.request_stage, self.request_endpoint = "teacher_token_scoring", teacher_url
        teacher_logps = (await score_teacher(self.client, self.config, teacher_prefix, ids)) if teacher_scored else None
        record = {"schema": "cv_opd.turn.v1", "sample_journal": generation.get("sample_journal"), "turn": len(self.records), "prompt_ids": list(prefix), "response_ids": ids,
                  "rollout_log_probs": logps, "teacher_log_probs": teacher_logps,
                  "teacher_scoring_status": "scored" if teacher_scored else "not_requested_zero_loss_screening_context",
                  "response": text, "finish_reason": finish, "weight_version": version,
                  "teacher_prompt_ids": teacher_prefix, "opd_weight": view["weight"] if train else 0.0,
                  "loss_mask": [int(train)] * len(ids), "supervision_kind": kind,
                  "student_prefix_sha256": view["student_prefix_sha256"], "context_audit": view["audit"],
                  "generation_token_budget": generation["params"]["max_new_tokens"],
                  "sampling_params": copy.deepcopy(generation["params"]),
                  "student_model": self.config.get("student_model"),
                  "teacher_model": self.config["teacher"].get("model"),
                  "reasoning_prefilled": generation.get("reasoning_prefilled", False),
                  "model_protocol_error": generation.get("model_protocol_error")}
        self.records.append(record)
        self.audit.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.audit.flush()
        return record

    async def complete(self, payload: dict) -> dict:
        self.request_stage, self.request_endpoint = "prepare_prefix", None
        messages = payload["messages"]
        qwen = (payload.get("metadata") or {}).get("tool_protocol") == PROTOCOL
        channel = (payload.get("metadata") or {}).get("channel")
        if qwen and channel not in {"agent", "document_helper"}:
            raise ValueError("unknown Qwen request channel")
        cache_session = self.session_id
        if qwen and channel == "document_helper":
            cache_session = f"{self.session_id}-helper-{uuid.uuid4().hex}"
            self.helper_sessions.add(cache_session)
        if qwen:
            from rl.examples.pi_finance.qwen_protocol import parse_response, request_settings
            tools, thinking = request_settings(payload)
            self.builder.configure_session(cache_session, tools, thinking)
            if channel == "document_helper" and tools:
                raise ValueError("document helper must be tool-free")
        elif self.config.get("runtime", {}).get("tool_protocol", PROTOCOL) != "legacy_json":
            raise ValueError("OPD requires the explicit Qwen XML request contract")
        main = ((payload.get("metadata") or {}).get("channel") == "agent" if qwen else
                bool(messages and messages[0].get("role") == "system"
                     and PROTOCOL_MARKER in messages[0]["content"]))
        if self.continuation:
            view = self.continuation.view(messages) if main else {
                "messages": copy.deepcopy(messages), "weight": 0.0,
                "audit": {"mode": "cv_helper_context_only"},
                "student_prefix_sha256": message_hash(messages)}
        else:
            view = freeze_teacher_view(self.config.get("context_hook"), messages, self.task, len(self.records))
        # Do not commit a mining FINISH to the Pi token cache. A retry uses these
        # exact input IDs, without a user continuation prompt or teacher evidence.
        prepared = await asyncio.to_thread(self.builder.prepare_generate_input, cache_session, messages)
        prefix = list(prepared.input_ids)
        teacher_prefix = await self.teacher_prefix(messages, prefix, view)
        generation = await self.sample_student(payload, prefix, teacher_prefix)
        if self.continuation and main:
            remaining_calls, remaining_seconds = self.remaining_research_budget(messages)
            if not generation.get("model_protocol_error") and natural_finish(
                    messages, generation["text"], generation["finish"], remaining_tool_calls=remaining_calls,
                    remaining_seconds=remaining_seconds, is_main=main,
                    allow_unmarked_stop=self.config["continuation"].get("mine_unmarked_stops", False),
                    reasoning_prefilled=generation.get("reasoning_prefilled", False)):
                stop_kind = "explicit_finish" if visible_answer(generation["text"]).upper().startswith("FINISH:") else "unmarked_no_tool_stop"
                stops = self.output / "continuation"
                stops.mkdir(exist_ok=True)
                with (stops / "stop_candidates.jsonl").open("a") as stream:
                    stream.write(json.dumps({"turn": len(self.records), "stop_kind": stop_kind,
                                             "student_prefix_sha256": view["student_prefix_sha256"]}) + "\n")
                # The first sample locates a stopping state. It remains an audited
                # student generation, but is not a teacher-informed training target.
                await self.score_and_record(prefix, teacher_prefix, view, generation, train=False, kind="cv_stop_mining")
                self.request_stage, self.request_endpoint = "continuation_probe", self.config["teacher"]["url"]
                active = await self.continuation.investigate(messages=messages, builder=self.builder, client=self.client,
                    remaining_tool_calls=remaining_calls, remaining_seconds=remaining_seconds, session_id=self.session_id,
                    student_stop_report=generation["text"])
                if active:
                    _, remaining_seconds = self.remaining_research_budget(messages)
                    if remaining_seconds > self.config["continuation"]["min_retry_seconds"]:
                        view = self.continuation.view(messages)
                        teacher_prefix = await self.teacher_prefix(messages, prefix, view)
                        generation = await self.sample_student(payload, prefix, teacher_prefix)
                        await self.score_and_record(prefix, teacher_prefix, view, generation, train=True, kind="cv_student_retry")
                    else:
                        self.continuation.active = None
                        write_json(self.output / "continuation/retry_budget.json", {
                            "status": "insufficient_retry_budget", "student_resampled": False,
                            "opportunity_label": None})
            else:
                forced_finish = visible_answer(generation["text"]).upper().startswith("FINISH:")
                await self.score_and_record(prefix, teacher_prefix, view, generation,
                    train=view["weight"] > 0 and not forced_finish,
                    kind="cv_student_continuation" if view["weight"] > 0 and not forced_finish else "cv_context_only")
        else:
            await self.score_and_record(prefix, teacher_prefix, view, generation,
                                        train=self.continuation is None, kind="plain_opd" if not self.continuation else "cv_helper_context_only")
        await asyncio.to_thread(self.builder.record_generation, prepared, generation["ids"],
                                generation["meta"]["output_token_logprobs"], generation["text"], generation["finish"])
        if cache_session != self.session_id:
            self.builder.clear_session(cache_session)
            self.helper_sessions.discard(cache_session)
        turn = len(self.records) - 1
        ids, text, finish, version = generation["ids"], generation["text"], generation["finish"], generation["version"]
        # Model-format errors remain auditable student decisions. The native Pi
        # agent sees the protocol error and may use its existing finish repair.
        protocol_error = generation.get("model_protocol_error")
        message = generation.get("parsed_message", {"role": "assistant", "content": text})
        self.returned_turns += 1
        return {"id": f"opd-{self.session_id}-{turn}", "object": "chat.completion", "created": int(time.time()),
                "model": payload["model"],
                "choices": [{"index": 0, "message": message,
                             "finish_reason": finish}],
                "usage": {"prompt_tokens": len(prefix), "completion_tokens": len(ids),
                          "total_tokens": len(prefix) + len(ids)},
                "metadata": {"weight_version": version, **({"tool_protocol": PROTOCOL,
                             "raw_content": text, "enable_thinking": thinking, "protocol_error": protocol_error} if qwen else {})}}


def native_settings(config: dict, benchmark: str, *, base_url: str, run_dir: Path) -> dict:
    budgets = Path(config["safactory"]) / "env/pi_finance/benchmark_budgets.yaml"
    value = yaml.safe_load(budgets.read_text())
    result = copy.deepcopy(value["common"]["runtime"])
    for key, section in value["benchmarks"].get(benchmark, {}).get("runtime", {}).items():
        result.setdefault(key, {}).update(section)
    runtime = config["runtime"]
    native_context = int(runtime.get("native_context_window", runtime.get("context_length", 262144)))
    if not 0 < native_context <= runtime.get("context_length", 262144):
        raise ValueError("native_context_window must fit the engine context")
    for key in ["max_tool_calls", "max_runtime_seconds"]:
        if runtime.get(key) is not None:
            result["analysis"][key] = runtime[key]
    result["analysis"]["readonly_code"] = True
    result["model"].update(model="openai/finance-opd-student", base_url=base_url,
                           models_json=str(run_dir / "models.json"), temperature=1.0,
                           thinking=runtime.get("thinking", "off"), transport="safactory_qwen", max_retries=0,
                           context_window=native_context)
    result["model"]["request_timeout_seconds"] = runtime.get("request_timeout_seconds", 600)
    if runtime.get("max_new_tokens") is not None:
        result["model"]["max_tokens"] = runtime["max_new_tokens"]
        result["model"]["max_tokens_limit"] = runtime["max_new_tokens"]
    # A short explicit smoke budget must also reduce the inherited finish reserve.
    total = result["analysis"]["max_runtime_seconds"]
    result["run"]["finish_grace_seconds"] = min(result["run"].get("finish_grace_seconds", 0), total // 5)
    if runtime.get("protocol_recovery_turns") is not None:
        result["run"]["protocol_recovery_turns"] = runtime["protocol_recovery_turns"]
    from .agent_state import prepare_agent_state
    result["env"] = {"PI_CODING_AGENT_DIR": str(prepare_agent_state(config, run_dir))}
    return result


def native_node_wrapper(config: dict, directory: Path, node: str) -> str:
    exports = {"PYTHONDONTWRITEBYTECODE": "1", "JITI_FS_CACHE": "0",
               "NODE_COMPILE_CACHE": str(directory / "node-cache"),
               "PI_FINANCE_WEB_STATE_FILE": str(directory / "native/web-state.jsonl")}
    if config.get("continuation", {}).get("enabled", False):
        from .privacy import guard_environment
        # Pi's YAML env allowlist excludes custom variables and PYTHONPATH.
        # Its existing --node-bin wrapper is the supported launch boundary:
        # inject here, after Pi constructs the Node environment, not in YAML.
        exports.update(guard_environment(config, run_dir=directory))
    return ("#!/usr/bin/env bash\nset -euo pipefail\n"
            + "".join(f"export {key}={shlex.quote(value)}\n" for key, value in exports.items())
            + f"exec {shlex.quote(node)} \"$@\"\n")


async def stop_native(process):
    """Stop the owned Python process and Node/bridge descendants (separate PGIDs)."""
    children = {}
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            ppid = next(line.split()[1] for line in status.read_text().splitlines() if line.startswith("PPid:"))
            children.setdefault(int(ppid), []).append(int(status.parent.name))
        except (OSError, StopIteration):
            continue
    descendants, queue = [], list(children.get(process.pid, []))
    while queue:
        child = queue.pop()
        descendants.append(child)
        queue.extend(children.get(child, []))
    for sig in [signal.SIGTERM, signal.SIGKILL]:
        for pid in reversed([process.pid] + descendants):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            await asyncio.sleep(0.3)
    await process.wait()


async def run_native(config: dict, task: dict, proxy: EpisodeProxy) -> dict:
    directory = proxy.output
    native = directory / "native"
    settings = native_settings(config, task["benchmark"], base_url=proxy.base_url, run_dir=directory)
    write_json(directory / "models.json", {"providers": {}})
    config_path = directory / "native_config.yaml"
    config_path.write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False))
    python = config["runtime"].get("pi_python", sys.executable)
    command = [python, "-B", "-m", "pifinance.run", "run", "--config", str(config_path),
               "--benchmark", task["benchmark"], "--objective", task["task"], "--run-dir", str(native)]
    for key, flag in [("db_path", "--db-path"), ("data_path", "--data-path")]:
        if task["source"].get(key):
            command += [flag, task["source"][key]]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=config["pi_finance"],
                       PI_FINANCE_API_KEY="EMPTY", PI_FINANCE_MODELS_JSON=str(directory / "models.json"))
    if config['runtime'].get('node_bin'):
        environment['PI_NODE'] = config['runtime']['node_bin']
    # A previous evaluation shell must not replace this task's source/tools/model.
    for key in ["PI_FINANCE_CONFIG_FILE", "PI_FINANCE_MODEL", "DATAELF_FINANCE_BENCHMARK",
                "DATAELF_FINANCE_DB_PATH", "DATAELF_FINANCE_DATA_PATH", "DATAELF_FINANCE_DB_LINK",
                "DATAELF_FINANCE_TOOLS", "DATAELF_FINANCE_READONLY_CODE",
                "DATAELF_FINANCE_MAX_TOOL_CALLS", "DATAELF_FINANCE_MAX_RUNTIME_SECONDS",
                "DATAELF_FINANCE_RETRIEVE_MODEL"]:
        environment.pop(key, None)
    for key in ["NO_PROXY", "no_proxy"]:
        environment[key] = ",".join(filter(None, [environment.get(key), "127.0.0.1", "localhost"]))
    # Pi deliberately filters the child env. Its supported --node-bin interface
    # lets us disable downstream Python/Jiti disk caches without patching Pi.
    probe = await asyncio.create_subprocess_exec(
        python, "-B", "-c", "from pifinance.run import resolve_node; print(resolve_node())",
        cwd=config["pi_finance"], env=environment,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(probe.communicate(), 30)
    except BaseException:
        if probe.returncode is None:
            probe.kill()
        await probe.wait()
        raise
    if probe.returncode:
        raise RuntimeError("native Node lookup failed: " + stderr.decode(errors="replace"))
    node = stdout.decode().strip()
    wrapper = directory / "node-without-cache.sh"
    wrapper.write_text(native_node_wrapper(config, directory, node))
    wrapper.chmod(0o700)
    command += ["--node-bin", str(wrapper)]
    write_json(directory / "native_command.json", {"argv": command, "cwd": config["pi_finance"]})
    with (directory / "native.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(*command, cwd=config["pi_finance"],
                                                       env=environment, stdout=log, stderr=log,
                                                       start_new_session=True)
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), settings["analysis"]["max_runtime_seconds"] + 90)
        except asyncio.TimeoutError:
            timed_out = True
            await stop_native(process)
        except BaseException:
            await stop_native(process)
            raise
    result = json.loads((native / "result.json").read_text()) if (native / "result.json").is_file() else {}
    status = {"exit_code": process.returncode, "timed_out": timed_out,
              "native_status": result.get("status"),
              "finished": bool(result.get("finished", False)) and proxy.context_exhausted is None,
              "native_finished": result.get("finished", False),
              "generated_turns": len(proxy.records), "proxy_error": proxy.failure,
              "returned_native_turns": proxy.returned_turns,
              "cv_opd_enabled": proxy.continuation is not None,
              "continuation_probes": proxy.continuation.probes if proxy.continuation else 0,
              "cv_trainable_tokens": sum(sum(record.get("loss_mask", [])) for record in proxy.records),
              "cv_zero_loss_episode": proxy.continuation is not None and not any(
                  any(record.get("loss_mask", [])) for record in proxy.records),
              "context_truncated": proxy.context_exhausted is not None,
              "context_budget_reason": proxy.context_exhausted}
    write_json(directory / "episode_status.json", status)
    if proxy.failure:
        raise RuntimeError(f"model/scoring proxy failed; see {directory / 'proxy_error.json'}")
    if not proxy.records:
        raise RuntimeError(f"native run produced no trainable model calls; see {directory / 'native.log'}")
    # Valid student decisions survive a failed final report. Native success is never fabricated.
    return status
