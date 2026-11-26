"""Slime custom-generate hook; one actual Pi model call becomes one Sample."""
from __future__ import annotations

import uuid
import copy
from pathlib import Path

from .config import add_dependency_paths, read_config
from .episode import EpisodeProxy, run_native, write_json
from .protocol import assert_compatible_tokenizers, finite


_VALIDATED = set()


def align_static_samples(samples, config):
    """Pad each episode to DP alignment without changing its supervised tokens.

    Slime's bshd path uses fixed single-sequence microbatches. Episodes emit a
    variable number of turns, so an even episode count alone cannot align DP.
    A masked copy retains the original group ID and contributes no group mass.
    """
    train = config["training"]
    if train.get("qkv_format", "thd") != "bshd" or not samples:
        return samples
    parallel = (train.get("tensor_parallel_size", 1)
                * train.get("pipeline_parallel_size", 1)
                * train.get("context_parallel_size", train["gpus"]))
    dp = train["gpus"] // parallel
    if dp < 1 or train["gpus"] % parallel:
        raise ValueError("invalid training parallel topology")
    source = min(samples, key=lambda item: len(item.tokens))
    for _ in range((-len(samples)) % dp):
        padding = copy.deepcopy(source)
        padding.loss_mask = [0] * padding.response_length
        padding.reward = {"opd_weight": 0.0}
        padding.metadata.update(supervision_kind="dp_alignment_padding",
                                trainable_tokens=0, opd_weight=0.0,
                                dp_alignment_padding=True)
        samples.append(padding)
    return samples


def task_text(prompt):
    """Unwrap Slime's text-only conversation without rendering a chat template."""
    if isinstance(prompt, str):
        return prompt
    if (isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], dict)
            and set(prompt[0]) == {"role", "content"}
            and prompt[0]["role"] == "user" and isinstance(prompt[0]["content"], str)):
        return prompt[0]["content"]
    raise ValueError("expected raw task text or one plain-text user message; Pi builds the native history")


def samples_from_records(parent, records: list[dict], status: dict, directory: Path, sample_type):
    if parent.index is None:
        raise ValueError("Slime must assign an authoritative parent sample index")
    versions = {str(record.get("weight_version")) for record in records}
    if len(versions) > 1:
        raise ValueError("student weights changed inside cached episode")
    samples = []
    for record in records:
        ids = record["response_ids"]
        prefix = record["prompt_ids"]
        if not prefix or not ids or any(type(token) is not int or token < 0 for token in prefix + ids):
            raise ValueError("cached prompt and response must contain valid original token IDs")
        if record.get("teacher_log_probs") is None:
            raise ValueError("teacher logprobs were not requested for this screening context; it cannot become a training sample")
        if not (len(ids) == len(record["rollout_log_probs"]) == len(record["teacher_log_probs"])):
            raise ValueError("sampled tokens and student/teacher logprobs differ in length")
        for value in record["rollout_log_probs"] + record["teacher_log_probs"]:
            finite(value, "cached logprob")
        weight = finite(record["opd_weight"], "opd weight")
        if not 0 <= weight <= 10:
            raise ValueError("opd weight outside [0,10]")
        mask = record.get("loss_mask", [1] * len(ids))
        if len(mask) != len(ids) or any(type(item) is not int or item not in {0, 1} for item in mask):
            raise ValueError("student loss mask must align with sampled tokens and contain only 0/1")
        sample = sample_type(
            index=parent.index, group_index=parent.group_index, group_id=parent.index,
            prompt=parent.prompt, tokens=record["prompt_ids"] + ids,
            response=record["response"], response_length=len(ids), loss_mask=list(mask),
            rollout_log_probs=record["rollout_log_probs"], teacher_log_probs=record["teacher_log_probs"],
            reward={"opd_weight": record["opd_weight"]},
            status=sample_type.Status.TRUNCATED if record["finish_reason"] == "length" else sample_type.Status.COMPLETED,
            session_id=parent.session_id,
            metadata={**parent.metadata, "turn": record["turn"], "episode": str(directory),
                      "native_status": status, "student_prefix_sha256": record["student_prefix_sha256"],
                      "supervision": "student_sampled_token_opd", "native_task_reward": None, "model_protocol_error": record.get("model_protocol_error")},
        )
        sample.weight_versions = [] if record["weight_version"] is None else [str(record["weight_version"])]
        sample.metadata.update(supervision_kind=record.get("supervision_kind", "plain_opd"),
                               continuation_audit=record.get("context_audit", {}),
                               opd_weight=record["opd_weight"], trainable_tokens=sum(mask))
        samples.append(sample)
    return samples


async def generate(args, sample, sampling_params, evaluation=False):
    if evaluation:
        raise ValueError("use native eval_finance evaluation separately; OPD weights are not task scores")
    if getattr(args, "apply_chat_template", False):
        raise ValueError("do not use --apply-chat-template; Pi constructs the native prompt")
    config = read_config()
    add_dependency_paths(config)
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.types import Sample
    from rl.examples.pi_finance.qwen_mask_builder import QwenFinanceMaskBuilder

    if config["config_path"] not in _VALIDATED:
        # vLLM teachers expose a served name, not a local path; tokenizer
        # compatibility is checked against the local snapshot.
        teacher_tokenizer = config["teacher"].get("tokenizer_path") or config["teacher"]["model"]
        assert_compatible_tokenizers(args.hf_checkpoint, teacher_tokenizer)
        from rl.examples.pi_finance.qwen_protocol import validate_native_parser
        import importlib.metadata
        import os
        validate_native_parser()
        write_json(Path(config["output"]) / f"protocol_runtime.{os.getpid()}.json", {
            "tool_protocol": "qwen3_xml_v1", "sglang_parser": "qwen3_coder",
            "sglang_version": importlib.metadata.version("sglang"),
            "native_parser_verified": True, "enable_thinking": config["runtime"].get("thinking", "off") != "off",
            "student_checkpoint": str(args.hf_checkpoint), "teacher_scoring": "exact_sampled_token_ids"})
        _VALIDATED.add(config["config_path"])
    if set(sample.metadata) - {"id", "benchmark", "native_case_id", "entity_id", "source", "split_group", "sampling_params"}:
        raise ValueError("unexpected task metadata; prepare a task-only dataset with finance_opd.data")
    task = {**sample.metadata, "task": task_text(sample.prompt)}
    sample.session_id = sample.session_id or str(uuid.uuid4())
    directory = Path(config["output"]) / "episodes" / sample.session_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "task.json", task)
    state = GenerateState(args)
    builder = QwenFinanceMaskBuilder(state.tokenizer, processor=None, enable_thinking=config["runtime"].get("thinking", "off") != "off")
    student_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    if config["runtime"].get("episode_worker_queue"):
        from .episode_queue import queued_episode
        status, records = await queued_episode(config, task, student_url, directory,
                                               sample.session_id, sampling_params)
    else:
        async with EpisodeProxy(builder=builder, task=task, student_url=student_url,
                                config=config, output=directory, session_id=sample.session_id,
                                sampling_params=sampling_params) as proxy:
            status = await run_native(config, task, proxy)
            records = proxy.records
    if config.get("continuation", {}).get("enabled", False):
        # Keep original mining calls on disk; train only actual student resamples.
        selected = [record for record in records if any(record.get("loss_mask", []))]
        records = selected or records[-1:]
        status["cv_zero_loss_episode"] = not selected
    result = samples_from_records(sample, records, status, directory, Sample)
    return align_static_samples(result, config)
