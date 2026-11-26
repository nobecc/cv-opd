"""Strict sampled-token scoring; no model/framework dependencies."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from pathlib import Path


def finite(value, name="logprob") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def sampled_output(response: dict) -> tuple[list[int], list[float], str, str, object]:
    meta = response["meta_info"]
    entries = meta["output_token_logprobs"]
    if not entries:
        raise ValueError("student engine returned no sampled tokens")
    ids, logps = [], []
    for entry in entries:
        if len(entry) < 2 or type(entry[1]) is not int or entry[1] < 0:
            raise ValueError("invalid sampled token id")
        ids.append(entry[1])
        logps.append(finite(entry[0]))
    if "output_ids" in response and response["output_ids"] != ids:
        raise ValueError("output_ids differ from authoritative engine logprob token ids")
    finish = meta.get("finish_reason", {}).get("type")
    if finish not in {"stop", "length"}:
        raise ValueError(f"student generation failed: {finish}")
    if not isinstance(response.get("text"), str):
        raise ValueError("student engine must return decoded response text")
    return ids, logps, response["text"], finish, meta.get("weight_version")


def score_payload(prefix: list[int], output: list[int]) -> dict:
    if not prefix or not output:
        raise ValueError("teacher scoring requires nonempty prefix and sampled output")
    return {"input_ids": prefix + output,
            "sampling_params": {"temperature": 1, "max_new_tokens": 0, "skip_special_tokens": False},
            "return_logprob": True, "logprob_start_len": 0}


def aligned_teacher_logps(response: dict, prefix: list[int], output: list[int]) -> list[float]:
    entries = response["meta_info"]["input_token_logprobs"]
    # SGLang versions may include or omit the first unconditioned token placeholder.
    if len(entries) not in {len(prefix) + len(output), len(prefix) + len(output) - 1}:
        raise ValueError("teacher returned an incomplete input-logprob sequence")
    expected = prefix + output
    if len(entries) == len(expected) - 1:
        expected = expected[1:]
    if any(len(e) < 2 or type(e[1]) is not int or e[1] != token for e, token in zip(entries, expected)):
        raise ValueError("teacher logprobs are not aligned to the requested conditioning prefix and sampled tokens")
    tail = entries[-len(output):]
    if len(tail) != len(output) or any(len(e) < 2 or e[1] != token for e, token in zip(tail, output)):
        raise ValueError("teacher logprobs are not aligned to the sampled student token ids")
    return [finite(entry[0], "teacher logprob") for entry in tail]


def assert_compatible_tokenizers(student: str | Path, teacher: str | Path) -> None:
    a = json.loads((Path(student) / "tokenizer.json").read_text())
    b = json.loads((Path(teacher) / "tokenizer.json").read_text())
    # Tokenizers serializes BPE merges as either pairs or space-joined strings.
    # Compare ordered pairs while preserving every vocabulary and pipeline field.
    for tokenizer in (a, b):
        model = tokenizer.get("model", {})
        if model.get("type") == "BPE" and isinstance(model.get("merges"), list):
            model["merges"] = [
                merge.split(" ") if isinstance(merge, str) else merge
                for merge in model["merges"]
            ]
            if any(not isinstance(pair, list) or len(pair) != 2 for pair in model["merges"]):
                raise ValueError("invalid BPE merge pair")
    for key in ["model", "normalizer", "pre_tokenizer", "post_processor", "decoder"]:
        if a.get(key) != b.get(key):
            raise ValueError(f"incompatible tokenizer component: {key}")
    tokens = {t["id"]: t for t in b.get("added_tokens", [])}
    if any(tokens.get(t["id"]) != t for t in a.get("added_tokens", [])):
        raise ValueError("teacher does not preserve all student added-token mappings")


def freeze_teacher_view(hook: str | None, messages: list[dict], task: dict, turn: int) -> dict:
    """Called BEFORE sampling. A hook receives only the current public history/task."""
    request = {"messages": copy.deepcopy(messages), "task": copy.deepcopy(task), "turn": turn}
    prefix_hash = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    value = {"messages": messages, "weight": 1.0}
    if hook:
        module, name = hook.rsplit(".", 1)
        value = getattr(importlib.import_module(module), name)(request)
    if not isinstance(value, dict) or set(value) - {"messages", "weight", "audit"}:
        raise ValueError("context hook must return messages, weight, and optional audit")
    history = value.get("messages")
    if not isinstance(history, list) or not history or any(
        not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant", "tool"}
        or not isinstance(m.get("content"), str) for m in history
    ):
        raise ValueError("teacher context must contain text-only wire messages")
    weight = finite(value.get("weight", 1.0), "opd weight")
    if not 0 <= weight <= 10:
        raise ValueError("opd weight must be in [0,10]")
    # Copy again so hooks cannot mutate frozen supervision after student generation.
    return copy.deepcopy({"messages": history, "weight": weight,
                          "audit": value.get("audit", {}), "student_prefix_sha256": prefix_hash})
