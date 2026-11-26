"""Teacher-side HTTP transports.

The default "sglang" transport talks to this project's native SGLang server:
scoring posts input_ids to /generate with return_logprob and identity reads
/get_model_info. The "vllm" transport targets an externally deployed
OpenAI-compatible vLLM server: scoring uses /v1/completions with a token-id
prompt plus prompt_logprobs, generation maps SGLang sampling_params onto
completions fields, and identity compares the /v1/models served name with
config teacher.model (plus a tokenizer echo probe in scripts/verify_teacher).
Scoring uses the student's original IDs; teacher continuations require
the inference server's original sampled IDs, never re-tokenized text.
"""
from __future__ import annotations

import os
import asyncio
import re
import aiohttp

from .protocol import aligned_teacher_logps, finite, sampled_output, score_payload

TRANSPORTS = ("sglang", "vllm")


def transport_name(config: dict) -> str:
    name = config.get("teacher", {}).get("transport", "sglang")
    if name not in TRANSPORTS:
        raise ValueError(f"teacher.transport must be one of {TRANSPORTS}, got {name!r}")
    return name


def validate_teacher_config(config: dict) -> None:
    from pathlib import Path
    name = transport_name(config)
    teacher = config["teacher"]
    if not isinstance(teacher.get("url"), str) or not teacher["url"].startswith(("http://", "https://")):
        raise ValueError("teacher.url must be an http(s) URL")
    if name == "vllm":
        # vLLM reports served names, not local paths: teacher.model is the
        # exact --served-model-name the server must report via /v1/models.
        if not isinstance(teacher.get("model"), str) or not teacher["model"].strip():
            raise ValueError("vllm transport requires teacher.model to be the served model name")
        tokenizer = teacher.get("tokenizer_path")
        if not isinstance(tokenizer, str) or not tokenizer.strip():
            raise ValueError("vllm transport requires teacher.tokenizer_path for local tokenizer checks")
    elif not Path(teacher["model"]).is_dir():
        # The SGLang identity check resolves teacher.model as a local path.
        raise ValueError("sglang transport requires teacher.model to be a local model directory")


def auth_headers() -> dict | None:
    key = os.environ.get("FINANCE_OPD_TEACHER_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else None


def _post_json(client, url: str, payload: dict):
    # Imported lazily: episode imports this module at load time.
    from .episode import post_json
    return post_json(client, url, payload, headers=auth_headers())


def vllm_score_payload(prefix: list[int], output: list[int], model: str) -> dict:
    if not prefix or not output:
        raise ValueError("teacher scoring requires nonempty prefix and sampled output")
    # vLLM needs at least one sampled token; the extra decode step is ignored
    # and only the prompt (input) logprobs are consumed.
    return {"model": model, "prompt": list(prefix) + list(output),
            "max_tokens": 1, "temperature": 0.0,
            "prompt_logprobs": 0, "logprobs": 0, "echo": False}


def _logprob_value(value, position: int) -> float:
    """Accept both vLLM serializations: bare float or {logprob, rank, ...}."""
    if isinstance(value, dict):
        value = value.get("logprob")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"vLLM logprob at position {position} is not a number: {value!r}")
    return finite(value, "teacher logprob")


def _entry_logprob(entry, token: int, position: int) -> float:
    """Fetch the logprob for the exact expected token; JSON may stringify keys."""
    if not isinstance(entry, dict):
        raise ValueError(f"vLLM prompt logprob at position {position} is not a mapping")
    if token in entry:
        return _logprob_value(entry[token], position)
    if str(token) in entry:
        return _logprob_value(entry[str(token)], position)
    raise ValueError(f"vLLM prompt logprob at position {position} does not include the requested token")


def parse_vllm_prompt_logprobs(response: dict, prefix: list[int], output: list[int]) -> list[float]:
    """Read choices[0].prompt_logprobs; every position must key the exact token."""
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("vLLM scoring must return exactly one choice")
    entries = choices[0].get("prompt_logprobs")
    expected = list(prefix) + list(output)
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError("vLLM prompt_logprobs do not cover the complete requested prompt")
    if entries[0] is not None:
        raise ValueError("vLLM prompt_logprobs must start with the unconditioned None entry")
    # Verify token alignment over the whole prompt, not just the scored tail.
    for index, (entry, token) in enumerate(zip(entries[1:], expected[1:]), start=1):
        _entry_logprob(entry, token, index)
    usage = response.get("usage") or {}
    if isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] != len(expected):
        raise ValueError("vLLM scored a different prompt length than requested")
    tail = entries[-len(output):]
    return [_entry_logprob(entry, token, position)
            for position, (entry, token) in enumerate(zip(tail, output), start=len(expected) - len(output))]


def vllm_generate_payload(prefix: list[int], params: dict, model: str, *, sampled: bool) -> dict:
    payload = {"model": model, "prompt": list(prefix),
               "max_tokens": int(params["max_new_tokens"]),
               "temperature": params.get("temperature", 1.0),
               "top_p": params.get("top_p", 1.0),
               "skip_special_tokens": bool(params.get("skip_special_tokens", True))}
    if params.get("top_k") is not None:
        payload["top_k"] = params["top_k"]
    if params.get("sampling_seed") is not None:
        # SGLang's deterministic sampling_seed maps to vLLM's per-request seed.
        payload["seed"] = params["sampling_seed"]
    if sampled:
        # Require original sampled IDs. Text round trips, even with matching
        # length and similar logprobs, cannot establish token identity.
        payload["logprobs"] = 1
        payload["return_tokens_as_token_ids"] = True
        payload["skip_special_tokens"] = False
    return payload


def parse_vllm_completion(response: dict, *, prefix_len: int, max_new_tokens: int) -> dict:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("vLLM completion must return exactly one choice")
    choice = choices[0]
    finish = choice.get("finish_reason")
    if finish not in {"stop", "length"}:
        raise ValueError(f"vLLM generation did not finish normally: {finish!r}")
    text = choice.get("text")
    if not isinstance(text, str):
        raise ValueError("vLLM completion must return decoded text")
    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is not None and prompt_tokens != prefix_len:
        raise ValueError("vLLM completion prompt length differs from the complete requested prefix")
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is not None and completion_tokens > max_new_tokens:
        raise ValueError("vLLM completion exceeded the fixed generation budget")
    result = {"text": text, "finish": finish,
              "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    return result


def parse_vllm_sampled_logprobs(choice: dict) -> list[float]:
    """Sampled per-position logprobs from the OpenAI completions layout."""
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise ValueError("vLLM completion returned no sampled-token logprobs; request logprobs=1")
    values = logprobs.get("token_logprobs")
    if not isinstance(values, list) or not values:
        raise ValueError("vLLM completion returned no sampled-token logprobs; request logprobs=1")
    return [finite(value, "teacher logprob") for value in values]


def reconstruct_sampled_ids(config: dict, choice: dict, prefix: list[int]) -> tuple[list[int], str]:
    """Read server-supplied IDs (historical function name retained for callers)."""
    text = choice.get("text")
    if not isinstance(text, str):
        raise ValueError("vLLM completion must return decoded text")
    values = parse_vllm_sampled_logprobs(choice)
    tokens = choice["logprobs"].get("tokens")
    if not isinstance(tokens, list) or not all(
        isinstance(token, str) and re.fullmatch(r"token_id:[0-9]+", token) for token in tokens
    ):
        raise ValueError("vLLM must return original sampled IDs via return_tokens_as_token_ids; text reconstruction is forbidden")
    ids = [int(token.split(":", 1)[1]) for token in tokens]
    if len(ids) != len(values):
        raise ValueError("vLLM sampled logprob count differs from the original token count")
    explicit = choice.get("token_ids")
    if explicit is not None and (
        not isinstance(explicit, list) or any(type(token) is not int for token in explicit) or explicit != ids
    ):
        raise ValueError("vLLM sampled token ID fields disagree")
    return ids, text


def parse_vllm_models(response: dict) -> str:
    data = response.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("vLLM /v1/models returned no models")
    names = [item.get("id") for item in data if isinstance(item, dict)]
    if len(names) != 1 or not isinstance(names[0], str) or not names[0]:
        raise ValueError("vLLM /v1/models must report exactly one served model")
    return names[0]


async def _score_teacher_once(client, config: dict, prefix: list[int], output: list[int]) -> list[float]:
    """Conditional logprobs of the sampled student tokens under the teacher."""
    teacher = config["teacher"]
    url = teacher["url"].rstrip("/")
    if transport_name(config) == "sglang":
        response = await _post_json(client, url + "/generate", score_payload(prefix, output))
        return aligned_teacher_logps(response, prefix, output)
    response = await _post_json(client, url + "/v1/completions",
                                vllm_score_payload(prefix, output, teacher["model"]))
    return parse_vllm_prompt_logprobs(response, prefix, output)


async def score_teacher(client, config: dict, prefix: list[int], output: list[int]) -> list[float]:
    """Retry only read-only scoring of identical supplied IDs within one deadline."""
    retries = config.get("runtime", {}).get("teacher_score_retries", 0)
    if type(retries) is not int or not 0 <= retries <= 3:
        raise ValueError("teacher_score_retries must be an integer in [0, 3]")
    timeout = config.get("runtime", {}).get("request_timeout_seconds", 600)
    async with asyncio.timeout(timeout):
        for attempt in range(retries + 1):
            try:
                return await _score_teacher_once(client, config, prefix, output)
            except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, aiohttp.ClientResponseError) as error:
                if isinstance(error, aiohttp.ClientResponseError) and error.status not in {502, 503, 504}:
                    raise
                if attempt == retries:
                    raise
                await asyncio.sleep(min(2 ** attempt, 4))


async def generate_teacher(client, config: dict, prefix: list[int], params: dict,
                           *, sampled: bool = False) -> dict:
    """Teacher generation; normalized fields shared by both transports.

    sampled=True also returns ids/logps aligned to the teacher's own tokens
    (continuation turns). The SGLang branch keeps the historical response
    checks by reusing sampled_output for parsing.
    """
    teacher = config["teacher"]
    url = teacher["url"].rstrip("/")
    if transport_name(config) == "sglang":
        payload = {"input_ids": prefix, "sampling_params": dict(params), "stream": False}
        if sampled:
            payload["return_logprob"] = True
        response = await _post_json(client, url + "/generate", payload)
        if not sampled:
            meta = response.get("meta_info", {})
            return {"text": response["text"],
                    "finish": meta.get("finish_reason", {}).get("type"),
                    "prompt_tokens": meta.get("prompt_tokens"),
                    "completion_tokens": meta.get("completion_tokens", 0)}
        ids, logps, text, finish, _ = sampled_output(response)
        return {"ids": ids, "logps": logps, "text": text, "finish": finish}
    payload = vllm_generate_payload(prefix, params, teacher["model"], sampled=sampled)
    response = await _post_json(client, url + "/v1/completions", payload)
    parsed = parse_vllm_completion(response, prefix_len=len(prefix),
                                   max_new_tokens=int(params["max_new_tokens"]))
    if not sampled:
        return parsed
    choice = response["choices"][0]
    ids, raw_text = reconstruct_sampled_ids(config, choice, prefix)
    # Score these original IDs through the same channel used for distillation;
    # the comparison also catches scoring/generation distribution mismatches.
    rescored = parse_vllm_prompt_logprobs(
        await _post_json(client, url + "/v1/completions",
                         vllm_score_payload(prefix, ids, teacher["model"])), prefix, ids)
    generated = parse_vllm_sampled_logprobs(choice)
    if any(abs(a - b) > 0.05 for a, b in zip(rescored, generated)):
        raise ValueError("vLLM re-scored logprobs differ from generation-time logprobs")
    parsed.update({"ids": ids, "logps": rescored, "text": raw_text})
    return parsed
