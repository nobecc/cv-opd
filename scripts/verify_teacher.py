"""Verify an externally deployed teacher and record trial readiness.

Transport-aware replacement for the self-hosted worker's readiness probes:
identity, a real input-token scoring probe, the configured context boundary
and (CV mode) one sampled teacher generation. Writes teacher_ready.json only
after every selected probe succeeds; watch_trial and the student submission
consume the same files the self-hosted worker produces.
"""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request

from finance_opd.config import ROOT, read_config
from finance_opd.protocol import finite
from finance_opd.teacher_transport import (parse_vllm_models, parse_vllm_prompt_logprobs,
                                           parse_vllm_sampled_logprobs, reconstruct_sampled_ids,
                                           transport_name, validate_teacher_config,
                                           vllm_generate_payload, vllm_score_payload)
from finance_opd.protocol import aligned_teacher_logps, score_payload


def request(url, payload=None, timeout=60):
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("FINANCE_OPD_TEACHER_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def probe_identity(config, url):
    transport = transport_name(config)
    if transport == "vllm":
        served = parse_vllm_models(request(url + "/v1/models", timeout=10))
        if served != config["teacher"]["model"]:
            raise ValueError(f"served model {served!r} does not match configured teacher.model")
        return {"transport": transport, "served_name": served}
    info = request(url + "/get_model_info", timeout=10)
    if Path(info["model_path"]).resolve() != Path(config["teacher"]["model"]).resolve():
        raise ValueError("teacher model identity mismatch")
    return {"transport": transport, "model_path": info["model_path"]}


def probe_echo_tokenizer(config, url, tokenizer):
    """vLLM reports no weight path; echo-decode token ids through the server."""
    ids = tokenizer.encode("What is 1+1? The answer is two.")
    payload = {"model": config["teacher"]["model"], "prompt": ids, "max_tokens": 1,
               "temperature": 0.0, "echo": True, "skip_special_tokens": False}
    text = request(url + "/v1/completions", payload)["choices"][0]["text"]
    expected = tokenizer.decode(ids)
    if not isinstance(text, str) or not text.startswith(expected):
        raise ValueError("teacher did not echo-decode the tokenizer probe ids; "
                         "served tokenizer differs from teacher.tokenizer_path")
    return {"probe_tokens": len(ids), "echoed_prefix_matches": True}


def score_once(config, url, prefix, output, timeout=180):
    if transport_name(config) == "vllm":
        response = request(url + "/v1/completions",
                           vllm_score_payload(prefix, output, config["teacher"]["model"]), timeout)
        return parse_vllm_prompt_logprobs(response, prefix, output), response
    response = request(url + "/generate", score_payload(prefix, output), timeout)
    return aligned_teacher_logps(response, prefix, output), response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--skip-context-probe", action="store_true",
                        help="omit the long-prefix boundary probe (not the scoring probe)")
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(ROOT / "runs"):
        raise ValueError("use a standalone run directory")
    config = read_config(directory / "config.json")
    validate_teacher_config(config)
    url = config["teacher"]["url"].rstrip("/")
    transport = transport_name(config)

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("verify_teacher needs transformers to load the probe tokenizer") from error
    tokenizer_path = config["teacher"].get("tokenizer_path") or config["student_model"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    identity = probe_identity(config, url)
    echo = probe_echo_tokenizer(config, url, tokenizer) if transport == "vllm" else {"skipped": True}

    prefix = tokenizer.encode("<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n")
    output = tokenizer.encode("2.")
    logps, raw = score_once(config, url, prefix, output)
    if any(value > 1e-5 for value in logps):
        raise ValueError("teacher returned positive log probabilities")
    write(directory / "teacher_probe.json",
          {"prefix": prefix, "output": output, "teacher_log_probs": logps,
           "real_model": True, "transport": transport, "raw_response": raw})

    if config["runtime"].get("probe_context_boundary", False) and not args.skip_context_probe:
        runtime = config["runtime"]
        student_limit = int(runtime.get("context_length", 262144))
        teacher_limit = int(config["teacher"].get("context_length", student_limit))
        margin = int(runtime.get("context_safety_margin", 8))
        ceiling = min(student_limit, teacher_limit) - margin
        unit = tokenizer.encode("context ")
        repeated = (unit * ((ceiling - len(output) + len(unit) - 1) // len(unit)))[:ceiling - len(output)]
        boundary, boundary_raw = score_once(config, url, repeated, output, timeout=300)
        write(directory / "teacher_context_probe.json",
              {"real_model": True, "transport": transport, "input_tokens": ceiling,
               "safety_margin": margin, "teacher_log_probs": boundary,
               "token_alignment_verified": True, "raw_response": boundary_raw})

    sampled = {"skipped": "continuation disabled"}
    if config.get("continuation", {}).get("enabled", False):
        if transport == "vllm":
            params = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 8,
                      "skip_special_tokens": True}
            payload = vllm_generate_payload(prefix, params, config["teacher"]["model"],
                                            sampled=True)
            response = request(url + "/v1/completions", payload, timeout=120)
            choice = response["choices"][0]
            ids, clean_text = reconstruct_sampled_ids(config, choice, prefix)
            rescored = parse_vllm_prompt_logprobs(
                request(url + "/v1/completions",
                        vllm_score_payload(prefix, ids, config["teacher"]["model"]), timeout=180),
                prefix, ids)
            generated = parse_vllm_sampled_logprobs(choice)
            if any(abs(a - b) > 0.05 for a, b in zip(rescored, generated)):
                raise ValueError("re-scored logprobs differ from generation-time logprobs")
            sampled = {"probe": "sampled_generation", "tokens": len(ids),
                       "finish": choice.get("finish_reason"), "logprob_crosscheck": "passed",
                       "text": clean_text[:80]}
        else:
            sampled = {"probe": "sampled_generation", "skipped":
                       "sglang continuation parsing is already covered by the self-hosted worker"}

    write(directory / "teacher_ready.json",
          {"url": config["teacher"]["url"], "model": config["teacher"]["model"],
           "scoring_verified": True, "time": time.time(), "transport": transport,
           "identity": identity, "echo_tokenizer": echo, "sampled_generation": sampled})
    print(json.dumps({"teacher_ready": True, "transport": transport, "url": config["teacher"]["url"],
                      "identity": identity, "echo_tokenizer": echo, "sampled_generation": sampled,
                      "scoring_log_probs": [finite(value) for value in logps]}))


if __name__ == "__main__":
    main()
