"""Serve one OpenAI-compatible judge model for the native evaluator.

Standalone companion to the trial workers: no training dependencies, one
judge per run directory, readiness only after a real chat completion. The
evaluator reaches the judge through the pod IP recorded in judge_ready.json.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def request(url, payload=None, timeout=30):
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def model_support(model):
    """Fail fast when the installed SGLang has no implementation for the arch."""
    from sglang.srt.models.registry import ModelRegistry
    import importlib.metadata
    architectures = json.loads((Path(model) / "config.json").read_text()).get("architectures", [])
    if not architectures or not any(ModelRegistry.is_arch_supported(arch) for arch in architectures):
        raise RuntimeError(f"SGLang has no native implementation for {architectures}")
    model_class, resolved = ModelRegistry.resolve_model_cls(architectures)
    if resolved not in architectures:
        raise RuntimeError(f"SGLang resolved {architectures} to an unexpected fallback: {resolved}")
    return {"architectures": architectures, "resolved": resolved,
            "implementation": model_class.__module__ + "." + model_class.__name__,
            "sglang_version": importlib.metadata.version("sglang")}


def pod_ip():
    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    connection.connect(("8.8.8.8", 80))
    host = connection.getsockname()[0]
    connection.close()
    return host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model snapshot path")
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--port", type=int, default=23455)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--extra-arg", action="append", default=[])
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    print({"argv": sys.argv, "run_dir_resolved": str(directory),
           "is_dir": directory.is_dir(), "root": str(ROOT)}, flush=True)
    if not directory.is_relative_to(ROOT / "runs") or not directory.is_dir():
        raise ValueError("run directory must already exist under cv-opd/runs")
    write(directory / "judge_model_support.json", model_support(args.model))
    command = [sys.executable, "-m", "sglang.launch_server",
               "--model-path", args.model, "--tokenizer-path", args.model,
               "--tp", str(args.tp), "--host", "0.0.0.0", "--port", str(args.port),
               "--context-length", str(args.context_length),
               "--mem-fraction-static", str(args.mem_fraction),
               "--served-model-name", args.served_name,
               "--max-running-requests", str(args.max_running_requests),
               "--dtype", "bfloat16"] + args.extra_arg
    write(directory / "judge_command.json", command)
    process = None
    try:
        # Tolerate resubmissions into the same run directory; earlier
        # attempts keep their own log for auditing.
        attempt = 1
        log_path = directory / "judge.log"
        while log_path.exists():
            attempt += 1
            log_path = directory / f"judge-attempt{attempt}.log"
        with log_path.open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            url = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + 2400
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"judge startup exited {process.returncode}; see judge.log")
                try:
                    info = request(url + "/get_model_info", timeout=5)
                    if Path(info["model_path"]).resolve() != Path(args.model).resolve():
                        raise ValueError("judge model identity mismatch")
                    break
                except (OSError, KeyError, ValueError):
                    time.sleep(5)
            else:
                raise TimeoutError("judge startup exceeded 40 minutes")
            completion = request(url + "/v1/chat/completions", {
                "model": args.served_name, "max_tokens": 8, "temperature": 0,
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}]}, timeout=120)
            content = completion["choices"][0]["message"].get("content")
            if not content or not str(content).strip():
                raise ValueError("judge returned an empty completion")
            write(directory / "judge_probe.json", {"real_model": True,
                  "served_name": args.served_name, "completion": completion})
            write(directory / "judge_ready.json", {"url": f"http://{pod_ip()}:{args.port}",
                  "model": args.model, "served_name": args.served_name,
                  "scoring_verified": True, "time": time.time()})
            # A frozen server must fail the job visibly instead of hanging
            # silently while training episodes time out their probes.
            # /health returns 200 with an empty body: judge readiness by
            # status code only, never by parsing the response.
            misses = 0
            while process.poll() is None:
                time.sleep(60)
                try:
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(url + "/health", timeout=20) as response:
                        healthy = 200 <= response.status < 300
                except OSError:
                    healthy = False
                misses = 0 if healthy else misses + 1
                if misses >= 3:
                    raise RuntimeError("judge /health unresponsive for 3 consecutive minutes")
            sys.exit(process.returncode)
    finally:
        stop_owned(process)


if __name__ == "__main__":
    main()
