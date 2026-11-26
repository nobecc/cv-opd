"""Bounded teacher/student trial workers; writes only into the standalone run."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from .checkpoints import prune_checkpoints
from .config import ROOT, read_config
from .launch import environment, teacher_arguments, training_topology
from .protocol import aligned_teacher_logps, score_payload


def write(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def remaining_seconds(config, default=10800):
    deadline = config.get("job", {}).get("deadline_epoch")
    if deadline is None:
        return default
    seconds = float(deadline) - time.time()
    if not math.isfinite(seconds) or seconds <= 0:
        raise TimeoutError("configured job deadline has expired")
    return seconds


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


def wait_file(path, failure, timeout, process=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return json.loads(path.read_text())
        if failure.is_file():
            raise RuntimeError(f"peer failed: {failure}")
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"owned process exited with {process.returncode}")
        time.sleep(5)
    raise TimeoutError(f"timed out waiting for {path}")


def student_assignment(config):
    """Return this replica's rank and local GPU count; reject split-brain launches."""
    topology = training_topology(config["training"])
    nodes = topology["nodes"]
    if nodes == 1:
        if int(os.environ.get("NODE_RANK", "0")) != 0:
            raise ValueError("single-node student must use NODE_RANK=0")
        return 0, topology["gpus_per_node"]
    if os.environ.get("DISTRIBUTED_JOB", "").lower() != "true":
        raise ValueError("multi-node student requires DISTRIBUTED_JOB=true")
    rank = int(os.environ.get("NODE_RANK", "0"))
    if int(os.environ.get("NODE_COUNT", "0")) != nodes or not 0 <= rank < nodes:
        raise ValueError("NODE_RANK/NODE_COUNT do not match training.nodes")
    if not os.environ.get("MASTER_ADDR"):
        raise ValueError("multi-node student requires MASTER_ADDR")
    if not 1 <= int(os.environ.get("MASTER_PORT", "0")) <= 65535:
        raise ValueError("multi-node student requires a valid MASTER_PORT")
    injected_gpus = os.environ.get("PROC_PER_NODE")
    if injected_gpus is not None and int(injected_gpus) != topology["gpus_per_node"]:
        raise ValueError("PROC_PER_NODE does not match training.gpus_per_node")
    return rank, topology["gpus_per_node"]


def inventory(role, directory, expected_gpus, artifact=None, node_rank=0):
    import torch
    result = {"role": role, "node_rank": node_rank, "gpu_count": torch.cuda.device_count(),
              "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
              "versions": {name: importlib.metadata.version(name)
                           for name in ["torch", "sglang", "transformers"]}}
    write(directory / f"{artifact or role}_inventory.json", result)
    if result["gpu_count"] != expected_gpus:
        raise RuntimeError(f"expected {expected_gpus} GPUs: {result}")


def teacher_model_support(config, registry):
    """Check the installed implementation, including model-specific backports."""
    model_config = json.loads((Path(config["teacher"]["model"]) / "config.json").read_text())
    architectures = model_config.get("architectures", [])
    if not architectures or not any(registry.is_arch_supported(arch) for arch in architectures):
        raise RuntimeError(f"SGLang has no native implementation for {architectures}")
    model_class, resolved = registry.resolve_model_cls(architectures)
    if resolved not in architectures:
        raise RuntimeError(f"SGLang resolved {architectures} to an unexpected fallback: {resolved}")
    return {"architectures": architectures, "resolved": resolved,
            "implementation": model_class.__module__ + "." + model_class.__name__,
            "sglang_version": importlib.metadata.version("sglang")}


def probe_teacher_context(config, url, prefix, output):
    """Exercise the actual model at the configured shared token ceiling."""
    runtime = config["runtime"]
    student_limit = int(runtime.get("context_length", 262144))
    teacher_limit = int(config["teacher"].get("context_length", student_limit))
    margin = int(runtime.get("context_safety_margin", 8))
    ceiling = min(student_limit, teacher_limit) - margin
    length = ceiling - len(output)
    if length < 1:
        raise ValueError("teacher context probe has no room for a conditioning prefix")
    repeated = (prefix * ((length + len(prefix) - 1) // len(prefix)))[:length]
    response = request(url + "/generate", score_payload(repeated, output), timeout=180)
    logps = aligned_teacher_logps(response, repeated, output)
    return {"real_model": True, "input_tokens": ceiling, "safety_margin": margin,
            "teacher_log_probs": logps, "token_alignment_verified": True,
            "qwen_training_executed": False}


def teacher(config, directory):
    from sglang.srt.models.registry import ModelRegistry
    write(directory / "teacher_model_support.json", teacher_model_support(config, ModelRegistry))
    command = teacher_arguments(config)
    write(directory / "teacher_command.json", command)
    process = None
    try:
        with (directory / "teacher.log").open("x") as log:
            process = subprocess.Popen(command, env=environment(config), stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            url = config["teacher"]["url"]
            deadline = time.monotonic() + 2400
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"teacher startup exited {process.returncode}; see teacher.log")
                try:
                    info = request(url + "/get_model_info", timeout=5)
                    if Path(info["model_path"]).resolve() != Path(config["teacher"]["model"]).resolve():
                        raise ValueError("teacher model identity mismatch")
                    break
                except (OSError, KeyError, ValueError):
                    time.sleep(5)
            else:
                raise TimeoutError("teacher startup exceeded 40 minutes")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config["student_model"], local_files_only=True)
            prefix = tokenizer.encode("<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n")
            output = tokenizer.encode("2.")
            response = request(url + "/generate", score_payload(prefix, output), timeout=180)
            logps = aligned_teacher_logps(response, prefix, output)
            if any(x > 1e-5 for x in logps):
                raise ValueError("teacher returned positive log probabilities")
            write(directory / "teacher_probe.json", {"prefix": prefix, "output": output,
                  "teacher_log_probs": logps, "raw_response": response, "real_model": True})
            if config["runtime"].get("probe_context_boundary", False):
                write(directory / "teacher_context_probe.json",
                      probe_teacher_context(config, url, prefix, output))
            connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            connection.connect(("8.8.8.8", 80))
            host = connection.getsockname()[0]
            connection.close()
            port = urllib.parse.urlsplit(url).port
            write(directory / "teacher_ready.json", {"url": f"http://{host}:{port}",
                  "model": config["teacher"]["model"], "scoring_verified": True, "time": time.time()})
            # Release only this worker's teacher after the paired trial finishes.
            wait_file(directory / "student_exit.json", directory / "student_error.json", remaining_seconds(config), process)
    finally:
        stop_owned(process)


def multinode_ray_worker(config, directory, node_rank):
    """Nonzero ranks only join Ray; rank zero alone writes training results."""
    artifact = f"student_rank{node_rank}"
    env = environment(config)
    env["PI_NODE"] = str(ROOT / ".runtime/node24")
    command = [sys.executable, "-B", "-m", "finance_opd.multinode_driver", "worker",
               "--config", str(config["config_path"]), "--run-dir", str(directory)]
    process = None
    try:
        with (directory / f"{artifact}.log").open("x") as log:
            process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=remaining_seconds(config))
        if code:
            raise RuntimeError(f"Ray worker exited {code}; see {artifact}.log")
        write(directory / f"{artifact}_result.json", {"process_exit_code": code,
              "node_rank": node_rank, "role": "ray_worker"})
    finally:
        stop_owned(process)


def student(config, directory, node_rank=0):
    if node_rank:
        multinode_ray_worker(config, directory, node_rank)
        return
    topology = training_topology(config["training"])
    local_gpus = topology["gpus_per_node"]
    ready = wait_file(directory / "teacher_ready.json", directory / "teacher_error.json", 5400)
    if ready.get("model") != config["teacher"]["model"] or ready.get("scoring_verified") is not True:
        raise ValueError("invalid teacher readiness record")
    config["teacher"]["url"] = ready["url"]
    config["runtime"]["pi_python"] = sys.executable
    path = directory / "student_config.json"
    write(path, config)
    env = environment(config)
    env["PI_NODE"] = str(ROOT / ".runtime/node24")
    for key in ["NO_PROXY", "no_proxy"]:
        env[key] += "," + urllib.parse.urlsplit(ready["url"]).hostname
    fraction = float(config["training"].get("optimizer_offload_fraction", 1.0))
    if fraction < 1:
        # Exercise the image's actual CPU/GPU Adam implementations before
        # spending minutes loading the full student and generating a rollout.
        with (directory / "optimizer_preflight.log").open("x") as log:
            subprocess.run([sys.executable, "-B", str(ROOT / "scripts/probe_optimizer.py"),
                            "--output", str(directory / "optimizer_preflight.json"),
                            "--fraction", str(fraction)], env=env, cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    if config["training"].get("checkpoint_format", "torch_dist") == "torch":
        with (directory / "parallel_preflight.log").open("x") as log:
            subprocess.run([sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
                            "--nproc_per_node", str(local_gpus),
                            str(ROOT / "scripts/probe_parallel.py"),
                            "--tensor-parallel-size", str(config["training"].get("tensor_parallel_size", 1)),
                            "--output", str(directory / "parallel_preflight.json")],
                           env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    if config["training"].get("log_probs_chunk_size"):
        model_config = json.loads((Path(config["student_model"]) / "config.json").read_text())
        vocab = model_config.get("text_config", model_config)["vocab_size"]
        with (directory / "logprob_preflight.log").open("x") as log:
            subprocess.run([sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
                            "--nproc_per_node", str(local_gpus),
                            str(ROOT / "scripts/probe_logprobs.py"), "--output",
                            str(directory / "logprob_preflight.json"), "--sequence-length",
                            str(config["runtime"]["context_length"]), "--vocab-size", str(vocab),
                            "--chunk-size", str(config["training"]["log_probs_chunk_size"]),
                            "--tensor-parallel-size", str(config["training"].get("tensor_parallel_size", 1))],
                           env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=360)
    process = None
    module = "finance_opd.multinode_driver" if topology["nodes"] > 1 else "finance_opd.launch"
    command = [sys.executable, "-B", "-m", module]
    if topology["nodes"] > 1:
        command.append("driver")
    else:
        command.append("train")
    command += ["--config", str(path)]
    if topology["nodes"] > 1:
        command += ["--run-dir", str(directory)]
    try:
        with (directory / "student.log").open("x") as log:
            process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + remaining_seconds(config)
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("student reached configured job deadline")
                try:
                    process.wait(timeout=min(30, max(0.1, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
                prune_checkpoints(config)
            code = process.returncode
            if code:
                raise RuntimeError(f"student exited {code}; see student.log")
            checkpoints = Path(config["output"]) / "checkpoints"
            latest = checkpoints / "latest_checkpointed_iteration.txt"
            if not latest.is_file() or int(latest.read_text().strip()) < config["training"].get("updates", 1) - 1:
                raise RuntimeError("training exited without the final zero-based rollout checkpoint")
            if config["training"].get("checkpoint_format", "torch_dist") == "torch":
                iteration = int(latest.read_text().strip())
                tp = config["training"].get("tensor_parallel_size", 1)
                pp = config["training"].get("pipeline_parallel_size", 1)
                expected = [checkpoints / f"iter_{iteration:07d}" /
                            (f"mp_rank_{t:02d}" if pp == 1 else f"mp_rank_{t:02d}_{p:03d}") /
                            "model_optim_rng.pt" for t in range(tp) for p in range(pp)]
                if not all(p.is_file() and p.stat().st_size > 0 for p in expected):
                    raise RuntimeError("training exited with incomplete tensor/pipeline checkpoint shards")
            write(directory / "student_result.json", {"process_exit_code": code,
                  "latest_checkpoint_iteration": int(latest.read_text().strip()),
                  "checkpointed_rollouts": int(latest.read_text().strip()) + 1,
                  "checkpoint_directory": str(checkpoints),
                  "episode_count": len(list((Path(config["output"]) / "episodes").glob("*/episode_status.json"))),
                  "parameter_change_independently_verified": False})
    finally:
        stop_owned(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["teacher", "student"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(ROOT / "runs") or not directory.is_dir():
        raise ValueError("run directory must already exist under cv-opd/runs")
    config = read_config(args.config)
    node_rank, student_gpus = student_assignment(config) if args.role == "student" else (0, 0)
    expected_gpus = student_gpus if args.role == "student" else config["teacher"].get("tensor_parallel_size", 4)
    artifact = args.role if node_rank == 0 else f"{args.role}_rank{node_rank}"
    code = 1
    try:
        inventory(args.role, directory, expected_gpus, artifact, node_rank)
        if args.role == "teacher":
            teacher(config, directory)
        else:
            student(config, directory, node_rank)
        code = 0
    except BaseException as error:
        write(directory / f"{artifact}_error.json", {"type": type(error).__name__,
              "node_rank": node_rank, "error": str(error)})
        raise
    finally:
        write(directory / f"{artifact}_exit.json", {"exit_code": code,
              "node_rank": node_rank, "time": time.time()})


if __name__ == "__main__":
    main()
