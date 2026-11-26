"""Ray head/job-submission driver and blocking workers for one multi-node student RJob."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

from .config import ROOT, add_dependency_paths, read_config
from .launch import environment, prepare_training, propagated_environment, training_topology


def validate_assignment(config: dict, mode: str) -> tuple[dict, int]:
    topology = training_topology(config["training"])
    if topology["nodes"] < 2:
        raise ValueError("multinode_driver requires training.nodes >= 2")
    if os.environ.get("DISTRIBUTED_JOB", "").lower() != "true":
        raise ValueError("multi-node student requires DISTRIBUTED_JOB=true")
    rank = int(os.environ.get("NODE_RANK", "0"))
    node_count = int(os.environ.get("NODE_COUNT", "0"))
    if node_count != topology["nodes"] or not 0 <= rank < node_count:
        raise ValueError("NODE_RANK/NODE_COUNT do not match training.nodes")
    if (mode == "driver") != (rank == 0):
        raise ValueError("only NODE_RANK=0 runs the driver; every other rank is a Ray worker")
    if not os.environ.get("MASTER_ADDR"):
        raise ValueError("multi-node student requires the RJob-injected MASTER_ADDR")
    master_port = int(os.environ.get("MASTER_PORT", "0"))
    if not 1 <= master_port <= 65535:
        raise ValueError("MASTER_PORT must be an integer in 1..65535")
    injected_gpus = os.environ.get("PROC_PER_NODE")
    if injected_gpus is not None and int(injected_gpus) != topology["gpus_per_node"]:
        raise ValueError("PROC_PER_NODE does not match training.gpus_per_node")
    return topology, rank


def add_master_no_proxy(env: dict) -> None:
    master = os.environ["MASTER_ADDR"]
    for key in ("NO_PROXY", "no_proxy"):
        values = [value for value in [env.get(key), "localhost", "127.0.0.1", master] if value]
        env[key] = ",".join(dict.fromkeys(values))


def driver(config: dict) -> None:
    """Prepare one output, start the Ray head, and submit the Slime job through upstream utility."""
    command, env = prepare_training(config)
    env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
    env["MASTER_PORT"] = os.environ["MASTER_PORT"]
    add_master_no_proxy(env)
    os.environ.update(env)
    os.environ["SLIME_SCRIPT_EXTERNAL_RAY"] = "0"
    os.environ["SLIME_SCRIPT_ENABLE_RAY_SUBMIT"] = "1"
    os.chdir(ROOT)
    add_dependency_paths(config)
    from slime.utils.external_utils.command_utils import ExecuteTrainConfig, execute_train

    topology = training_topology(config["training"])
    execute_train(
        train_args=shlex.join(command[2:]),
        num_gpus_per_node=topology["gpus_per_node"],
        megatron_model_type=None,  # training_arguments already contains MODEL_ARGS.
        train_script=str(Path(config["slime"]) / "train.py"),
        extra_env_vars=propagated_environment(config, env),
        config=ExecuteTrainConfig(num_nodes=topology["nodes"]),
    )


def local_ip(master: str) -> str:
    for key in ("LOCAL_IP", "POD_IP"):
        if os.environ.get(key):
            return os.environ[key]
    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        connection.connect((master, 6379))
        return connection.getsockname()[0]
    finally:
        connection.close()


def ray_worker_command(config: dict) -> list[str]:
    topology = training_topology(config["training"])
    master = os.environ["MASTER_ADDR"]
    return ["ray", "start", f"--address={master}:6379",
            "--num-gpus", str(topology["gpus_per_node"]),
            "--node-ip-address", local_ip(master), "--disable-usage-stats", "--block"]


def stop_owned(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def head_completed(directory: Path) -> bool:
    error = directory / "student_error.json"
    if error.is_file():
        raise RuntimeError(f"rank-zero driver failed: {error}")
    exit_path = directory / "student_exit.json"
    if not exit_path.is_file():
        return False
    result = json.loads(exit_path.read_text())
    if result.get("exit_code") != 0:
        raise RuntimeError(f"rank-zero driver exited unsuccessfully: {result}")
    return True


def worker(config: dict, directory: Path) -> None:
    """Join the head with --block, retrying only until the head is available."""
    env = environment(config)
    add_master_no_proxy(env)
    failures = 0
    while not head_completed(directory):
        process = subprocess.Popen(ray_worker_command(config), env=env, start_new_session=True)
        try:
            while process.poll() is None:
                if head_completed(directory):
                    stop_owned(process)
                    return
                time.sleep(5)
            if head_completed(directory):
                return
            failures += 1
            if failures >= 60:
                raise RuntimeError(f"Ray worker exited {process.returncode} 60 times before head completion")
            print(f"Ray worker exited {process.returncode}; retrying head connection", flush=True)
        finally:
            stop_owned(process)
        time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["driver", "worker"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(ROOT / "runs") or not directory.is_dir():
        raise ValueError("run directory must already exist under cv-opd/runs")
    config = read_config(args.config)
    validate_assignment(config, args.mode)
    if args.mode == "driver":
        driver(config)
    else:
        worker(config, directory)


if __name__ == "__main__":
    main()
