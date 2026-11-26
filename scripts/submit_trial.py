"""Submit one trial worker; allocate student GPUs only after teacher readiness."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import urllib.request
import hashlib


def worker_resources(config, role):
    train = config.get("training", {})
    nodes = train.get("nodes", 1) if role == "student" else 1
    total_gpus = (train.get("gpus", 4) if role == "student"
                  else config["teacher"].get("tensor_parallel_size", 4))
    if type(nodes) is not int or nodes < 1 or type(total_gpus) is not int or total_gpus < 1:
        raise ValueError("nodes and total GPUs must be positive integers")
    default_per_node = total_gpus if nodes == 1 else (
        total_gpus // nodes if total_gpus % nodes == 0 else 0)
    gpus_per_node = train.get("gpus_per_node", default_per_node) if role == "student" else total_gpus
    if type(gpus_per_node) is not int or not 1 <= gpus_per_node <= 8:
        raise ValueError("each teacher/student replica supports 1..8 GPUs on one node")
    if total_gpus != nodes * gpus_per_node:
        raise ValueError("training.gpus must equal nodes * gpus_per_node")
    # CPU/memory are per replica. Preserve the tested 100G/GPU student host
    # ratio for partial CPU Adam offload (above the required 64G/GPU minimum).
    return {"nodes": nodes, "gpus": gpus_per_node, "gpus_per_node": gpus_per_node,
            "total_gpus": total_gpus, "cpus": 8 * gpus_per_node,
            "memory_mib": (102400 if role == "student" else 65536) * gpus_per_node}


def submission_command(config, role, image, resources, root, directory):
    multinode = role == "student" and resources["nodes"] > 1
    namespace = "ailab-safeshared" if multinode else "ailab-evobox"
    charged_group = "safeshared_gpu" if multinode else "evobox_gpu_pool"
    command = ["rjob", "submit", "--name", "opd-" + role + "-" + directory.name[-15:],
               "--namespace", namespace, "--charged-group", charged_group, "--priority", "9",
               "--private-machine", "group", "-P", str(resources["nodes"]),
               "--gpu", str(resources["gpus_per_node"]), "--cpu", str(resources["cpus"]),
               "--memory", str(resources["memory_mib"]), "--image", image]
    mounts = config.get("job", {}).get("mounts")
    if not isinstance(mounts, list) or not mounts or not all(isinstance(m, str) and m for m in mounts):
        raise ValueError("configure job.mounts explicitly for your cluster")
    for mount in mounts:
        command += ["--mount", mount]
    if multinode:
        master_port = config.get("job", {}).get("master_port", 29500)
        if type(master_port) is not int or not 1 <= master_port <= 65535:
            raise ValueError("job.master_port must be an integer in 1..65535")
        command += ["--custom-resources", "rdma/mlnx_shared=8", "--gang-start=true",
                    "-e", "DISTRIBUTED_JOB=true", "-e", f"MASTER_PORT={master_port}"]
    command += ["--", "bash", str(root / "scripts/cluster_worker.sh"), role,
                "--config", str(directory / "config.json"), "--run-dir", str(directory)]
    return command


def check_teacher_ready(directory):
    for name in ("teacher_error.json", "teacher_exit.json"):
        if (directory / name).exists():
            raise ValueError(f"teacher already exited or failed: {name}")
    ready_path = directory / "teacher_ready.json"
    if not ready_path.is_file():
        raise ValueError("teacher scoring is not ready; student GPUs were not requested")
    ready = json.loads(ready_path.read_text())
    config = json.loads((directory / "config.json").read_text())
    if config.get("continuation", {}).get("enabled", False):
        audit_path = directory / "cv_judge_preflight.json"
        if config.get("job", {}).get("judge_selection_pending") or not audit_path.is_file():
            raise ValueError("CV native judge preflight is not ready; student GPUs were not requested")
        audit = json.loads(audit_path.read_text())
        for key, path in [("evaluator_sha256", config["continuation"]["evaluator_config"]),
                          ("dataset_sha256", config["dataset"]),
                          ("references_sha256", config["continuation"]["references_path"])]:
            if audit.get(key) != hashlib.sha256(Path(path).read_bytes()).hexdigest():
                raise ValueError("CV native judge preflight is stale")
        if audit.get("passed") is not True or audit.get("real_native_scoring") is not True:
            raise ValueError("CV native judge preflight failed")
    model = config["teacher"]["model"]
    if ready.get("scoring_verified") is not True or ready.get("model") != model:
        raise ValueError("teacher readiness record does not match this trial")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = ready["url"].rstrip("/")
    if config["teacher"].get("transport", "sglang") == "vllm":
        # vLLM servers report served names, not local weight paths.
        with opener.open(url + "/v1/models", timeout=10) as response:
            listed = json.load(response).get("data")
        names = [item.get("id") for item in listed if isinstance(item, dict)]
        if names != [model]:
            raise ValueError(f"live teacher served models {names} do not match {model}")
    else:
        with opener.open(url + "/get_model_info", timeout=10) as response:
            info = json.load(response)
        if Path(info["model_path"]).resolve() != Path(model).resolve():
            raise ValueError("live teacher model identity mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--role", choices=["teacher", "student"], required=True)
    parser.add_argument("--image", help="Override only the selected worker's image")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(root / "runs"):
        raise ValueError("run directory must be inside cv-opd/runs")
    if not (directory / "config.json").is_file():
        raise ValueError("prepare config.json before submitting")
    config = json.loads((directory / "config.json").read_text())
    resources = worker_resources(config, args.role)
    result = directory / "submission.json"
    record = json.loads(result.read_text()) if result.exists() else {
        "jobs": {}, "gpu_training_verified": False}
    jobs = record["jobs"]
    if args.role in jobs:
        raise ValueError(f"{args.role} submission already exists; use a new run for retries")
    if args.role == "student":
        check_teacher_ready(directory)
    image = args.image or config.get("job", {}).get(f"{args.role}_image")
    if not isinstance(image, str) or not image:
        raise ValueError("provide --image or configure job.<role>_image for your cluster")
    command = submission_command(config, args.role, image, resources, root, directory)
    # Keep the lock as an audit record, including an uncertain CLI timeout.
    with (directory / f"{args.role}_submission.lock").open("x") as lock:
        lock.write("one submission attempt; inspect scheduler before any retry\n")
    for role, image in [(args.role, image)]:
        record["status"] = f"submitting_{role}"
        result.write_text(json.dumps(record, indent=2) + "\n")
        # 27B FP32 master weights and two Adam moments alone total ~324 GB.
        memory_mib = resources["memory_mib"]
        (directory / f"{role}_submit_argv.json").write_text(json.dumps(command, indent=2) + "\n")
        proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
        output = proc.stdout + proc.stderr
        (directory / f"{role}_submit.log").write_text(output)
        match = re.search(r"created rjob_name:\s*([\w-]+)", output)
        jobs[role] = {"image": image, "exit_code": proc.returncode,
                      "name": match.group(1) if match else None, "gpus": resources["gpus_per_node"],
                      "nodes": resources["nodes"], "gpus_per_node": resources["gpus_per_node"],
                      "total_gpus": resources["total_gpus"], "cpus": resources["cpus"],
                      "memory_mib": memory_mib}
        result.write_text(json.dumps(record, indent=2) + "\n")
        print(output, flush=True)
        if proc.returncode or not match:
            record["status"] = "submission_failed"
            result.write_text(json.dumps(record, indent=2) + "\n")
            raise RuntimeError(f"{role} submission failed; see {directory}")
    record["status"] = ("teacher_submitted_student_not_requested" if args.role == "teacher"
                        else "submitted_not_yet_verified_running")
    result.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
