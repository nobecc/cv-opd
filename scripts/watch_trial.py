"""Read-only scheduler monitor for one submitted trial; no notifications or job mutations."""
import argparse
from datetime import datetime
import json
import math
import os
import tempfile
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

from brainpp.rjob.client import RJobClient


def verified_parameter_change(directory, output, iteration):
    path = directory / "PARAMETER_VERIFICATION.json"
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text())
        if not (value.get("passed") is True
                and value.get("method") in {"sampled_checkpoint_model_parameters", "sampled_hf_to_checkpoint_parameters"}
                and value.get("run_dir") == str(directory.resolve())
                and value.get("after_iteration") == iteration
                and value.get("changed_sampled_elements", 0) > 0
                and value.get("all_sampled_values_finite") is True
                and len(value.get("files", [])) >= 2):
            return False
        method = value["method"]
        if method == "sampled_checkpoint_model_parameters" and len(value["files"]) != 2:
            return False
        config = json.loads((directory / "config.json").read_text()) if method == "sampled_hf_to_checkpoint_parameters" else {}
        baseline = Path(config["student_model"]).resolve() if config else None
        allowed_hf = set()
        if baseline is not None:
            index = json.loads((baseline / "model.safetensors.index.json").read_text())["weight_map"]
            allowed_hf = {(baseline / name).resolve() for name in index.values()}
        seen_checkpoint = seen_baseline = False
        for info in value["files"]:
            file = Path(info["path"]).resolve()
            if not file.is_relative_to((output / "checkpoints").resolve()):
                if value.get("method") != "sampled_hf_to_checkpoint_parameters":
                    return False
                if value.get("input_hf_checkpoint") != str(baseline) or file not in allowed_hf:
                    return False
                seen_baseline = True
            else:
                seen_checkpoint = True
            stat = file.stat()
            if stat.st_size != info["size"] or stat.st_mtime_ns != info["mtime_ns"]:
                return False
        return method != "sampled_hf_to_checkpoint_parameters" or (seen_checkpoint and seen_baseline)
    except (ValueError, KeyError, TypeError, OSError):
        return False


def update(directory):
    submission = json.loads((directory / "submission.json").read_text())
    visible = {job.metadata.name: job for job in RJobClient().list()}
    states = {role: visible[info["name"]].status.current.name if info["name"] in visible else "Unknown"
              for role, info in submission["jobs"].items()}
    errors = {role: json.loads((directory / f"{role}_error.json").read_text())
              for role in states if (directory / f"{role}_error.json").is_file()}
    result_path = directory / "student_result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    exits = {role: json.loads((directory / f"{role}_exit.json").read_text())
             for role in states if (directory / f"{role}_exit.json").is_file()}
    config = json.loads((directory / "config.json").read_text())
    output = Path(config["output"])
    marker = output / "checkpoints/latest_checkpointed_iteration.txt"
    marker_text = marker.read_text().strip() if marker.is_file() else ""
    iteration = int(marker_text) if marker_text.isdigit() else None
    checkpoint_files = [p for p in ((output / "checkpoints").glob(f"iter_{iteration:07d}/**/*") if iteration is not None else [])
                        if p.is_file() and p.name != "common.pt" and p.stat().st_size > 0 and p.suffix in {".pt", ".distcp", ".safetensors"}]
    shards_complete = bool(checkpoint_files)
    if config["training"].get("checkpoint_format", "torch_dist") == "torch" and iteration is not None:
        tp = config["training"].get("tensor_parallel_size", 1)
        pp = config["training"].get("pipeline_parallel_size", 1)
        expected = [output / "checkpoints" / f"iter_{iteration:07d}" /
                    (f"mp_rank_{t:02d}" if pp == 1 else f"mp_rank_{t:02d}_{p:03d}") /
                    "model_optim_rng.pt" for t in range(tp) for p in range(pp)]
        shards_complete = all(p.is_file() and p.stat().st_size > 0 for p in expected)
    # Slime saves rollout_id directly: completed rolls 1/2 have indices 0/1.
    checkpointed_rollouts = iteration + 1 if iteration is not None and shards_complete else 0
    complete = bool(result and result.get("process_exit_code") == 0
                    and checkpointed_rollouts >= config["training"].get("updates", 1)
                    and result.get("latest_checkpoint_iteration") == iteration and checkpoint_files)
    if errors or any(state in {"Failed", "Stopped", "Killed"} for state in states.values()) or any(
            v.get("exit_code", v.get("code", 0)) != 0 for v in exits.values()):
        phase = "failed_or_stopped"
    elif complete:
        phase = "training_process_completed_checkpoint_saved"
    elif result or "student" in exits or states.get("student") == "Succeeded":
        phase = "student_exited_incomplete_or_unverified"
    elif (directory / "student.log").exists():
        phase = "student_process_started_updates_not_yet_verified"
    elif (directory / "teacher_ready.json").exists():
        phase = "teacher_scoring_verified_waiting_for_student"
    elif (directory / "teacher.log").exists():
        phase = "teacher_loading"
    elif (directory / "teacher_inventory.json").exists():
        phase = "teacher_process_started_initializing"
    elif any(state == "Starting" for state in states.values()):
        phase = "container_starting"
    else:
        phase = "queued_waiting_for_resources"
    probe_path = directory / "teacher_probe.json"
    probe = json.loads(probe_path.read_text()) if probe_path.is_file() else {}
    scores = probe.get("teacher_log_probs", [])
    verified = (probe.get("real_model") is True and bool(scores)
                and len(scores) == len(probe.get("output", []))
                and all(isinstance(x, (float, int)) and math.isfinite(x) for x in scores))
    if phase == "teacher_scoring_verified_waiting_for_student" and not verified:
        phase = "teacher_readiness_unverified"
    episodes = list((output / "episodes").glob("*/episode_status.json"))
    status = {"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "phase": phase,
              "jobs": {role: {"name": submission["jobs"][role]["name"], "state": state}
                       for role, state in states.items()},
              "teacher_scoring_verified": verified,
              "mode": "cv_opd" if config.get("continuation", {}).get("enabled", False) else "plain_opd",
              "startup": {"judge_selection_pending": config.get("job", {}).get("judge_selection_pending", False),
                          "native_judge_preflight_present": (directory / "cv_judge_preflight.json").is_file()},
              "evidence": {"episode_status_files": len(episodes),
                           "rollout_files": len(list((output / "rollout").glob("*.pt"))),
                           "checkpoint_iteration": iteration,
                           "checkpointed_rollouts": checkpointed_rollouts,
                           "checkpoint_tensor_files": len(checkpoint_files),
                           "checkpoint_shards_complete": shards_complete,
                           "target_updates": config["training"].get("updates", 1)},
              "student_result": result, "errors": errors,
              "parameter_change_independently_verified": verified_parameter_change(directory, output, iteration)}
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, prefix=".STATUS-", delete=False) as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
        f.write("\n")
        temporary = Path(f.name)
    os.replace(temporary, directory / "STATUS.json")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--hours", type=float, default=4)
    args = parser.parse_args()
    deadline = time.monotonic() + 3600 * args.hours
    previous = None
    while time.monotonic() < deadline:
        try:
            status = update(args.run_dir)
            if args.once:
                print(json.dumps(status, ensure_ascii=False), flush=True)
                return
            signature = (status["phase"], str(status["jobs"]))
            if signature != previous:
                print(json.dumps(status, ensure_ascii=False), flush=True)
                previous = signature
                for role, job in status["jobs"].items():
                    if job["state"] in {"Running", "Succeeded", "Failed"}:
                        response = subprocess.run(["rjob", "logs", "job", job["name"], "-n", "120"],
                                                  capture_output=True, text=True, timeout=25)
                        (args.run_dir / f"{role}_scheduler.log").write_text(response.stdout + response.stderr)
            if args.once or status["phase"] in {"failed_or_stopped", "training_process_completed_checkpoint_saved", "student_exited_incomplete_or_unverified"}:
                return
        except Exception as error:
            print(f"monitor query error: {type(error).__name__}: {error}", flush=True)
            if args.once:
                raise
        time.sleep(30)


if __name__ == "__main__":
    main()
