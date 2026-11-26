"""Bounded acceptance stage after one OPD update; never submits GPU training."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    if not directory.is_relative_to(root / "runs"):
        parser.error("use an owned standalone OPD run")
    config = json.loads((directory / "config.json").read_text())
    if config["training"]["updates"] != 1:
        raise ValueError("one-update acceptance requires training.updates=1")
    output = Path(config["output"])
    deadline = config["job"]["deadline_epoch"] + 900
    state_path = directory / "ACCEPTANCE.json"
    def write(value):
        value["time"] = time.time()
        tmp = state_path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(value, indent=2) + "\n")
        tmp.replace(state_path)
    while time.time() < deadline:
        error = directory / "student_error.json"
        if error.exists():
            write({"status": "blocked_student_failed", "gpu_training_verified": False,
                   "error": json.loads(error.read_text())})
            return
        result = directory / "student_result.json"
        marker = output / "checkpoints/latest_checkpointed_iteration.txt"
        if result.exists() and marker.exists():
            value = json.loads(result.read_text())
            if value.get("process_exit_code") == 0 and marker.read_text().strip() == "0":
                break
        write({"status": "waiting_for_one_update_checkpoint", "gpu_training_verified": False})
        time.sleep(20)
    else:
        write({"status": "deadline_exceeded", "gpu_training_verified": False})
        return
    record = directory / "PARAMETER_VERIFICATION_JOB.json"
    if record.exists():
        job = json.loads(record.read_text())["name"]
    else:
        image = config.get("job", {}).get("verification_image")
        mounts = config.get("job", {}).get("mounts")
        if not isinstance(image, str) or not image or not isinstance(mounts, list) or not mounts or not all(isinstance(m, str) and m for m in mounts):
            raise ValueError("configure job.verification_image and job.mounts for CPU verification")
        with (directory / "parameter_verification_submit.lock").open("x") as lock:
            lock.write("one CPU acceptance submission; no GPU request\n")
        command = ["rjob", "submit", "--name", "opd-verify-" + directory.name[-15:],
            "--namespace", "ailab-evobox", "--charged-group", "evobox_cpu_task", "--priority", "9",
            "--cpu", "4", "--memory", "16384", "--image", image]
        for mount in mounts:
            command += ["--mount", mount]
        command += ["--", "timeout", "--signal=TERM", "600", "python3", "-B",
            str(root / "scripts/verify_hf_delta.py"), str(directory)]
        (directory / "parameter_verification_argv.json").write_text(json.dumps(command, indent=2))
        process = subprocess.run(command, capture_output=True, text=True, timeout=150)
        text = process.stdout + process.stderr
        (directory / "parameter_verification_submit.log").write_text(text)
        match = re.search(r"created rjob_name:\s*([\w-]+)", text)
        if process.returncode or not match:
            write({"status": "cpu_verification_submit_failed", "gpu_training_verified": False})
            return
        job = match.group(1)
        record.write_text(json.dumps({"name": job, "gpus": 0, "cpu": 4, "memory_mib": 16384}, indent=2))
    write({"status": "checkpoint_saved_waiting_for_parameter_verification", "job": job,
           "gpu_training_verified": False})
    proof = directory / "PARAMETER_VERIFICATION.json"
    while time.time() < deadline:
        if proof.exists():
            value = json.loads(proof.read_text())
            passed = value.get("passed") is True
            write({"status": "one_update_verified" if passed else "parameter_change_unverified",
                   "gpu_training_verified": passed, "parameter_verification": str(proof), "job": job})
            # Refresh the human/machine summary with the now-available proof.
            subprocess.run(["python3", str(root / "scripts/watch_trial.py"), str(directory), "--once"], timeout=60)
            return
        time.sleep(20)
    write({"status": "cpu_verification_pending_deadline", "gpu_training_verified": False, "job": job})


if __name__ == "__main__":
    main()
