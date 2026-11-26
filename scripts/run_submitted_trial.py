"""Continue one submitted trial: wait for teacher scoring, then submit the student."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from watch_trial import update


def save_logs(directory, status):
    for role, job in status["jobs"].items():
        response = subprocess.run(["rjob", "logs", "job", job["name"], "-n", "160"],
                                  capture_output=True, text=True, timeout=30)
        (directory / f"{role}_scheduler.log").write_text(response.stdout + response.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--hours", type=float, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(root / "runs") or args.hours <= 0:
        raise ValueError("use a prepared cv-opd/runs directory and positive deadline")
    if not (directory / "submission.json").is_file():
        # Externally deployed teachers (e.g. vLLM) have no submitted teacher
        # job; scripts/verify_teacher.py writes their readiness record and an
        # initial empty submission.json instead.
        raise ValueError("submit the teacher first (or run scripts/verify_teacher.py)")
    with (directory / "controller.lock").open("x") as lock:
        lock.write("one controller per trial; inspect controller.log before retrying\n")
    deadline = time.monotonic() + 3600 * args.hours
    previous = None
    while time.monotonic() < deadline:
        try:
            status = update(directory)
        except Exception as error:
            print(f"scheduler query failed; retrying: {type(error).__name__}: {error}", flush=True)
            time.sleep(30)
            continue
        signature = (status["phase"], str(status["jobs"]))
        if signature != previous:
            print(json.dumps(status, ensure_ascii=False), flush=True)
            previous = signature
        if status["phase"] in {"failed_or_stopped", "training_process_completed_checkpoint_saved", "student_exited_incomplete_or_unverified"}:
            save_logs(directory, status)
            return
        if status["phase"] == "teacher_scoring_verified_waiting_for_student" and "student" not in status["jobs"]:
            config = json.loads((directory / "config.json").read_text())
            if config.get("continuation", {}).get("enabled", False):
                # Continue enforcing the resource deadline while a judge choice
                # or real native-scoring preflight remains pending.
                if (config.get("job", {}).get("judge_selection_pending")
                        or not (directory / "cv_judge_preflight.json").is_file()):
                    time.sleep(30)
                    continue
            # submit_trial performs an additional live identity/readiness check
            # before allocating any student GPUs and prevents duplicate attempts.
            try:
                subprocess.run([sys.executable, "-B", str(root / "scripts/submit_trial.py"),
                                str(directory), "--role", "student"], check=True, timeout=150)
            except (subprocess.SubprocessError, OSError) as error:
                (directory / "controller_error.json").write_text(json.dumps({
                    "error": str(error), "action": "submit_student"}) + "\n")
                raise
        time.sleep(30)
    status = update(directory)
    save_logs(directory, status)
    for job in status["jobs"].values():
        if job["state"] not in {"Failed", "Stopped", "Killed", "Succeeded"}:
            subprocess.run(["rjob", "stop", job["name"]], check=True, timeout=60)
    (directory / "controller_timeout.json").write_text(json.dumps({
        "reason": "bounded trial deadline exceeded", "hours": args.hours}) + "\n")
    print(json.dumps(update(directory), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
