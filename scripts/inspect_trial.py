"""Persistent local inspection of current-trial.json; no job submission or messaging.

One flock-protected process follows the current run, including after a failure.
Queries are bounded. Failed queries retain explicitly stale last-known status.
The separate trial controller owns resource deadlines and job termination.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai"))


def write(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".inspection-", delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(f.name, path)


def tail(path, limit=96000):
    if not path.is_file():
        return ""
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - limit))
        return f.read(limit).decode("utf-8", errors="replace")


def diagnose(status, logs):
    if status["phase"] not in {"failed_or_stopped", "student_exited_incomplete_or_unverified"}:
        return None  # Log ERROR/warning text alone never establishes job failure.
    if "were killed due to the node running low on memory" in logs and "MegatronTrainRayActor.save_model" in logs:
        return {"cause": "checkpoint_host_out_of_memory",
                "evidence": "Ray killed MegatronTrainRayActor.save_model for host memory",
                "next_action": "Inspect checkpoint CPU staging; consider native torch format and lower optimizer_offload_fraction.",
                "repair_status": "needs_repair"}
    rules = [
        ("exceeds the maximum allowed length", "engine_context_limit",
         "Budget student sampling against both actual token prefixes with engine safety margin; compact native history earlier."),
        ("were killed due to the node running low on memory", "ray_host_out_of_memory",
         "Reduce CPU optimizer state residency; check optimizer_offload_fraction and offload_train together. Keep Ray memory protection enabled."),
        ("GDN does not support packed sequence", "gdn_packed_sequence_unsupported",
         "Set training.qkv_format=bshd (static micro-batch=1); validate and use a new run directory."),
        ("prompt must be a list when processor is not None", "processor_prompt_type",
         "Enable Slime message wrapping with --multimodal-keys {}; preserve task text in the adapter."),
        ("CUDA out of memory", "cuda_out_of_memory", "Inspect GPU memory and sequence lengths before changing resource bounds."),
        ("ImagePullBackOff", "image_pull_failure", "Inspect image availability and scheduler events."),
    ]
    for needle, cause, action in rules:
        if needle in logs:
            return {"cause": cause, "evidence": needle, "next_action": action, "repair_status": "needs_repair"}
    return {"cause": "unclassified_failure", "next_action": "Inspect archived logs and exit records before retrying.",
            "repair_status": "needs_repair"}


def snapshot_log(directory, incident, role, name):
    if not name:
        return
    try:
        p = subprocess.run(["rjob", "logs", "job", name, "-n", "200"],
                           capture_output=True, text=True, timeout=20)
        (incident / f"{role}_scheduler.log").write_text(p.stdout + p.stderr)
    except (OSError, subprocess.SubprocessError) as error:
        (incident / f"{role}_scheduler.log").write_text(str(error))


def inspect(directory, interval=30):
    record = {"checked_at": now().isoformat(), "run_dir": str(directory),
              "pid": os.getpid(), "interval_seconds": interval, "monitor_state": "running",
              "next_check_at": (now() + timedelta(seconds=interval)).isoformat()}
    try:
        # Isolate the RJob SDK so a stuck network request cannot block monitoring forever.
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/watch_trial.py"),
                                 str(directory), "--once"], capture_output=True, text=True, timeout=45)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout)[-2000:])
        status = json.loads((directory / "STATUS.json").read_text())
        record.update(query_ok=True, status_is_stale=False, last_known_status=status)
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
        path = directory / "STATUS.json"
        try:
            last = json.loads(path.read_text()) if path.is_file() else None
        except ValueError:
            last = None
        record.update(query_ok=False, status_is_stale=True, last_known_status=last,
                      query_error=f"{type(error).__name__}: {error}")
        write(directory / "INSPECTION.json", record)
        write(RUNS / "INSPECTION_STATUS.json", record)
        return record
    logs = "\n".join(tail(directory / f"{role}.log") for role in ("student", "teacher"))
    diagnosis = diagnose(status, logs)
    if diagnosis:
        signature = json.dumps([str(directory), status["phase"], diagnosis["cause"], status.get("errors")], sort_keys=True)
        incident = directory / "inspection" / hashlib.sha256(signature.encode()).hexdigest()[:16]
        record.update(diagnosis=diagnosis, incident_dir=str(incident), attention_required=True)
        # Publish failure immediately, before fetching scheduler logs.
        write(directory / "INSPECTION.json", record)
        write(RUNS / "INSPECTION_STATUS.json", record)
        incident.mkdir(parents=True, exist_ok=True)
        if not (incident / "complete.json").exists():
            write(incident / "STATUS.json", status)
            write(incident / "DIAGNOSIS.json", diagnosis)
            for role in ("student", "teacher"):
                (incident / f"{role}.tail.log").write_text(tail(directory / f"{role}.log"))
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda item: snapshot_log(directory, incident, item[0], item[1].get("name")),
                              status.get("jobs", {}).items()))
            write(incident / "complete.json", {"archived_at": now().isoformat()})
    else:
        record["attention_required"] = False
        log = directory / "student.log"
        if log.exists() and status["phase"] == "student_process_started_updates_not_yet_verified":
            age = time.time() - log.stat().st_mtime
            record["student_log_age_seconds"] = round(age)
            if age > 1200:
                record.update(attention_required=True, attention_reason="student_log_unchanged_over_20_minutes_not_proof_of_failure")
    write(directory / "INSPECTION.json", record)
    write(RUNS / "INSPECTION_STATUS.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("interval must be at least 5 seconds")
    with (RUNS / "inspection.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("inspection already running")
        write(RUNS / "inspection_process.json", {"pid": os.getpid(), "started_at": now().isoformat(),
              "interval_seconds": args.interval, "scope": "follow current-trial.json; no submissions; no external notifications"})
        while True:
            start = time.monotonic()
            try:
                directory = Path(json.loads((RUNS / "current-trial.json").read_text())["run_dir"]).resolve()
                if not directory.is_relative_to(RUNS.resolve()):
                    raise ValueError("current run must be inside cv-opd/runs")
                record = inspect(directory, args.interval)
                print(json.dumps({"time": record["checked_at"], "query_ok": record["query_ok"],
                                  "phase": (record.get("last_known_status") or {}).get("phase"),
                                  "attention_required": record.get("attention_required")}), flush=True)
            except Exception as error:
                write(RUNS / "INSPECTION_STATUS.json", {"checked_at": now().isoformat(),
                      "pid": os.getpid(), "monitor_state": "running", "query_ok": False,
                      "status_is_stale": True, "query_error": f"{type(error).__name__}: {error}"})
                print(f"inspection error: {type(error).__name__}: {error}", flush=True)
            if args.once:
                return
            time.sleep(max(1, args.interval - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
