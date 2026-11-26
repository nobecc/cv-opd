"""Verify actual native scoring before requesting CV student GPUs."""
import argparse
import hashlib
import json
from pathlib import Path

from finance_opd.config import ROOT, read_config
from finance_opd.value import native_report_score, validate_dataset_references
from finance_opd.protocol import finite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(ROOT / "runs"):
        raise ValueError("use a standalone run directory")
    config = read_config(directory / "config.json")
    if config.get("job", {}).get("judge_selection_pending"):
        raise ValueError("select a native judge before running this preflight")
    if not config["continuation"]["enabled"]:
        raise ValueError("CV-OPD must be enabled")
    validate_dataset_references(config)
    row = next(json.loads(line) for line in Path(config["dataset"]).read_text().splitlines() if line.strip())
    task = {**row["metadata"], "task": row["task"]}
    score = finite(native_report_score({"task": task,
        "report": "FINISH: I have not inspected the financial evidence. No financial conclusions can be established.",
        "references_path": config["continuation"]["references_path"],
        "evaluator_config": config["continuation"]["evaluator_config"],
        "output": str(directory / "judge-preflight-native")}), "preflight answer quality")
    audit = {"passed": True, "real_native_scoring": True, "score": score,
             "task_id": task["id"], "gpu_training_executed": False}
    for key, path in [("evaluator_sha256", config["continuation"]["evaluator_config"]),
                      ("dataset_sha256", config["dataset"]),
                      ("references_sha256", config["continuation"]["references_path"])]:
        audit[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    (directory / "cv_judge_preflight.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
