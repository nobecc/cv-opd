"""Scoped file-read guard environment for native CV research tools.

This prevents ordinary Python file access to supervision artifacts. It is not
an operating-system sandbox for adversarial native code or child processes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def private_paths(config: dict) -> list[str]:
    reference = config.get("continuation", {}).get("references_path")
    if reference is None and config.get("dataset"):
        reference = Path(config["dataset"]).parent / "private_references.jsonl"
    paths = [reference, config.get("continuation", {}).get("evaluator_config"),
             config.get("runtime", {}).get("web_config_path")]
    return [str(Path(path).expanduser().absolute()) for path in paths if path is not None]


def guard_environment(config: dict, *, run_dir: Path | None = None) -> dict[str, str]:
    paths = private_paths(config)
    if run_dir is not None:
        paths += [str((run_dir / name).absolute()) for name in
                  ("continuation", "turns.jsonl.gz", "sampled.private", "pi-state/web-search.json")]
    guards = Path(__file__).resolve().parents[1] / "runtime_guards"
    return {"FINANCE_OPD_PRIVATE_PATHS": json.dumps(paths),
            "PYTHONPATH": os.pathsep.join([str(guards), str(config["pi_finance"])])}
