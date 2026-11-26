"""Small JSON configuration; heavyweight training imports remain lazy."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = (
    "ddr_10k", "finance_complex_qa", "finsearchcomp_t2", "finsearchcomp_t3",
    "dataclawbench", "finance_agent_bench_v1_1", "finance_agent_bench_v2",
    "finfirst", "frontier_finance",
)


def read_config(path: str | Path | None = None) -> dict:
    source = Path(path or os.environ.get("FINANCE_OPD_CONFIG", ROOT / "configs/qwen35_9b.json")).resolve()
    config = json.loads(source.read_text())
    workspace = (source.parent / config.get("workspace", "../..")).resolve()
    config["workspace"] = str(workspace)
    for key, default in {
        "slime": "slime", "safactory": "SAfactory", "pi_finance": "pi-finance",
        "dataset": str(ROOT / "data/current/train.jsonl"), "output": str(ROOT / "runs/qwen35_9b"),
    }.items():
        config[key] = str((workspace / config.get(key, default)).resolve())
    config["config_path"] = str(source)
    config.setdefault("teacher", {})
    config.setdefault("training", {})
    config.setdefault("runtime", {})
    config["runtime"].setdefault("tool_protocol", "qwen3_xml_v1")
    if config["runtime"]["tool_protocol"] != "qwen3_xml_v1":
        raise ValueError("new OPD runs require tool_protocol=qwen3_xml_v1; preserve old runs for replay")
    config.setdefault("context_hook", None)
    from .cv import validate_continuation_config
    validate_continuation_config(config)
    for key in ("references_path", "evaluator_config"):
        if config["continuation"].get(key):
            config["continuation"][key] = str((workspace / config["continuation"][key]).resolve())
    if config["runtime"].get("temperature", 1.0) != 1.0:
        raise ValueError("v0.1 requires temperature=1 to keep engine and trainer log-prob definitions aligned")
    for key in ["dataset", "output"]:
        if not Path(config[key]).is_relative_to(ROOT):
            raise ValueError(f"{key} must be inside the standalone project {ROOT}")
    return config


def add_dependency_paths(config: dict) -> None:
    import sys
    # No importing llm_proxy: its import would create logs in SAfactory.
    for path in [config["slime"], config["safactory"], str(Path(config["safactory"]) / "rl")]:
        if path not in sys.path:
            sys.path.insert(0, path)
    sys.dont_write_bytecode = True
