"""Bounded retention of completed checkpoints created by one standalone run."""
import json
import re
import shutil
import time
from pathlib import Path

from .config import ROOT


def prune_checkpoints(config):
    keep = config["training"].get("keep_checkpoints")
    if keep is None:
        return []
    if not isinstance(keep, int) or isinstance(keep, bool) or keep < 1:
        raise ValueError("keep_checkpoints must be an integer >= 1")
    output = Path(config["output"]).resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("checkpoint retention is limited to standalone runs")
    if config["training"].get("checkpoint_format") != "torch":
        raise ValueError("retention currently requires native torch checkpoints")
    root = output / "checkpoints"
    if root.is_symlink():
        raise ValueError("checkpoint directory must not be a symlink")
    marker = root / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        return []
    text = marker.read_text().strip()
    if not text.isdigit():
        return []  # The upstream writer may be replacing the marker right now.
    latest = int(text)
    tp = config["training"].get("tensor_parallel_size", 1)
    pp = config["training"].get("pipeline_parallel_size", 1)

    def complete(path):
        return all((f := path / (f"mp_rank_{t:02d}" if pp == 1 else f"mp_rank_{t:02d}_{p:03d}") /
                    "model_optim_rng.pt").is_file() and not f.is_symlink() and f.stat().st_size > 0
                   for t in range(tp) for p in range(pp))

    current = root / f"iter_{latest:07d}"
    if current.is_symlink() or not complete(current):
        return []
    completed = []
    for path in root.iterdir():
        match = re.fullmatch(r"iter_(\d{7})", path.name)
        if match and path.is_dir() and not path.is_symlink():
            iteration = int(match[1])
            if iteration <= latest and complete(path):
                completed.append((iteration, path))
    removed = []
    for iteration, path in sorted(completed)[:-keep]:
        if iteration >= latest or path.resolve().parent != root.resolve():
            raise ValueError("unsafe checkpoint retention target")
        shutil.rmtree(path)
        removed.append(str(path))
        with (output / "checkpoint_retention.jsonl").open("a") as log:
            log.write(json.dumps({"time": time.time(), "deleted": str(path), "latest": latest, "keep": keep}) + "\n")
    return removed
