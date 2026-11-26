"""Reject CV episodes without a supervised student token before optimizer work."""
from __future__ import annotations

import json
from pathlib import Path
import time

from .config import read_config
from .protocol import finite

_REJECTIONS: dict[str, int] = {}


def supervised_tokens(group):
    """Slime wraps each episode's variable-length list of model-call samples."""
    total = 0
    for item in group:
        if isinstance(item, list):
            total += supervised_tokens(item)
            continue
        mask = item.loss_mask
        if len(mask) != item.response_length or any(type(x) is not int or x not in (0, 1) for x in mask):
            raise ValueError("invalid sampled-token mask at OPD admission")
        weight = finite(item.reward["opd_weight"], "opd weight")
        if not 0 <= weight <= 10:
            raise ValueError("opd weight outside [0,10]")
        if weight > 0:
            total += sum(mask)
    return total


def require_cv_signal(args, group):
    from slime.rollout.filter_hub.base_types import DynamicFilterOutput

    config = read_config()
    if not config.get("continuation", {}).get("enabled", False):
        raise ValueError("CV signal filter requires continuation-enabled training")
    limit = config["training"].get("max_zero_signal_rollouts", 4)
    if type(limit) is not int or limit < 1:
        raise ValueError("max_zero_signal_rollouts must be a positive integer")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    key = str(output.resolve())
    tokens = supervised_tokens(group)
    rejected = 0 if tokens else _REJECTIONS.get(key, 0) + 1
    _REJECTIONS[key] = rejected
    with (output / "cv_signal_filter.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": time.time(), "keep": bool(tokens),
            "supervised_tokens": tokens, "consecutive_zero_signal_rollouts": rejected,
            "rejection_limit": limit, "original_samples_unchanged": True}) + "\n")
    if rejected >= limit:
        raise RuntimeError("No CV training signal in bounded fresh rollouts; optimizer not advanced for these attempts")
    return DynamicFilterOutput(keep=bool(tokens), reason=None if tokens else "no_cv_supervised_tokens")
