"""Execute selected upstream function bodies unchanged, without importing GPU/Ray services.

This is a CPU contract harness, NOT a Megatron/SGLang deployment. Only the
parallel-rank metadata is simulated. Tensor math uses real PyTorch; source
definitions are parsed directly from the current read-only Slime checkout.
"""
import ast
import importlib
from argparse import Namespace
from collections.abc import Callable
from types import SimpleNamespace

from finance_opd.config import add_dependency_paths, read_config


def load_definitions(path, names, namespace, *, class_name=None):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    selected = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise AssertionError(f"upstream contract changed: {path}: {names}")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class ParallelRanks:
    rank = 0
    size = 1
    def get_context_parallel_rank(self): return self.rank
    def get_context_parallel_world_size(self): return self.size
    def is_pipeline_last_stage(self): return True


def load_core():
    import torch
    from pathlib import Path
    config = read_config()
    add_dependency_paths(config)
    from slime.utils.types import Sample
    from slime.utils.ppo_utils import compute_policy_loss
    ranks = ParallelRanks()
    def load_function(path):
        module, name = path.rsplit(".", 1)
        return getattr(importlib.import_module(module), name)
    ns = {"torch": torch, "Namespace": Namespace, "Callable": Callable,
          "Sample": Sample, "mpu": ranks, "load_function": load_function}
    root = Path(config["slime"]) / "slime"
    load_definitions(root / "backends/megatron_utils/cp_utils.py",
                     ["get_logits_and_tokens_offset_with_cp", "slice_log_prob_with_cp", "get_sum_of_sample_mean"], ns)
    load_definitions(root / "backends/megatron_utils/loss.py", ["compute_advantages_and_returns"], ns)
    tree = ast.parse((root / "ray/rollout.py").read_text())
    manager_name = next(n.name for n in tree.body if isinstance(n, ast.ClassDef)
                        and any(getattr(child, "name", None) == "_convert_samples_to_train_data" for child in n.body))
    load_definitions(root / "ray/rollout.py", ["_convert_samples_to_train_data"], ns, class_name=manager_name)
    ns["compute_policy_loss"] = compute_policy_loss
    ns["ranks"] = ranks
    return SimpleNamespace(**ns)
