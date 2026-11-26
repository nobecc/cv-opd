"""Read-only sampled parameter comparison between two completed local torch checkpoints."""
import argparse
import json
import pickle
import types
from pathlib import Path

from inspect_trial import now, write


def metadata_enum_value(value):
    # Training-argument metadata only; model tensors retain normal torch loading.
    return value


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in {("megatron.core.enums", "ModelType"),
                              ("megatron.core.transformer.enums", "AttnBackend")}:
            return metadata_enum_value
        return super().find_class(module, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--before", type=int, default=0)
    parser.add_argument("--after", type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(root / "runs") or args.before >= args.after:
        parser.error("use a local trial and increasing checkpoint indices")
    config = json.loads((directory / "config.json").read_text())
    checkpoint = Path(config["output"]) / "checkpoints"
    if int((checkpoint / "latest_checkpointed_iteration.txt").read_text()) < args.after:
        raise ValueError("final checkpoint has not finished saving")
    files = [checkpoint / f"iter_{i:07d}/mp_rank_00/model_optim_rng.pt" for i in (args.before, args.after)]
    if any(not f.is_file() for f in files):
        raise ValueError("expected two complete native torch TP-rank-0 checkpoints")
    # These files are created by our own training job. mmap avoids materializing
    # the full ~14 GB shard in host memory; only small blocks are compared.
    import torch
    torch.set_num_threads(2)
    # Avoid installing the entire GPU Megatron stack merely to decode two
    # enums in unused argparse metadata. Preserve their values; never substitute
    # tensor classes, storages, model weights, or reconstruction functions.
    metadata_pickle = types.ModuleType("finance_checkpoint_metadata_pickle")
    metadata_pickle.Unpickler = MetadataUnpickler
    metadata_pickle.load = pickle.load
    metadata_pickle.loads = pickle.loads
    values = [torch.load(f, map_location="cpu", mmap=True, weights_only=False,
                         pickle_module=metadata_pickle) for f in files]

    def weights(obj, prefix=""):
        found = {}
        if isinstance(obj, dict):
            for key, value in obj.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value) and name.endswith(("weight", "bias")):
                    found[name] = value
                elif isinstance(value, dict):
                    found.update(weights(value, name))
        return found

    a, b = [weights(value["model"]) for value in values]
    if not a or a.keys() != b.keys():
        raise ValueError("model parameter keys are missing or differ")
    total = changed = 0
    examples = []
    max_delta = 0.0
    for name, left in a.items():
        right = b[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"parameter shape/dtype changed: {name}")
        width = 128 if left.ndim <= 2 else 8
        slices = tuple(slice(0, min(width, size)) for size in left.shape)
        x, y = left[slices].float(), right[slices].float()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError(f"nonfinite sampled parameters: {name}")
        count = int(torch.count_nonzero(x != y))
        delta = float((x - y).abs().max()) if x.numel() else 0.0
        total += x.numel()
        changed += count
        max_delta = max(max_delta, delta)
        if count and len(examples) < 8:
            examples.append({"parameter": name, "changed_sampled_elements": count, "max_abs_change": delta})
    report = {"time": now().isoformat(), "run_dir": str(directory),
              "method": "sampled_checkpoint_model_parameters", "passed": changed > 0,
              "sampled_only": True, "tensor_parallel_rank": 0,
              "training_metadata_enums_loaded_as_values": True,
              "before_iteration": args.before, "after_iteration": args.after,
              "compared_parameters": len(a), "sampled_elements": total,
              "changed_sampled_elements": changed, "all_sampled_values_finite": True,
              "max_abs_change": max_delta, "examples": examples,
              "files": [{"path": str(f), "size": f.stat().st_size, "mtime_ns": f.stat().st_mtime_ns} for f in files]}
    write(directory / "PARAMETER_VERIFICATION.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit("no change found in sampled parameter blocks")


if __name__ == "__main__":
    main()
