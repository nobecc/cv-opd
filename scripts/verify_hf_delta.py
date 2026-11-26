"""Read-only sampled parameter check after ONE OPD update (rollout index 0).

Compare TP-rank-0 vocabulary/norm slices with the exact input HF checkpoint;
never require a second optimizer update merely to obtain two checkpoints.
Run on a CPU worker with the existing Slime/PyTorch image, no GPUs needed.
"""
import argparse
from datetime import datetime
import json
import pickle
from pathlib import Path
import types
from zoneinfo import ZoneInfo


def enum_value(value):
    return value


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in {("megatron.core.enums", "ModelType"),
                              ("megatron.core.transformer.enums", "AttnBackend")}:
            return enum_value
        return super().find_class(module, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    config = json.loads((directory / "config.json").read_text())
    base = Path(config["student_model"]).resolve()
    output = Path(config["output"]).resolve()
    iteration = int((output / "checkpoints/latest_checkpointed_iteration.txt").read_text())
    if config["training"]["updates"] != 1 or iteration != 0:
        raise ValueError("this verifier is for exactly one completed OPD rollout/update, index 0")
    checkpoint = output / "checkpoints/iter_0000000/mp_rank_00/model_optim_rng.pt"
    if not checkpoint.is_file():
        raise ValueError("TP-rank-0 checkpoint is incomplete")
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    meta = types.ModuleType("finance_owned_checkpoint_metadata")
    meta.Unpickler, meta.load, meta.loads = MetadataUnpickler, pickle.load, pickle.loads
    value = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False, pickle_module=meta)
    def flatten(obj, prefix=""):
        found = {}
        for key, item in obj.items():
            name = prefix + str(key)
            if isinstance(item, torch.Tensor):
                found[name] = item
            elif isinstance(item, dict):
                found.update(flatten(item, name + "."))
        return found
    weights = flatten(value["model"])
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    mapping = {"embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
               "output_layer.weight": "lm_head.weight",
               "decoder.final_layernorm.weight": "model.language_model.norm.weight"}
    comparisons, files = [], {str(checkpoint): checkpoint}
    for suffix, hf_name in mapping.items():
        candidates = [name for name in weights if name.endswith(suffix)]
        if len(candidates) != 1 or hf_name not in index:
            continue
        name = candidates[0]
        tensor = weights[name]
        if tensor.ndim not in {1, 2}:
            raise ValueError("unexpected mapped parameter shape")
        width = [min(512, tensor.shape[0])] + ([min(128, tensor.shape[1])] if tensor.ndim == 2 else [])
        slices = tuple(slice(0, n) for n in width)
        hf_file = base / index[hf_name]
        files[str(hf_file)] = hf_file
        with safe_open(hf_file, framework="pt", device="cpu") as handle:
            before = handle.get_slice(hf_name)[slices].to(tensor.dtype).float()
        after = tensor[slices].float()
        if before.shape != after.shape or not torch.isfinite(before).all() or not torch.isfinite(after).all():
            raise ValueError("mapped parameter shape/value verification failed")
        comparisons.append({"checkpoint_parameter": name, "hf_parameter": hf_name,
            "elements": before.numel(), "changed": int(torch.count_nonzero(before != after)),
            "max_abs_change": float((before - after).abs().max()), "dtype": str(tensor.dtype)})
    if not comparisons:
        raise ValueError("no explicitly mapped checkpoint parameters found")
    changed = sum(item["changed"] for item in comparisons)
    report = {"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "run_dir": str(directory),
        "method": "sampled_hf_to_checkpoint_parameters", "passed": changed > 0,
        "sampled_only": True, "tensor_parallel_rank": 0, "after_iteration": 0,
        "all_sampled_values_finite": True, "changed_sampled_elements": changed,
        "input_hf_checkpoint": str(base), "comparisons": comparisons,
        "files": [{"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                  for p in files.values()]}
    (directory / "PARAMETER_VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("no parameter change in the sampled slices; update is not independently verified")


if __name__ == "__main__":
    main()
