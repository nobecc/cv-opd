"""Verify dense checkpoint rank eligibility with the deployed Megatron groups."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    tp = args.tensor_parallel_size or world
    if tp <= 0 or world % tp:
        raise ValueError("tensor parallel size must divide GPU world size")
    records = []
    for expert_tp in sorted(set([1, tp])):
        mpu.initialize_model_parallel(tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=1, context_parallel_size=1,
            expert_model_parallel_size=1, expert_tensor_parallel_size=expert_tp)
        local = {"rank": dist.get_rank(), "tensor_rank": mpu.get_tensor_model_parallel_rank(),
                 "expert_data_rank": mpu.get_expert_data_parallel_rank()}
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
        eligible = [x["tensor_rank"] for x in gathered if x["expert_data_rank"] == 0]
        records.append({"expert_tensor_parallel_size": expert_tp, "ranks": gathered,
                        "legacy_checkpoint_eligible_tensor_ranks": sorted(eligible)})
        if expert_tp == tp:
            assert sorted(eligible) == list(range(tp)), records
        dist.barrier()
        mpu.destroy_model_parallel()
    if dist.get_rank() == 0:
        args.output.write_text(json.dumps({"passed": True, "real_gpu_process_group_check": True,
                                          "gpus": world, "tensor_parallel_size": tp,
                                          "qwen_training_executed": False, "checks": records}, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
