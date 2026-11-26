"""Exercise upstream chunked TP vocabulary CE at the real maximum sequence length."""
import argparse
import json
import math
import os
from pathlib import Path


def context_parallel_size(explicit, *, config_path, sequence_length, tensor_parallel_size):
    """Use the owning run's topology even when an older worker omits the flag."""
    if not config_path:
        return explicit if explicit is not None else 1
    config = json.loads(Path(config_path).read_text())
    training = config["training"]
    cp = int(training.get("context_parallel_size", training.get("gpus", 8)))
    if (sequence_length != config["runtime"]["context_length"]
            or tensor_parallel_size != int(training.get("tensor_parallel_size", 1))
            or (explicit is not None and explicit != cp)):
        raise ValueError("logprob probe differs from the owning run's context/topology")
    return cp


def sequence_ranges(length, tp, cp, world, rank):
    """Megatron's mirrored CP chunks, with TP ranks sharing each local sequence."""
    if min(length, tp, cp, world) < 1 or world % (tp * cp) or not 0 <= rank < world:
        raise ValueError("invalid TP/CP probe topology")
    if cp == 1:
        return [(0, length)]
    if length % (2 * cp):
        raise ValueError("global probe sequence must divide into 2*CP chunks")
    chunk = length // (2 * cp)
    context_rank = (rank // tp) % cp
    return [(context_rank * chunk, (context_rank + 1) * chunk),
            ((2 * cp - context_rank - 1) * chunk, (2 * cp - context_rank) * chunk)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--context-parallel-size", type=int)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from torch.utils.checkpoint import checkpoint
    from slime.utils.ppo_utils import calculate_log_probs_and_entropy

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    world, rank = dist.get_world_size(), dist.get_rank()
    tp = args.tensor_parallel_size or world
    cp = context_parallel_size(args.context_parallel_size,
        config_path=os.environ.get("FINANCE_OPD_CONFIG"), sequence_length=args.sequence_length,
        tensor_parallel_size=tp)
    ranges = sequence_ranges(args.sequence_length, tp, cp, world, rank)
    local_length = sum(end - start for start, end in ranges)
    if tp <= 0 or world % tp or args.vocab_size % tp:
        raise ValueError("probe vocabulary must divide evenly over TP ranks")
    tp_group = None
    for start in range(0, world, tp):
        ranks = list(range(start, start + tp))
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group
    torch.manual_seed(1234)
    logits = torch.zeros((local_length, args.vocab_size // tp),
                         device="cuda", dtype=torch.float32, requires_grad=True)
    tokens = torch.cat([torch.arange(start, end, device="cuda", dtype=torch.long)
                        for start, end in ranges]) % args.vocab_size
    torch.cuda.reset_peak_memory_stats()

    def objective(values):
        logps, entropy = calculate_log_probs_and_entropy(
            values, tokens, tp_group, with_entropy=True, chunk_size=args.chunk_size,
            entropy_requires_grad=False)
        return -logps.mean() + 0.0 * entropy.mean()

    loss = checkpoint(objective, logits, use_reentrant=False)
    loss.backward()
    torch.cuda.synchronize()
    expected_loss = math.log(args.vocab_size)
    # Local CE normalization; this is not an end-to-end CP optimizer proof.
    target_is_first_local_class = int(tokens[0]) == (rank % tp) * (args.vocab_size // tp)
    expected_gradient = (1.0 / args.vocab_size - float(target_is_first_local_class)) / local_length
    observed_gradient = float(logits.grad[0, 0])
    assert math.isfinite(float(loss)) and abs(float(loss) - expected_loss) < 2e-4
    assert math.isclose(observed_gradient, expected_gradient, rel_tol=1e-4, abs_tol=1e-10)
    local = {"rank": rank, "tensor_rank": rank % tp,
             "context_parallel_rank": (rank // tp) % cp, "data_parallel_rank": rank // (tp * cp),
             "local_sequence_length": local_length, "global_sequence_ranges": ranges,
             "loss": float(loss.detach()),
             "sample_gradient": observed_gradient, "expected_gradient": expected_gradient,
             "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
             "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    if rank == 0:
        report = {"passed": True, "real_gpu_upstream_ce_backward": True,
                  "qwen_training_executed": False, "sequence_length": args.sequence_length,
                  "vocab_size": args.vocab_size, "chunk_size": args.chunk_size,
                  "gpus": world, "tensor_parallel_size": tp,
                  "context_parallel_size": cp, "local_sequence_length": local_length,
                  "recompute_loss_function": True, "zero_coefficient_entropy_detached": True, "ranks": gathered}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
