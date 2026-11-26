"""Tiny real-device check of the deployed Megatron hybrid optimizer (not Qwen training)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, required=True)
    args = parser.parse_args()
    import torch
    from megatron.core.optimizer import Adam, CPUAdam
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    torch.set_num_threads(2)
    params = [torch.nn.Parameter(torch.zeros(4096, device="cuda", dtype=torch.bfloat16)) for _ in range(4)]
    references = [torch.nn.Parameter(torch.zeros(4096)) for _ in params]
    options = dict(lr=0.01, weight_decay=0.1, betas=(0.9, 0.98), eps=1e-8)
    optimizer = HybridDeviceOptimizer(params, offload_fraction=args.fraction,
        cpu_optimizer_cls=CPUAdam, gpu_optimizer_cls=Adam,
        overlap_cpu_optimizer_d2h_h2d=True, pin_cpu_grads=True, pin_cpu_params=True,
        param_update_in_fp32=True, bias_correction=True, fused=True, **options)
    reference = torch.optim.AdamW(references, **options)
    errors = []
    for step in range(2):
        for i, (param, expected) in enumerate(zip(params, references)):
            grad = torch.full_like(param, (i + 1) * 0.125 * (step + 1))
            param.decoupled_grad = grad.float()
            expected.grad = grad.float().cpu()
        optimizer.step()
        torch.cuda.synchronize()
        reference.step()
        for param, expected in zip(params, references):
            actual = param.detach().float().cpu()
            target = expected.detach().to(torch.bfloat16).float()
            torch.testing.assert_close(actual, target, rtol=0, atol=2e-4)
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            errors.append((actual - target).abs().max().item())
    cpu_numel = sum(p.numel() for group in optimizer.cpu_param_groups for p in group["params"])
    gpu_numel = sum(p.numel() for group in optimizer.gpu_param_groups for p in group["params"])
    assert cpu_numel > 0 and gpu_numel > 0, (cpu_numel, gpu_numel)
    report = {"passed": True, "real_gpu_optimizer_steps": 2, "qwen_training_executed": False,
              "cpu_parameter_elements": cpu_numel, "gpu_parameter_elements": gpu_numel,
              "max_abs_error_against_adamw": max(errors), "offload_fraction": args.fraction,
              "gpu_optimizer": Adam.__module__ + "." + Adam.__name__,
              "cpu_optimizer": CPUAdam.__module__ + "." + CPUAdam.__name__}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
