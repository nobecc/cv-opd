"""Small weighted extension of Slime's sampled reverse-KL OPD signal."""
from .protocol import finite


def post_process_weights(args, samples, **kwargs):
    # Slime's scalar transport column carries named distillation weights only.
    # No native task reward is synthesized or used as a GRPO outcome signal.
    raw = [sample.reward for sample in samples]
    weights = [finite(value["opd_weight"], "opd weight") for value in raw]
    if any(not 0 <= w <= 10 for w in weights):
        raise ValueError("opd weight outside [0,10]")
    if not any(weight > 0 and (getattr(sample, "loss_mask", None) is None or any(sample.loss_mask))
               for sample, weight in zip(samples, weights)):
        raise ValueError("OPD batch has no positive masked supervision; skip it before optimizer update")
    # Upstream metrics sum both columns. Keep named dictionaries on Sample/audit,
    # but pass numeric weights through raw_reward as well as rewards.
    return list(weights), weights


def compute_advantages(args, batch):
    import torch

    if args.use_opd or args.normalize_advantages or not args.use_rollout_logprobs:
        raise ValueError("weighted OPD requires rollout logprobs, no whitening, and no second built-in OPD term")
    coefficient = finite(args.opd_kl_coef, "opd coefficient")
    if coefficient <= 0:
        raise ValueError("opd coefficient must be positive")
    old = batch["rollout_log_probs"]
    teacher = batch["teacher_log_probs"]
    weights = batch["rewards"]
    if not (len(old) == len(teacher) == len(weights)):
        raise ValueError("OPD batch columns differ in length")
    advantages = []
    for student_lp, teacher_lp, weight in zip(old, teacher, weights):
        if not isinstance(student_lp, torch.Tensor) or not isinstance(teacher_lp, torch.Tensor):
            raise ValueError("student and teacher logprobs must be tensors")
        if student_lp.ndim != 1 or student_lp.shape != teacher_lp.shape:
            raise ValueError("teacher and student logprobs differ in shape")
        if not student_lp.is_floating_point() or not teacher_lp.is_floating_point():
            raise ValueError("student and teacher logprobs must be floating point")
        if student_lp.device != teacher_lp.device:
            raise ValueError("student and teacher logprobs must share a device")
        # A zero loss mask/weight does not neutralize NaN (0 * NaN is NaN).
        # Validate at the tensor boundary as well as the HTTP/cache boundary.
        if not torch.isfinite(student_lp).all() or not torch.isfinite(teacher_lp).all():
            raise ValueError("student and teacher logprobs must be finite")
        weight = finite(weight, "opd weight")
        if not 0 <= weight <= 10:
            raise ValueError("opd weight outside [0,10]")
        # Do not subtract near-equal probabilities in BF16/FP16. Preserve FP64
        # for numerical checks; production transport normally provides FP32.
        dtype = torch.promote_types(torch.float32, torch.promote_types(student_lp.dtype, teacher_lp.dtype))
        advantage = (coefficient * weight * (teacher_lp.to(dtype) - student_lp.to(dtype))).detach()
        if not torch.isfinite(advantage).all():
            raise ValueError("OPD advantage overflowed")
        advantages.append(advantage)
    batch["advantages"] = advantages
    batch["returns"] = [a.clone() for a in advantages]
