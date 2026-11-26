# Loss, sampling and cache review — 2026-11-26

## Finding

The default implementation is online, student-on-policy, sampled reverse-KL distillation with PPO clipping and episode weighting. It is not fixed-cache offline training. There is no universal OPD conformance standard; the checks below establish this implementation's contract, not equivalence to every algorithm named OPD.

## Actual objective

For a sampled student token `a_t` at student history `h_t`, with frozen teacher history `hT_t`, the custom advantage hook computes:

```text
A_t = detach(beta * w * (log q(a_t | hT_t) - log p_old(a_t | h_t)))
r_t = exp(log p_current(a_t | h_t) - log p_old(a_t | h_t))
loss_t = -min(r_t * A_t, clip(r_t, 0.8, 1.2) * A_t)
L = mean_over_episodes(sum(mask_t * loss_t) / sum(mask_t))
```

`beta` must be finite and positive; weights are finite in `[0,10]`. Empty supervised batches are rejected before optimizer update. Masked context and DP padding do not add episode token mass. The reviewed upstream reducer preserves episode denominators across turn splitting and microbatches. This is different from averaging all tokens globally or weighting each turn equally.

`finance_opd/advantages.py` supplies the detached token signal. `finance_opd/launch.py` selects the custom advantage hook, rollout logprobs, no advantage whitening, one update step per rollout, and no second built-in OPD term. The installed Slime `policy_loss_function` computes the importance ratio and clipping. A GRPO-named default option in upstream arguments does not turn this custom loss into group-relative task-reward RL. Native task scores influence CV admission/weights; they are not fed here as the final-answer GRPO advantage.

For a fixed prefix and frozen teacher context/weight independent of the sampled action, at `p_current=p_old`, the expected action gradient matches the local reverse-KL gradient `beta*w*KL(p_current || q)`. The regression test enumerates a **nonuniform** three-action behavior distribution and checks this equality using real float64 autograd. Sign, clipping, masking, unequal episode lengths, and simulated CP sizes 1/2/4/8 are also covered.

This is not an exact full-trajectory KL estimator. Histories are sampled/fixed during the local update; episode normalization, selection, clipping and privileged teacher context change the optimization problem. A fixed advantage becomes a local surrogate as parameters move away from the sampling policy. This review does not establish unbiased optimization under arbitrary stale replay, successful long-horizon credit assignment, or teacher correctness.

## On-policy versus offline

The inspected `slime/train.py` loop awaits generation, awaits actor training, and then synchronizes actor weights before the next generation. The adapter creates a fresh episode directory and queue request. The default launcher saves `rollout/slime-{rollout_id}.pt`; it does not enable an upstream debug-rollout loader. The queue worker rejects stale/orphan episode outputs instead of silently replaying them.

The student generates actions; the teacher scores those exact actions. In CV mode, teacher continuation and gain calculation precede an independent retry on the original student prefix. The initial stop proposal is not used as a teacher-conditioned training action. The retry avoids selecting the same training action based on guidance obtained after observing it, but does not remove bias from choosing which states to train on.

**Persisting an online rollout before training does not make it offline.** If a separate experiment repeatedly trains on a fixed cache after the student has changed, that is offline/replay distillation and increasingly off-policy relative to the updated student. The one-step clipped ratio is not a guarantee for arbitrary stale trajectory reuse. Such a replay experiment is not selected by the current default entry point.

`doctor --offline` skips a live teacher health probe. `WANDB_MODE=offline` stores experiment logs locally. `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` control weight loading. None of these changes the sampling policy.

## What is cached

| Content | Primary artifact/field | Meaning |
|---|---|---|
| Native trajectory | `episodes/<id>/native/events.jsonl`, native transcript/artifacts | Actual tool interactions and agent output |
| Student prompt and sampled IDs | `turns.jsonl.gz`: `prompt_ids`, `response_ids` | Original IDs, not re-encoded final text |
| Old student logprobs | `rollout_log_probs` | Probabilities returned when sampling these actions |
| Teacher logprobs | `teacher_log_probs` | Teacher likelihoods of the same student IDs |
| Teacher input | `teacher_prompt_ids`, `context_audit` | Exact teacher context, including privileged guidance where enabled |
| Supervision and provenance | `loss_mask`, `opd_weight`, `supervision_kind`, `weight_version`, prefix hash | Which tokens train, their weight and behavior version |
| Generation definition | `sampling_params`, `student_model`, `teacher_model`, `schema` | Added provenance for newly generated records |
| Trainer transport | `rollout/slime-*.pt` Sample dictionaries | Tokens, probabilities, masks, weights, versions and episode pointers passed upstream |

The four requested items are present for **teacher-scored training trajectories**. That claim does not apply to every screening record. Context-only/stop-mining screening may deliberately have `teacher_log_probs: null`, zero masks and zero weight. `samples_from_records` refuses to convert records with missing teacher scores into training samples. Missing teacher probabilities must not be replaced with zero. Native trajectories and detailed teacher contexts live beside, rather than fully inside, the `.pt` transport.

One historical real plain-OPD update was round-tripped: 28 serialized Samples exactly match 27 generation records plus one zero-mask DP padding Sample; 8,642 response tokens; IDs, student logprobs, teacher logprobs, masks, weights and versions agree. A recent screening cache had 404 records and no trainable/teacher-scored records. The latter is not evidence of a completed optimizer update.

## Repairs made in this review

1. Reject nonpositive/nonfinite OPD coefficients before launch and at the advantage boundary. Validate tensor dimensions, dtype, device and finiteness even with zero weights. Promote half-precision inputs before subtraction and detach the result.
2. Validate original IDs, probabilities, masks and weights at cache conversion, and reject mixed student weight versions inside one episode.
3. Require original server-sampled IDs in vLLM teacher continuations. Re-tokenizing decoded text with matching length and similar probabilities does **not** prove token identity. The server must return `token_id:N` entries; otherwise CV preflight/generation fails explicitly. SGLang's original-ID path remains in use.
4. Reject batches containing no positive supervision weight before optimizer update; zero gradient alone does not prevent AdamW weight decay.
5. Add cache schema/model/sampling provenance to new records; preserve existing historical records unchanged.
6. Make an offline doctor check distinguish passing local checks from live training readiness. Require Python 3.11+ for the APIs actually used.

## Remaining acceptance boundaries

The teacher must remain frozen and use a genuinely matching tokenizer. A served model name or a short echo probe alone does not prove the entire remote weight/tokenizer identity. A new vLLM deployment must pass the original-ID interface probe. Probability rescore checks can detect drift, but do not substitute for identity checks.

CV scores and evidence verification are estimates. Privileged evidence may be wrong, unavailable to the student at inference, or impossible to infer from its current history. Finite/correct loss and private-context separation do not prove that distilling this guidance improves independent task quality. That needs controlled plain-OPD/CV/no-guidance comparisons and held-out evaluation.

The prepared source still relies on the inspected local Slime and finance extensions. CPU tests and fixture services do not establish GPU distributed execution, current teacher availability, a new checkpoint, or improved benchmark scores.

Primary conceptual references: [On-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/), [Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649). Original-ID transport contract: [vLLM completion protocol](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/openai/completion/protocol/). Repository code and cache artifacts, rather than these references, support the implementation-specific findings above.
