# CV-OPD

Repository: [nobecc/cv-opd](https://github.com/nobecc/cv-opd).

Continuation-value guided on-policy distillation for financial research agents. This source project adapts Slime training to the Pi finance agent and optionally gives a frozen teacher evidence from bounded continuation probes. The student is trained on its own sampled actions.

The repository/distribution is named `cv-opd`; the Python import name remains `finance_opd` for compatibility with existing Slime hooks and saved configurations. This is a source release prepared from the standalone `finance_opd` project. Historical runs and services remain in their original location; they are not bundled here.

## What the code trains

- Plain OPD: sample a fresh student trajectory, obtain teacher probabilities for exactly those sampled token IDs, and update the student with a sampled reverse-KL signal.
- CV-OPD: probe a candidate stopping state, estimate continuation gain using a common writer and task evaluator, verify new evidence, then condition the teacher on admitted guidance. Sample a **fresh student retry from the original prefix**, and distill the teacher's probabilities on the retry tokens. The initial stop sample and context-only turns have zero loss masks.
- The default training loop generates new samples after weight synchronization. It is **online, on-policy distillation with disk caches**, not fixed-dataset offline distillation. `doctor --offline`, `WANDB_MODE=offline`, and offline weight loading do not select an offline learning algorithm.

## Loss

For a student-sampled token, the teacher scores that same token under its own conditioning context:

```text
A_t = detach(opd_coef * weight * (teacher_logp_t - old_student_logp_t))
ratio_t = exp(current_student_logp_t - old_student_logp_t)
loss_t = -min(ratio_t * A_t, clip(ratio_t, 0.8, 1.2) * A_t)
loss = mean_over_episodes(sum(mask_t * loss_t) / sum(mask_t))
```

`old_student_logp` is the original sampling probability and must not be overwritten by a later forward pass. The coefficient must be finite and positive; weights must be finite in `[0,10]`. Empty supervised batches are rejected. Tool observations, context-only turns and DP padding are excluded from the loss. Episode denominators are preserved across turn splitting and microbatches.

`finance_opd/advantages.py` supplies the detached signal; the Slime policy loss applies the ratio, clipping and reduction. The launcher disables advantage whitening and a second built-in OPD term. Native task scores determine CV admission/weights rather than serving as a GRPO outcome advantage. This is a sampled reverse-KL surrogate, not hard-label SFT or full-vocabulary KL.

For a fixed prefix and frozen teacher context/weight independent of the sampled action, its expected action gradient at the sampling policy matches local reverse-KL. Clipping, state selection and policy drift mean this is not an exact full-trajectory KL objective or a guarantee for arbitrary stale replay. Teacher guidance and continuation-value estimates still require independent task-quality evaluation.

## Cached trajectories

Each episode retains the native trajectory alongside token-level training records:

| Content | Artifact or field |
|---|---|
| Native trajectory and tool interactions | `episodes/<id>/native/events.jsonl`, native transcript/artifacts |
| Original student prompt and sampled IDs | `turns.jsonl.gz`: `prompt_ids`, `response_ids` |
| Student probabilities at sampling time | `rollout_log_probs` |
| Teacher probabilities of those same student tokens | `teacher_log_probs` |
| Exact teacher context and guidance provenance | `teacher_prompt_ids`, `context_audit` |
| Supervision, model version and sampling definition | `loss_mask`, `opd_weight`, `supervision_kind`, `weight_version`, prefix hash, `sampling_params`, model identifiers |
| Samples passed to the trainer | `rollout/slime-*.pt`, including episode pointers |

IDs, probabilities and masks must stay aligned; text re-tokenization must not replace original sampled IDs. Screening-only records may have no teacher scores and zero loss masks/weights. Records with missing teacher scores cannot be converted into training samples, and missing probabilities must not be filled with zero.

The default loop uses a fresh batch for one optimizer step, synchronizes student weights and samples again. Caches support training handoff, debugging and reproducibility; they are not automatically replayed across later updates. Repeated historical replay would require a separate sampling and off-policy-control design. A frozen teacher's cached score is reusable only for the same exact teacher context, token sequence and scoring definition.

## Dependencies and layout

Python **3.11+** is required (`asyncio.timeout`). Use an editable source checkout:

```text
workspace/
  cv-opd/
  slime/
  SAfactory/
  pi-finance/
  eval_finance/     # native evaluation for CV supervision
```

```bash
cd cv-opd
python -m pip install -e .
cp configs/qwen35_9b.json configs/train.local.json
```

The editable install supplies only this adapter's lightweight dependencies. Install compatible PyTorch, Ray, SGLang, Megatron, Transformers, the Pi Node runtime, and the native finance evaluator separately. GPU/container compatibility must be checked on the target cluster.

**The inspected Slime and finance repositories contain local extensions. A stock upstream checkout is not guaranteed compatible.** In particular, training needs `Sample.group_id`, episode-level `group_mask_sums`, teacher logprob transport, and the installed Qwen protocol/mask implementation. A Git commit alone does not capture the required local patches; compatible dependency implementations must be supplied separately. Those repositories and their assets are not redistributed by this package. Scripts/runtime assets live beside the package, so a bare wheel is not a standalone training system.

In the local config set exact model/tokenizer snapshots, teacher URL, runtime paths and resource counts. The checked-in `/models/student` and `/models/teacher` values are placeholders. Set thinking mode explicitly and use `runtime.tool_protocol=qwen3_xml_v1`. Tokenizer compatibility means the token-to-ID mapping and tokenization semantics agree, not merely equal vocabulary size.

Keep dataset/output paths inside this source tree. Task data may be exported from the configured finance harness:

```bash
python -m finance_opd.data --workspace .. --output data/current
python -m finance_opd.launch doctor --config configs/train.local.json --offline
python -m finance_opd.launch train --config configs/train.local.json --dry-run
```

The first command requires existing benchmark assets. `doctor --offline` checks local prerequisites only and always reports `training_ready: false`; a passing dry run is not a training execution. Before an actual run, verify the teacher's exact supplied-token scoring interface, tokenizer, original generated token IDs if using CV, the native tool protocol, and the intended parallel layout. After those checks, the training entry point is `scripts/train.sh --config configs/train.local.json`.

For CV, start from `configs/cv_opd_qwen38_27b.json`, supply private scoring references and a working evaluator, and run the existing `scripts/preflight_cv.py` before submitting. References stay outside agent inputs. Teacher-only guidance stays outside the student's retry prefix. Missing teacher/judge results are failures or pending records, never fabricated zero scores.

SGLang is the default teacher transport. An external vLLM teacher requires `teacher.transport=vllm`, an exact served name in `teacher.model`, and a local `teacher.tokenizer_path`. Scoring needs `/v1/completions` with token-ID prompts and `prompt_logprobs`; CV generation additionally requires `return_tokens_as_token_ids=true` to return actual sampled IDs. Text re-tokenization is forbidden. API credentials are read from `FINANCE_OPD_TEACHER_API_KEY` and must not be committed.

## Running tests

With the compatible sibling repositories, CPU PyTorch/Transformers, tokenizer and Node runtime installed:

```bash
export FINANCE_OPD_CONFIG="$PWD/configs/train.local.json"
export FINANCE_OPD_TEST_TOKENIZER=/path/to/student-tokenizer
export PYTHONPATH="$PWD:$PWD/tests:$PWD/../SAfactory:$PWD/../slime"
export PYTHONDONTWRITEBYTECODE=1 TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1
mkdir -p runs validation
python -m unittest discover -s tests -p 'test_*.py' -v
```

Tests cover actual CPU autograd, loaded upstream loss/reduction functions, simulated CP partitions, cache integrity, and native Pi tools with deterministic model HTTP fixtures. They do not establish actual GPU collective behavior or model quality. Inspect skips rather than treating an `OK` line as complete integration acceptance. `scripts/validate.sh` additionally audits the locally exported complete dataset and is not a dependency-free unit-test command.

CV effectiveness and context-guidance reliability must be measured on held-out tasks; passing local tests does not establish a successful GPU training run or improved model quality.

## Optional cluster utilities

The `submit_trial`, `run_submitted_trial`, `watch_trial` and `finalize_one_update` scripts are H-cluster-specific utilities, not portable launchers. Configure `job.mounts`, `job.teacher_image`, `job.student_image`, and `job.verification_image` in an ignored local config; do not rely on someone else's mounts or images. Review live resource routing and controller ownership before using them. Only one controller should submit a given run.

Run data, references, model weights, logs, local configs, API credentials, and dependency/runtime directories are excluded from Git. Third-party code/data/model licenses remain applicable; this source release does not assign a new project license.
