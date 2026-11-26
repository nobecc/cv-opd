# Source release review — 2026-11-26

Prepared repository name: `cv-opd`. Distribution version: `0.2.0`. Import name: `finance_opd` for compatibility. The new source root is separate from the original runtime/archive root; no historical run directory was moved or rewritten.

## Reviewed and repaired

- Traced fresh student sampling, exact-token teacher scoring, serialized Sample conversion, advantage construction, upstream clipping/reduction, optimizer scheduling and weight synchronization.
- Validated the reverse-KL gradient sign with nonuniform-action CPU autograd, masks/group denominators, simulated CP partitions, invalid cache entries and numerical failures.
- Added strict original-ID requirements for vLLM teacher generation; kept original sampled student IDs/probabilities unchanged.
- Prevented an entirely unsupervised batch from reaching optimizer update; rejected invalid OPD coefficients and nonfinite tensors, including zero-weight cases.
- Added new-record provenance; distinguished offline prerequisite checks from live teacher readiness.
- Removed personal model paths, cluster mounts and internal image defaults from the publication tree. H-cluster utilities require explicit local mounts/images. Source examples declare the tool protocol and thinking mode.
- Updated dependency revisions/hashes and Python minimum version. Checked Python syntax, shell syntax, package metadata, explicit mount handling, and teacher-probe credential propagation without network calls.

## Publication contents

Include adapter sources, native tool bridge, runtime guard, scripts, tests, generic example configs, dependency manifest and review documentation. Exclude task/reference data, logs, runs, model weights, caches, local configs, credentials, dependency installations and runtime symlinks. The initial source release targets [nobecc/cv-opd](https://github.com/nobecc/cv-opd), branch `main`, at the project owner's request. Only the reviewed source, tests, generic configs and documentation are included.

The original Python module/environment-variable names are retained deliberately. The old project root still serves historical absolute paths and existing workers. New development/publication should use this source root; migrating a live controller and its run state is a separate operation.

## Validation verdict

Machine-readable results are in [REVIEW_RESULTS.json](REVIEW_RESULTS.json). They distinguish CPU math, real native tools with fixture model services, protocol preflight, historical cache evidence, and stages not executed. Local review artifacts contain detailed logs; private native trajectories are not published.

This release is prepared for **source review**, not certified as a fresh end-to-end GPU run. The stricter vLLM continuation path still needs a live deployment probe. Current teacher/judge availability, distributed collectives, optimizer execution under the new source tree, checkpoint conversion, and independent benchmark quality were not newly verified here.

External reproduction requires the recorded local Slime/finance extensions. Publishing this adapter alone does not supply those extensions, native assets, weights or access to the evaluator. There is no claim that vanilla upstream Slime is sufficient. Choose a project license separately before describing the repository as license-granted open source; no third-party license has been changed.

The machine-readable review records the validation stage before publication. GitHub commit history records publication separately; it does not change the GPU-validation verdict or add the subsequently discussed cache-replay training scheme.
