#!/usr/bin/env bash
# Standalone native-evaluator judge worker (see scripts/judge_worker.py).
set -euo pipefail
FINANCE_OPD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/usr/local/nvidia/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_IB_DISABLE=1
cd "${FINANCE_OPD_ROOT}"
exec python3 -B scripts/judge_worker.py "$@"
