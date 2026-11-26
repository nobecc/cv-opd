#!/usr/bin/env bash
set -euo pipefail
FINANCE_OPD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/usr/local/nvidia/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${FINANCE_OPD_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NODE_RANK="${NODE_RANK:-0}" NODE_COUNT="${NODE_COUNT:-1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
if [[ "${PROC_PER_NODE:-8}" -lt 8 ]]; then
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
else
  unset NCCL_IB_DISABLE
fi
export PI_NODE="${FINANCE_OPD_ROOT}/.runtime/node24"
export JITI_FS_CACHE=0
cd "${FINANCE_OPD_ROOT}"
exec python3 -B -m finance_opd.cluster_worker "$@"
