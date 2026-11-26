#!/usr/bin/env bash
set -euo pipefail
FINANCE_OPD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${FINANCE_OPD_ROOT}:${FINANCE_OPD_ROOT}/../SAfactory:${FINANCE_OPD_ROOT}/../slime${FINANCE_OPD_TEST_DEPS:+:${FINANCE_OPD_TEST_DEPS}}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="${FINANCE_OPD_ROOT}/.cache/inductor"
cd "${FINANCE_OPD_ROOT}"
exec "${PYTHON_BIN:-python3}" tests/run_validation.py "$@"
