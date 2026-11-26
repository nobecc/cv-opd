#!/usr/bin/env bash
set -euo pipefail
FINANCE_OPD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${FINANCE_OPD_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
cd "${FINANCE_OPD_ROOT}"
exec "${PYTHON_BIN:-python3}" -m finance_opd.launch teacher "$@"
