#!/usr/bin/env bash
# One shot: create a venv, install the ModelScope downloader, download data, retarget.
#
#   bash scripts/一键准备数据.sh
#   SKIP_BONES=0 bash scripts/一键准备数据.sh
#   SKIP_BONES=0 RETARGET_BONES=1 JOBS=16 bash scripts/一键准备数据.sh
#
# LAFAN1 and BONES-SEED both come from ModelScope. BONES is skipped unless
# SKIP_BONES=0. See docs/服务器运行代码指南.md.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[data]"

export PYTHON="${ROOT}/.venv/bin/python"
exec bash "${ROOT}/scripts/prepare_data.sh"
