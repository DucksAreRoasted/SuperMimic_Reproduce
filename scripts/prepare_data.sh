#!/usr/bin/env bash
# Download motions and retarget G1 clips onto R1 and K1.
# Run from anywhere on the server, after pip install -e ".[data]" in this repo.
#
#   bash scripts/prepare_data.sh
#
# Datasets come from ModelScope, not Hugging Face.
#   bash scripts/prepare_data.sh                  # LAFAN1, then retarget to R1/K1
#   SKIP_BONES=0 bash scripts/prepare_data.sh     # also download and filter BONES-SEED
#   RETARGET_BONES=1 SKIP_BONES=0 bash scripts/prepare_data.sh
#
# Knobs: JOBS (8), RETARGET_LAFAN (1), SKIP_BONES (1), RETARGET_BONES (0).
# Optional: MODELSCOPE_TOKEN for higher ModelScope rate limits.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
JOBS="${JOBS:-8}"
RETARGET_LAFAN="${RETARGET_LAFAN:-1}"
RETARGET_BONES="${RETARGET_BONES:-0}"

echo "== LAFAN1 =="
"$PY" scripts/download_datasets.py --dataset lafan1 --dest "$ROOT/data"

if [[ "${RETARGET_LAFAN}" == "1" ]]; then
  echo "== retarget LAFAN1 -> r1, k1 =="
  "$PY" scripts/retarget.py \
    --input "$ROOT/data/lafan1/g1" \
    --targets r1,k1 \
    --out "$ROOT/data/retarget/lafan1" \
    --jobs "$JOBS"
fi

SKIP_BONES="${SKIP_BONES:-1}"
if [[ "${SKIP_BONES}" == "0" ]]; then
  echo "== BONES-SEED (ModelScope) =="
  "$PY" scripts/download_datasets.py --dataset bones --dest "$ROOT/data"
  echo "== filter BONES-SEED =="
  "$PY" scripts/filter_bones.py \
    --archive "$ROOT/data/bones/g1.tar.gz" \
    --out "$ROOT/data/bones/kept.txt" \
    --rejected-out "$ROOT/data/bones/rejected.tsv" \
    --extract "$ROOT/data/bones/g1_filtered" \
    --feasibility
  if [[ "${RETARGET_BONES}" == "1" ]]; then
    echo "== retarget BONES-SEED -> r1, k1 =="
    "$PY" scripts/retarget.py \
      --input "$ROOT/data/bones/g1_filtered" \
      --targets r1,k1 \
      --out "$ROOT/data/retarget/bones" \
      --jobs "$JOBS"
  else
    echo "BONES clips are filtered in data/bones/g1_filtered."
    echo "Retarget them with RETARGET_BONES=1 (this is much larger than LAFAN1)."
  fi
else
  echo "SKIP_BONES=1, so BONES-SEED is not downloaded."
  echo "Download it with: SKIP_BONES=0 bash scripts/prepare_data.sh"
fi

echo "done."
echo "  LAFAN1 G1 csv:     data/lafan1/g1"
echo "  LAFAN1 R1/K1 npz:  data/retarget/lafan1/{r1,k1}"
echo "  BONES filtered:    data/bones/g1_filtered   (only if SKIP_BONES=0)"
