#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MATH_E_DATA="${MATH_E_DATA:-$REPO_ROOT/E题/E题数据}"
export MATH_E_BERT="${MATH_E_BERT:-$REPO_ROOT/models/bert-tiny}"
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/solution/weights/distill_v4}"
cd "$REPO_ROOT"
if [[ ! -f "$RUN_ROOT/teacher/teacher_audit.json" ]]; then
  "$PYTHON_BIN" scripts/prepare_distillation.py --out "$RUN_ROOT/teacher"
fi
cd "$REPO_ROOT/solution"
"$PYTHON_BIN" -m src.distill_experiments --out "$RUN_ROOT/runs" \
  --teacher-cache "$RUN_ROOT/teacher/teacher_train.npz" --bert-dir "$MATH_E_BERT" "$@"
