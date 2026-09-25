#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MATH_E_DATA="${MATH_E_DATA:-$REPO_ROOT/E题/E题数据}"
export MATH_E_BERT="${MATH_E_BERT:-$REPO_ROOT/models/bert-tiny}"
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$REPO_ROOT/solution"
"$PYTHON_BIN" -m src.preflight --bert-dir "$MATH_E_BERT"
"$PYTHON_BIN" -m src.optimize_valid --bert-dir "$MATH_E_BERT" "$@"
