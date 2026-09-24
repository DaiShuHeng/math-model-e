#!/bin/bash
# v2 training queue: post-RNG-fix retrains + new bert-base text branch.
# All rows of the final ablation table come from this single code version.
set -e
cd $E_ROOT/solution
PY=python
export CUDA_VISIBLE_DEVICES=0
BASE=$E_ROOT/models/bert-base-uncased

run () {  # run <out_dir> <extra args...>
  local out=$1; shift
  for s in 2026 7 42; do
    echo "==== $out/s$s : $* ===="
    $PY -m src.train_q2 --seed "$s" --out "$out/s$s" "$@"
  done
}

# 5. base frozen + word-level aug + class weight   (NEW champion candidate)
run weights/q2_base_cw        --freeze-text 1 --cls-weight 1 --bert-dir "$BASE"
# 6. base frozen + NO aug + class weight           (aug ablation at base)
run weights/q2_base_cw_noaug  --freeze-text 1 --cls-weight 1 --no-aug --bert-dir "$BASE"
# 1. tiny frozen + aug                             (baseline, retrained)
run weights/q2_v2             --freeze-text 1
# 2. tiny fine-tune + aug
run weights/q2_ft_v2          --freeze-text 0
# 3. tiny fine-tune + aug + class weight           (tiny champion, retrained)
run weights/q2_ft_cw_v2       --freeze-text 0 --cls-weight 1
# 4. tiny fine-tune + NO aug                       (aug ablation at tiny)
run weights/q2_ft_noaug_v2    --freeze-text 0 --no-aug

echo "ALL DONE"
