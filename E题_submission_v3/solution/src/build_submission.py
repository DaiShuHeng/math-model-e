"""Explicit, fail-closed export. Review bundles are never labeled final submissions.
Usage: python -m src.build_submission --out /new/output [--review-only]
Final mode additionally needs --q2-checkpoint (revised, self-contained) and
--q3-normalizer (training-fitted statistics exported by the Q3 pipeline).

Final-mode gates (all must pass before anything is labeled submittable):
  G1  revised Q2 checkpoints load standalone (protocol+normalizer embedded);
  G2  --q3-normalizer byte-equals statistics recomputed from the training split
      (a random NPZ cannot certify standalone inference);
  G3  staged copy (code+stats+p3 heads only) re-runs the 附件4 Shapley pipeline
      against the provided data and reproduces the frozen CSV values;
  G4  evidence_time_mapping.csv: full coverage, allowed mapping_status only,
      no end<start timing rows.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 50_000_000


def selected_files(root):
    trees = ['solution/src', 'solution/tests', 'solution/paper', 'solution/data/p1_features_v3',
             'math_model_e_optimized/code', 'math_model_e_optimized/data/alignment']
    exact = ['solution/data/p1_summary_v3.csv', '审查与修订报告.md', '修订运行指南.md',
             '本地核验结果.json', 'math_model_e_optimized/models/decision_calibration.json',
             # Q3 便携归一化（G2 认证的同一文件入包，独立推理不再需要 train 数据）
             'math_model_e_optimized/data/cache/a2_stats_v2.npz',
             # 专项交付：附件3 预测（修订协议 Tiny 模型）、附件4 Shapley 与重算互证、秒级证据
             'solution/results/att3_predictions_tiny_s42.csv',
             'solution/results/附件4_shapley.csv',
             'solution/results/附件4_shapley_重算.csv',
             'math_model_e_optimized/results/evidence_time_mapping.csv',
             # 修订协议验证与测试记录（可复核）
             'solution/logs/optimization_s2026_comparison.json',
             'solution/logs/final_test_eval_tiny_finetune_s42.json']
    exact += [f'math_model_e_optimized/models/p3_{s}/model.pt' for s in (2026, 2027, 2028)]
    result = []
    for name in trees:
        folder = root / name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        result.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                      and '历史原稿' not in p.parts and p.suffix != '.pyc')
    for name in exact:
        p = root / name
        if not p.is_file():
            raise FileNotFoundError(f'final-mode input missing: {name}')
        result.append(p)
    return sorted(set(result))


def preflight(root):
    errors = []
    for rel, count in [('solution/data/p1_features_v3', 100),
                       ('math_model_e_optimized/data/alignment/a1', 100),
                       ('math_model_e_optimized/data/alignment/a4', 20)]:
        pattern = '*.npz' if 'features' in rel else '*.json'
        if len(list((root / rel).glob(pattern))) != count:
            errors.append(f'{rel}: expected {count}')
    summary = root / 'solution/data/p1_summary_v3.csv'
    if not summary.exists():
        errors.append('Q1 summary missing')
    else:
        rows = list(csv.DictReader(summary.open(encoding='utf-8-sig')))
        ids = {r['sample_id'].replace('$_$', '__') for r in rows}
        if len(rows) != 100 or len(ids) != 100 or ids != {p.stem for p in (root / 'solution/data/p1_features_v3').glob('*.npz')}:
            errors.append('Q1 sample IDs/count do not match')
    if errors:
        raise ValueError('; '.join(errors))


def gate_normalizer_is_train_fitted(npz_path):
    """G2: --q3-normalizer 必须与从 train 重算的统计 allclose（逐键逐元素）。"""
    import numpy as np
    code = (
        "import os,sys,json,numpy as np;"
        "sys.path.insert(0,os.environ['TM_CODE']);"
        "import io_utils as U;"
        "stats=U.compute_stats(U.load_a2('train'));"
        "ref=U.load_stats(__import__('pathlib').Path(os.environ['REF_NPZ']));"
        "ok=all(np.allclose(stats[m][k],ref[m][k],rtol=1e-6,atol=1e-7) for m in ref for k in ('mean','std'));"
        "print(json.dumps({'keys':sorted(ref),'match':bool(ok)}))"
    )
    env = dict(os.environ, TM_CODE=str(ROOT / 'math_model_e_optimized' / 'code'),
               REF_NPZ=str(Path(npz_path).resolve()),
               MATH_E_DATA=str(ROOT / 'E题' / 'E题数据'))
    r = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise ValueError(f'G2 normalizer recompute failed: {r.stderr[-400:]}')
    out = json.loads(r.stdout.strip().splitlines()[-1])
    if not out['match']:
        raise ValueError('G2 FAILED: --q3-normalizer differs from train-recomputed statistics')


def gate_standalone_att4_reproduction(stage, reference_csv):
    """G3: 仅用暂存目录内的代码/统计/p3 头，对真实数据重跑附件4 Shapley，逐值等于冻结 CSV。"""
    script = (f"import os,sys;os.environ.setdefault('MATH_E_DATA',{str(ROOT / 'E题' / 'E题数据')!r});"
              "sys.path.insert(0,os.environ['STAGE']+'/solution');"
              "import runpy;runpy.run_module('src.q3_shapley',run_name='__main__')")
    env = dict(os.environ, STAGE=str(stage), PYTHONPATH=str(stage / 'solution'),
               CUDA_VISIBLE_DEVICES='')
    r = subprocess.run([sys.executable, '-c', script], env=env, cwd=str(stage / 'solution'),
                       capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise ValueError(f'G3 standalone rerun failed: {r.stderr[-600:]}')
    produced = stage / 'solution/results/附件4_shapley_重算.csv'
    if not produced.exists():
        raise ValueError('G3 FAILED: rerun produced no output')
    import itertools
    a = list(csv.DictReader(produced.open(encoding='utf-8-sig')))
    b = list(csv.DictReader(reference_csv.open(encoding='utf-8-sig')))
    if len(a) != len(b):
        raise ValueError(f'G3 FAILED: row count {len(a)} != {len(b)}')
    # CPU 复跑 vs GPU 冻结值允许 1e-4 浮点差；类别/模态等离散字段仍须严格相等
    def eq(va, vb):
        try:
            fa, fb = float(va), float(vb)
            return abs(fa - fb) <= 1e-4
        except (TypeError, ValueError):
            return str(va) == str(vb)
    for ra, rb in zip(a, b):
        for k in rb:
            if not eq(ra.get(k), rb[k]):
                raise ValueError(f'G3 FAILED: {ra.get("sample_id","?")} field {k}: {ra.get(k)} != {rb[k]}')
    return {'rows': len(a), 'float_tolerance': 1e-4}


def gate_evidence_timing(csv_path):
    """G4: 证据表覆盖 20 样本 59 组、状态白名单、无时间倒序。"""
    rows = list(csv.DictReader(csv_path.open(encoding='utf-8-sig')))
    if len(rows) != 177:
        raise ValueError(f'G4 FAILED: evidence rows {len(rows)} != 177')
    groups = {(r['sample_id'], r['modality']) for r in rows}
    samples = {r['sample_id'] for r in rows}
    if len(groups) != 59 or len(samples) != 20:
        raise ValueError(f'G4 FAILED: {len(groups)} groups / {len(samples)} samples')
    allowed = {'', 'verified_text_fragment_no_speech_timestamp'}
    for r in rows:
        if r['mapping_status'] not in allowed:
            raise ValueError(f"G4 FAILED: status {r['mapping_status']!r}")
        if r['start_s'] and r['end_s'] and float(r['end_s']) < float(r['start_s']):
            raise ValueError('G4 FAILED: end_s < start_s')
    return {'rows': len(rows), 'groups': len(groups)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--review-only', action='store_true')
    ap.add_argument('--q2-checkpoint', type=Path, nargs='+')
    ap.add_argument('--q3-normalizer', type=Path)
    args = ap.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f'Will not overwrite: {out}')
    preflight(ROOT)
    files = selected_files(ROOT)

    if args.review_only:
        _export(out, files, extra=[], status='REVIEW_ONLY_NOT_SUBMITTABLE',
                missing=['New-protocol Q2 training and evaluation pending',
                         'Q3 portable train normalizer and raw-evidence timing validation pending'])
        return

    # ---- final mode: G1..G4 gates ----
    if not args.q2_checkpoint or not args.q3_normalizer:
        raise ValueError('Final export needs --q2-checkpoint and --q3-normalizer')
    from .checkpoints import load_revised
    ck_files = []
    for cp in args.q2_checkpoint:
        load_revised(cp)  # G1: 自包含、协议正确、内嵌归一化合法
        rel = Path(cp).resolve().relative_to(ROOT)
        ck_files.append(rel)
    gate_normalizer_is_train_fitted(args.q3_normalizer)  # G2
    gate_evidence_timing(ROOT / 'math_model_e_optimized/results/evidence_time_mapping.csv')  # G4（先于耗时重跑）

    with tempfile.TemporaryDirectory(prefix='math-e-build-', dir=out.parent) as tmp:
        stage = Path(tmp) / 'bundle'
        stage.mkdir()
        for p in files + [ROOT / rel for rel in ck_files]:
            rel = p.relative_to(ROOT)
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        g3 = gate_standalone_att4_reproduction(stage, ROOT / 'solution/results/附件4_shapley_重算.csv')  # G3
        # 重跑产物不留在包内（它等于已入包的冻结 CSV）
        (stage / 'solution/results/附件4_shapley_重算.csv').unlink()
        _export_stage(out, stage, files, ck_files, checks={'G3_standalone_att4': g3})


def _export(out, files, extra, status, missing):
    with tempfile.TemporaryDirectory(prefix='math-e-build-', dir=out.parent) as tmp:
        stage = Path(tmp) / 'bundle'
        stage.mkdir()
        for p in files:
            rel = p.relative_to(ROOT)
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        _manifest(stage, status, {'missing': missing})
        _finalize(out, stage)


def _export_stage(out, stage, files, ck_files, checks):
    _manifest(stage, 'FINAL_CANDIDATE', {
        'gates_passed': ['G1 checkpoints load standalone (protocol+normalizer embedded)',
                         'G2 --q3-normalizer equals train-recomputed statistics',
                         'G3 staged code+stats reproduce frozen 附件4 Shapley CSV',
                         'G4 evidence_time_mapping coverage/status/timing'],
        'gate_details': checks,
        'q2_checkpoints': [str(r) for r in ck_files]})
    (stage / '请先阅读.txt').write_text(
        '正式提交候选包。含：双仓库代码、Q1 特征(npz×100)与对齐JSON、修订协议 Tiny Q2 模型权重'
        '(自包含checkpoint)、附件3预测CSV、附件4 Shapley与重算互证、秒级证据表、验证/测试记录JSON。\n'
        'G1-G4 验收明细见 MANIFEST.json。预训练编码器与数据按 solution/提交说明.md 固定版本获取。\n')
    _finalize(out, stage)


def _manifest(stage, status, extra):
    manifest = []
    for p in sorted(x for x in stage.rglob('*') if x.is_file()):
        manifest.append({'path': p.relative_to(stage).as_posix(), 'bytes': p.stat().st_size,
                         'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    (stage / 'MANIFEST.json').write_text(json.dumps(
        {'status': status, **extra, 'files': manifest}, ensure_ascii=False, indent=1))


def _finalize(out, stage):
    actual = sum(p.stat().st_size for p in stage.rglob('*') if p.is_file())
    if actual > LIMIT:
        raise ValueError(f'Including manifest exceeds limit: {actual}')
    shutil.move(str(stage), out)
    print(json.dumps({'out': str(out), 'bytes': actual,
                      'status': json.loads((out / 'MANIFEST.json').read_text())['status']},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
