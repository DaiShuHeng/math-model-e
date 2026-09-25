"""Explicit, fail-closed export. Review bundles are never labeled final submissions.
Usage: python -m src.build_submission --out /new/output [--review-only]
Final mode additionally needs --q2-checkpoint (revised, self-contained), and
--q3-normalizer (training-fitted statistics exported by the Q3 pipeline).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 50_000_000

def selected_files(root):
    trees = ['solution/src','solution/tests','solution/paper','solution/data/p1_features_v3',
             'math_model_e_optimized/code','math_model_e_optimized/data/alignment']
    exact = ['solution/data/p1_summary_v3.csv','审查与修订报告.md','修订运行指南.md',
             '本地核验结果.json','math_model_e_optimized/models/decision_calibration.json']
    exact += [f'math_model_e_optimized/models/p3_{s}/model.pt' for s in (2026,2027,2028)]
    result = []
    for name in trees:
        folder = root/name
        if not folder.is_dir(): raise FileNotFoundError(folder)
        result.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '历史原稿' not in p.parts and p.suffix != '.pyc')
    for name in exact:
        p=root/name
        if not p.is_file(): raise FileNotFoundError(p)
        result.append(p)
    return sorted(set(result))

def preflight(root):
    import csv
    errors=[]
    for rel, count in [('solution/data/p1_features_v3',100),('math_model_e_optimized/data/alignment/a1',100),('math_model_e_optimized/data/alignment/a4',20)]:
        pattern='*.npz' if 'features' in rel else '*.json'
        if len(list((root/rel).glob(pattern))) != count:errors.append(f'{rel}: expected {count}')
    summary=root/'solution/data/p1_summary_v3.csv'
    if not summary.exists(): errors.append('Q1 summary missing')
    else:
        rows=list(csv.DictReader(summary.open(encoding='utf-8-sig')))
        ids={r['sample_id'].replace('$_$','__') for r in rows}
        if len(rows)!=100 or len(ids)!=100 or ids!={p.stem for p in (root/'solution/data/p1_features_v3').glob('*.npz')}:errors.append('Q1 sample IDs/count do not match')
    if errors:raise ValueError('; '.join(errors))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--review-only',action='store_true')
    ap.add_argument('--q2-checkpoint',type=Path,nargs='+')
    ap.add_argument('--q3-normalizer',type=Path)
    args=ap.parse_args()
    out=args.out.resolve()
    if out.exists():raise FileExistsError(f'Will not overwrite: {out}')
    preflight(ROOT)
    files=selected_files(ROOT)
    missing=['New-protocol Q2 training and evaluation pending','Q3 portable train normalizer and raw-evidence timing validation pending']
    if not args.review_only:
        # Q3 currently recomputes normalization from train; accepting a random NPZ would
        # incorrectly certify standalone inference. Keep blocked until loader is wired.
        if not args.q2_checkpoint:raise ValueError('No revised Q2 checkpoints. Historical q2_base_cw cannot certify the new method.')
        from .checkpoints import load_revised
        for cp in args.q2_checkpoint:load_revised(cp)
        raise ValueError('Final export blocked: Q3 portable normalization/evidence verification is not yet completed. Use --review-only for an explicitly incomplete review bundle.')
    size=sum(p.stat().st_size for p in files)
    if size>LIMIT:raise ValueError(f'Export exceeds 50MB: {size} bytes; nothing silently skipped')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='math-e-build-',dir=out.parent) as tmp:
        stage=Path(tmp)/'bundle';stage.mkdir()
        manifest=[]
        for p in files:
            rel=p.relative_to(ROOT);dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
            manifest.append({'path':rel.as_posix(),'bytes':dst.stat().st_size,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
        (stage/'MANIFEST.json').write_text(json.dumps({'status':'REVIEW_ONLY_NOT_SUBMITTABLE','missing':missing,'files':manifest},ensure_ascii=False,indent=2))
        (stage/'请先阅读.txt').write_text('修订评审包，不能直接参赛提交。缺少修订后Q2训练权重/结果，Q3可移植归一化与原始证据核验未完成。旧实验未被冒充为新实验。\n')
        actual=sum(p.stat().st_size for p in stage.rglob('*') if p.is_file())
        if actual>LIMIT:raise ValueError(f'Including manifest exceeds limit: {actual}')
        shutil.move(str(stage),out)
    print(json.dumps({'out':str(out),'bytes':actual,'status':'REVIEW_ONLY_NOT_SUBMITTABLE'},ensure_ascii=False))

if __name__=='__main__':main()
