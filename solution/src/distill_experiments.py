"""Bounded, resumable TRAIN/VALID experiment suite. Does not evaluate TEST/A3/A4."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

VARIANTS = {'baseline': 0.0, 'kd03': 0.3, 'kd10': 1.0}
SEEDS = (2026, 7, 42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--teacher-cache', type=Path, required=True)
    ap.add_argument('--bert-dir', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seeds', default='2026,7,42')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    seeds = list(map(int, args.seeds.split(',')))
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(SEEDS):
        ap.error('seeds must be a unique subset of 2026,7,42')
    out = args.out.resolve()
    teacher = args.teacher_cache.resolve()
    from .data_adapter import load_training_splits
    from .distillation import split_fingerprint
    import numpy as np
    splits=load_training_splits()
    fingerprints={s:split_fingerprint(d) for s,d in splits.items()}
    del splits
    with np.load(teacher,allow_pickle=False) as cache:
        if str(cache['split'])!='train' or str(cache['fingerprint'])!=fingerprints['train']:
            raise ValueError('Teacher cache does not match current training data')
    # Full candidate set fixed before first run; --seeds only schedules subsets.
    source_hash = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        source_hash.update(p.name.encode()); source_hash.update(p.read_bytes())
    plan = {'variants': VARIANTS, 'seeds': SEEDS, 'epochs': args.epochs,
            'teacher_sha256': hashlib.sha256(teacher.read_bytes()).hexdigest(),
            'source_sha256': source_hash.hexdigest(), 'data_fingerprints': fingerprints,
            'encoder_sha256': hashlib.sha256((args.bert_dir/'pytorch_model.bin').read_bytes()).hexdigest(),
            'selection': 'mean(clean MacroF1, continuous TAV r20 MacroF1)',
            'distillation': 'clean unchanged views with correct teacher class only; temperature 2',
            'scope': 'TRAIN/VALID only; diagnostic grids do not change early stopping'}
    plan = json.loads(json.dumps(plan))
    out.mkdir(parents=True, exist_ok=True)
    path = out/'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Existing plan differs; use a NEW output directory')
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    for seed in seeds:
        for name, weight in VARIANTS.items():
            run = out/f'{name}_s{seed}'
            if (run/'metrics.json').exists() and (run/'best.pt').exists():
                print('Completed; skipping:', run, flush=True)
                continue
            if run.exists():
                raise FileExistsError(f'Incomplete run {run}; preserve it and use a new output root. '
                                     'Completed runs are resumable, interrupted optimizer state is not.')
            cmd = [sys.executable, '-m', 'src.train_q2', '--out', str(run), '--seed', str(seed),
                   '--epochs', str(args.epochs), '--bert-dir', str(args.bert_dir.resolve()),
                   '--freeze-text', '0', '--ft-lr', '0.00001', '--cls-weight', '1',
                   '--threads', str(args.threads), '--teacher-cache', str(teacher),
                   '--distill-weight', str(weight)]
            print('Starting', name, seed, flush=True)
            with (out/f'{name}_s{seed}.log').open('w') as log:
                with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
                    for line in proc.stdout:
                        print(line, end='', flush=True); log.write(line); log.flush()
                    if proc.wait():
                        raise subprocess.CalledProcessError(proc.returncode, cmd)
            summarize(out)
    summarize(out)


def summarize(out):
    rows = []
    for name in VARIANTS:
        for seed in SEEDS:
            path = out/f'{name}_s{seed}'/'metrics.json'
            if not path.exists(): continue
            r = json.loads(path.read_text())
            c, d = r['valid_clean'], r['valid_corrupt20']
            rows.append({'variant': name, 'seed': seed, 'best_epoch': r['best_epoch'],
                         'clean': c, 'r20': d, 'score': (c['MacroF1']+d['MacroF1'])/2,
                         'checkpoint_bytes': path.with_name('best.pt').stat().st_size})
    paired = []
    for name in ('kd03', 'kd10'):
        for seed in SEEDS:
            b = next((r for r in rows if r['variant']=='baseline' and r['seed']==seed), None)
            c = next((r for r in rows if r['variant']==name and r['seed']==seed), None)
            if b and c: paired.append({'variant': name, 'seed': seed, 'delta_score':c['score']-b['score'],
                                       'delta_clean_F1':c['clean']['MacroF1']-b['clean']['MacroF1'],
                                       'delta_r20_F1':c['r20']['MacroF1']-b['r20']['MacroF1'],
                                       'delta_clean_MAE':c['clean']['MAE']-b['clean']['MAE']})
    aggregate=[]
    for name in VARIANTS:
        matched=[r for r in rows if r['variant']==name]
        if not matched: continue
        scores=[r['score'] for r in matched]
        import statistics
        aggregate.append({'variant':name,'n_seeds':len(matched),
                          'mean_score':statistics.mean(scores),
                          'score_sample_std':statistics.stdev(scores) if len(scores)>1 else None,
                          'mean_clean_F1':statistics.mean(r['clean']['MacroF1'] for r in matched),
                          'mean_r20_F1':statistics.mean(r['r20']['MacroF1'] for r in matched),
                          'mean_clean_MAE':statistics.mean(r['clean']['MAE'] for r in matched)})
    summary = {'runs': rows, 'paired_deltas': paired, 'aggregate':aggregate,
               'status': 'complete' if len(rows)==9 else 'partial',
               'interpretation': 'Validation experiments, not test results or a guaranteed improvement.'}
    (out/'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
