"""Bounded validation-only ablations. --plan is read-only; never evaluates TEST/A3/A4."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path

EXPERIMENTS = [
    ('baseline_cw', []),
    ('sqrt_cw', ['--cls-weight-power','0.5']),
    ('paired_cw', ['--paired-clean']),
    ('paired_consistency_cw', ['--paired-clean','--consistency-weight','0.1']),
    ('finetune_cw', ['--freeze-text','0','--ft-lr','0.00001']),
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--bert-dir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--plan', action='store_true')
    args=ap.parse_args()
    plan=[]
    for name,flags in EXPERIMENTS:
        command=[sys.executable,'-m','src.train_q2','--seed',str(args.seed),'--epochs',str(args.epochs),
                 '--num-workers','0','--cls-weight','1','--bert-dir',str(args.bert_dir.resolve()),
                 '--out',str(args.out.resolve()/name),*flags]
        plan.append({'name':name,'command':command})
    if args.plan:
        print(json.dumps({'selection':'mean(clean MacroF1, continuous TAV r20 MacroF1)',
                          'scope':'TRAIN/VALID only; results unknown','plan':plan},ensure_ascii=False,indent=2));return
    # Verify inputs before spending compute or creating experiment outputs.
    from .data_adapter import load_splits
    from .models import Q2Model
    load_splits()
    model=Q2Model(bert_dir=args.bert_dir);del model
    if args.out.exists():raise FileExistsError('Choose a new experiment directory')
    args.out.mkdir(parents=True)
    (args.out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    summary=[]
    for item in plan:
        print(f"Starting {item['name']} (TRAIN/VALID only)",flush=True)
        with (args.out/(item['name']+'.log')).open('w') as log:
            subprocess.run(item['command'],check=True,stdout=log,stderr=subprocess.STDOUT)
        run=args.out/item['name']
        report=json.loads((run/'metrics.json').read_text())
        clean=report['valid_clean'];corrupt=report['valid_corrupt20']
        summary.append({'name':item['name'],'seed':args.seed,'clean':clean,'continuous_r20':corrupt,
                        'score':0.5*(clean['MacroF1']+corrupt['MacroF1']),
                        'checkpoint_bytes':(run/'best.pt').stat().st_size,
                        'wall_time_sec':report['wall_time_sec']})
        (args.out/'validation_comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    baseline=summary[0]
    for row in summary:
        row['delta_score_vs_baseline']=row['score']-baseline['score']
        row['delta_clean_mae_vs_baseline']=row['clean']['MAE']-baseline['clean']['MAE']
    (args.out/'validation_comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(sorted(summary,key=lambda x:-x['score']),ensure_ascii=False,indent=2))
    print('Pilot results only: confirm candidates with additional seeds and full validation missingness grid. Do not tune on TEST.')

if __name__=='__main__':main()
