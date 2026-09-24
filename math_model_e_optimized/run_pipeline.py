"""Sequential execution avoids competing for memory on desktop machines."""
import argparse,os,subprocess,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STEPS=[('train','train_experiments.py',[]),('features','p1_refine.py',['--workers','1']),('feature-validation','p1_validation.py',[]),('baselines','linear_baselines.py',[]),('evaluate','evaluate_final.py',[]),('robustness','robustness_evaluation.py',[]),('explain','explain_v2.py',[]),('explanation-validation','explanation_validation.py',[])]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--start',choices=[s[0] for s in STEPS],default='train');ap.add_argument('--only',choices=[s[0] for s in STEPS]);args=ap.parse_args()
    env=os.environ.copy();env.update(MATH_E_DATA=str(Path(args.data_root).resolve()),PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',KMP_DUPLICATE_LIB_OK='TRUE')
    (ROOT/'logs').mkdir(exist_ok=True);enabled=False;run=[]
    for name,script,extra in STEPS:
        enabled=enabled or name==args.start
        if args.only and args.only!=name:continue
        if not args.only and not enabled:continue
        print('START',name,flush=True);t=time.time()
        with (ROOT/'logs'/f'{name}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,str(ROOT/'code'/script),*extra],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        run.append(dict(step=name,exit_code=result.returncode,seconds=round(time.time()-t,1)))
        (ROOT/'logs'/'pipeline_status.json').write_text(json.dumps(run,indent=2))
        print('END',name,result.returncode,run[-1]['seconds'],flush=True)
        if result.returncode:raise SystemExit(f'{name} failed; see logs/{name}.log')
if __name__=='__main__':main()
