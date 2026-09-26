"""Train all planned variants using training/validation only; test is separate."""
import argparse,json,time,types
import config as C
import torch
import io_utils as U,models as M,trainer as T

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='2026,2027,2028');ap.add_argument('--epochs',type=int,default=35);ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--force',action='store_true',help='Retrain and overwrite completed runs in this project copy')
    args=ap.parse_args();torch.set_num_threads(args.threads)
    data=U.load_a2_all(splits=('train','valid'));dev=torch.device('cpu')
    plan=[('p2',.6,.1),('p3',.1,0.),('p2_no_reconstruction',.6,0.)]
    summary=[]
    for tag,aug,rec in plan:
        seeds=list(map(int,args.seeds.split(','))) if tag in ('p2','p3') else [2026]
        for seed in seeds:
            out_dir=C.MODEL_DIR/f'{tag}_{seed}';out_dir.mkdir(exist_ok=True,parents=True)
            if (out_dir/'valid_metrics.json').exists() and not args.force:
                summary.append(json.loads((out_dir/'valid_metrics.json').read_text()));continue
            cfg=types.SimpleNamespace(**C.train_cfg_dict());cfg.seed=seed;cfg.epochs=args.epochs;cfg.lambda_rec=rec;cfg.lambda_comp=0.
            U.set_seed(seed);net=M.RobustModel().to(dev)
            print(f'TRAIN {tag} seed={seed} params={M.count_params(net)}',flush=True)
            result=T.train_model(net,data['train'],data['valid'],dev,cfg=cfg,aug_prob=aug,tag=f'{tag}_{seed}',log=lambda s:print(s,flush=True))
            torch.save({'state':net.state_dict(),'model_cfg':C.model_cfg_dict(),'train_cfg':vars(cfg),'aug':aug,'interface':'bert_direct_content_mask_v2'},out_dir/'model.pt')
            (out_dir/'history.json').write_text(json.dumps(result['history'],indent=2),encoding='utf-8')
            row={'tag':tag,'seed':seed,'best_epoch':result['best']['epoch'],'params':M.count_params(net),**result['best']['metrics']}
            (out_dir/'valid_metrics.json').write_text(json.dumps(row,indent=2),encoding='utf-8');summary.append(row)
            print('VALID',json.dumps(row),flush=True)
    (C.RESULT_DIR/'validation_training_runs.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__':main()
