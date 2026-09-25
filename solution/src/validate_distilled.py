"""Validation-only missingness diagnostics; preserve per-sample predictions."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .augment import WordLevelTAV
from .checkpoints import load_revised
from .data_adapter import load_training_splits
from .datasets import Q2Dataset
from .metrics import cls_metrics, reg_metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    torch.set_num_threads(args.threads)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,std,ck=load_revised(args.checkpoint,device)
    sd=load_training_splits()['valid']
    conditions=[('clean',None)]
    for mt in ('t','a','v','ta','tv','av','tav'):
        for rate in (.1,.3,.5):
            conditions.append((f'{mt}_r{rate:.1f}_any',WordLevelTAV(p_file=1,rate_pool=[rate],modalities=tuple(mt))))
    for region in ('head','mid','tail'):
        conditions.append((f'tav_r0.3_{region}',WordLevelTAV(p_file=1,rate_pool=[.3],region=region)))
    rows=[];predictions={}
    for name,aug in conditions:
        ds=Q2Dataset(sd,std,train=False,augmenter=aug,fixed_corrupt_seed=0)
        probs=[];regs=[]
        with torch.no_grad():
            for b in DataLoader(ds,batch_size=128):
                out=model({k:v.to(device) for k,v in b.items()})
                probs.append(out['logits'].float().softmax(-1).cpu().numpy())
                regs.append(out['reg'].float().cpu().numpy())
        prob=np.concatenate(probs);reg=np.concatenate(regs)
        row={'condition':name,**cls_metrics(sd.y_cls,prob.argmax(-1)),**reg_metrics(sd.y_reg,reg)}
        rows.append(row);predictions[name+'__prob']=prob;predictions[name+'__reg']=reg
        print(json.dumps(row),flush=True)
    main_grid=rows[1:22]
    report={'scope':'VALID only; raw-token UNK before encoder; continuous A/V zero blocks',
            'checkpoint':str(args.checkpoint),'train_args':ck['args'],'conditions':rows,
            'grid_mean_F1':float(np.mean([r['MacroF1'] for r in main_grid])),
            'grid_worst_F1':float(min(r['MacroF1'] for r in main_grid)),
            'grid_mean_MAE':float(np.mean([r['MAE'] for r in main_grid]))}
    args.out.mkdir(parents=True)
    (args.out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    np.savez_compressed(args.out/'predictions.npz',sample_id=np.asarray(sd.sample_id).astype(str),
                        y_cls=sd.y_cls,y_reg=sd.y_reg,**predictions)

if __name__=='__main__':main()
