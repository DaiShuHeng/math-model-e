"""Feature-position occlusion ranking evaluated by paired group deletion.
This evaluates local feature sensitivity, not raw-word causality or Shapley faithfulness."""
import json, argparse
from pathlib import Path
import config as C
import numpy as np,pandas as pd,torch
import io_utils as U
from inference import load_model
import trainer as T
from torch.utils.data import DataLoader
from explain_v2 import remove

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model', choices=('p2','p3'), default='p2')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=4)
    args=ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    torch.set_num_threads(args.threads)
    raw=U.load_a2_all(normalize=False,splits=('train','valid'))
    stats=U.compute_stats(raw['train']);samples=U.apply_stats(raw['valid'],stats)
    del raw
    U._a2_raw.cache_clear()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def predict(model,seq):
        dl=DataLoader(T.MoseiDataset(seq),batch_size=128,collate_fn=T.collate)
        return T.predict(model,dl,device)
    net=load_model(args.model).to(device)
    # Compare uncalibrated models, without reusing validation-fitted biases.
    full=predict(net,samples)
    # Prespecified subset selected independently of model predictions.
    indices=np.random.default_rng(2026).choice(len(samples),128,replace=False)
    samples=[samples[i] for i in indices]
    base=predict(net,samples)
    print(json.dumps({'model':args.model,'valid_metrics':full['metrics'],'status':'ranking feature positions'},ensure_ascii=False),flush=True)
    cls=base['p_cls'];p=base['prob'][np.arange(len(samples)),cls];rows=[];detail=[]
    effects=np.full((len(samples),3,C.N_POS),-np.inf)
    # Stream removals to bound RAM; never materialize every copied feature array.
    tasks=[];positions=[]
    def flush():
        if tasks:
            local=predict(net,tasks)
            for n,(i,mi,j) in enumerate(positions):
                effects[i,mi,j]=p[i]-local['prob'][n,cls[i]]
            tasks.clear();positions.clear()
    for i,sample in enumerate(samples):
        for mi,m in enumerate(C.MODALITIES):
            for j in np.flatnonzero(sample.mask[m]):
                tasks.append(remove(sample,m,[j]));positions.append((i,mi,j))
                if len(tasks)>=128: flush()
    flush()
    print('Position ranking complete; evaluating grouped deletions.',flush=True)
    groups=np.asarray([s.sid.split(C.SEP)[0] for s in samples]);unique=np.unique(groups)
    for frac in [.1,.2,.4]:
        changed=[];randomized=[[] for _ in range(5)]
        for i,s in enumerate(samples):
            top_s=s
            for mi,m in enumerate(C.MODALITIES):
                valid=np.flatnonzero(s.mask[m]);k=min(len(valid),max(1,round(len(valid)*frac)))
                idx=valid[np.argsort(-effects[i,mi,valid])[:k]];top_s=remove(top_s,m,idx)
            changed.append(top_s)
            for repeat in range(5):
                sr=s;rng=np.random.default_rng(100000*int(frac*10)+i*10+repeat)
                for m in C.MODALITIES:
                    valid=np.flatnonzero(s.mask[m]);k=min(len(valid),max(1,round(len(valid)*frac)))
                    sr=remove(sr,m,rng.choice(valid,k,replace=False))
                randomized[repeat].append(sr)
        a=predict(net,changed);drop=p-a['prob'][np.arange(len(samples)),cls]
        random_drop=[]
        for seq in randomized:
            b=predict(net,seq);random_drop.append(p-b['prob'][np.arange(len(samples)),cls])
        random_drop=np.mean(random_drop,0);difference=drop-random_drop
        rng=np.random.default_rng(2026)
        boot=np.asarray([np.concatenate([difference[groups==g] for g in rng.choice(unique,len(unique),replace=True)]).mean() for _ in range(2000)])
        rows.append({'fraction':frac,'n':len(samples),'n_video_groups':len(unique),'scope':'feature_position_occlusion','local_occlusion_drop':float(drop.mean()),'random_drop':float(random_drop.mean()),'paired_difference':float(difference.mean()),'ci95_low':float(np.quantile(boot,.025)),'ci95_high':float(np.quantile(boot,.975))})
        for i,s in enumerate(samples):detail.append({'sample_id':s.sid,'fraction':frac,'local_occlusion_drop':float(drop[i]),'random_drop':float(random_drop[i])})
    args.out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(args.out/'feature_deletion.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(detail).to_csv(args.out/'feature_deletion_pairs.csv',index=False,encoding='utf-8-sig')
    (args.out/'report.json').write_text(json.dumps({
        'model':args.model,'calibrated':False,'valid_metrics':full['metrics'],
        'scope':'VALID feature-position group deletion versus five random controls; not raw-word causal evidence',
        'subset_seed':2026,'subset_size':128,'bootstrap_unit':'source video',
        'explanation_validation':rows},ensure_ascii=False,indent=2))

    print(rows,flush=True)

if __name__=='__main__':main()
