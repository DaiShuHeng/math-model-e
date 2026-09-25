"""Feature-position occlusion ranking evaluated by paired group deletion.
This evaluates local feature sensitivity, not raw-word causality or Shapley faithfulness."""
import json
from dataclasses import replace
import config as C
import numpy as np,pandas as pd,torch
import io_utils as U
from inference import load_model,predict
from evaluate_final import calibrated
from explain_v2 import remove

def main():
    torch.set_num_threads(4);samples=U.load_a2_all(splits=('valid',))['valid']
    # Prespecified subset, selected before examining explanation effects.
    indices=np.random.default_rng(2026).choice(len(samples),128,replace=False);samples=[samples[i] for i in indices]
    net=load_model('p3');bias=json.loads((C.MODEL_DIR/'decision_calibration.json').read_text())['p3'];base=calibrated(predict(net,samples),bias)
    cls=base['p_cls'];p=base['prob'][np.arange(len(samples)),cls];rows=[];detail=[]
    effects=np.full((len(samples),3,C.N_POS),-np.inf)
    tasks=[];positions=[]
    for i,s in enumerate(samples):
        for mi,m in enumerate(C.MODALITIES):
            for j in np.flatnonzero(s.mask[m]):
                tasks.append(remove(s,m,[j]));positions.append((i,mi,j))
    if tasks:
        local=calibrated(predict(net,tasks),bias)
        for n,(i,mi,j) in enumerate(positions):effects[i,mi,j]=p[i]-local['prob'][n,cls[i]]
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
        a=calibrated(predict(net,changed),bias);drop=p-a['prob'][np.arange(len(samples)),cls]
        random_drop=[]
        for seq in randomized:
            b=calibrated(predict(net,seq),bias);random_drop.append(p-b['prob'][np.arange(len(samples)),cls])
        random_drop=np.mean(random_drop,0);difference=drop-random_drop
        rng=np.random.default_rng(2026)
        boot=np.asarray([np.concatenate([difference[groups==g] for g in rng.choice(unique,len(unique),replace=True)]).mean() for _ in range(2000)])
        rows.append({'fraction':frac,'n':len(samples),'n_video_groups':len(unique),'scope':'feature_position_occlusion','local_occlusion_drop':float(drop.mean()),'random_drop':float(random_drop.mean()),'paired_difference':float(difference.mean()),'ci95_low':float(np.quantile(boot,.025)),'ci95_high':float(np.quantile(boot,.975))})
        for i,s in enumerate(samples):detail.append({'sample_id':s.sid,'fraction':frac,'local_occlusion_drop':float(drop[i]),'random_drop':float(random_drop[i])})
    pd.DataFrame(rows).to_csv(C.RESULT_DIR/'revised_local_explanation_validation.csv',index=False,encoding='utf-8-sig');pd.DataFrame(detail).to_csv(C.RESULT_DIR/'revised_local_explanation_validation_pairs.csv',index=False,encoding='utf-8-sig')
    print(rows,flush=True)

if __name__=='__main__':main()
