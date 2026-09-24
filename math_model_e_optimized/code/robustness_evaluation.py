"""Controlled held-out validation corruption experiments, with actual mask audits."""
import json
from dataclasses import replace
import config as C
import numpy as np,pandas as pd,torch
import io_utils as U
from inference import load_model,predict
from evaluate_final import calibrated

def corruption(samples,mt,rate,segments,seed,position='random'):
    rng=np.random.default_rng(seed);out=[];audits=[]
    for s in samples:
        if position=='random':new=U.apply_missing(s,mt,rate,rng,n_segments=segments)
        else:
            feats={m:getattr(s,m).copy() for m in C.MODALITIES};masks={m:s.mask[m].copy() for m in C.MODALITIES}
            for m in U.parse_modalities(mt):
                valid=np.flatnonzero(masks[m]);n=round(len(valid)*rate)
                start={'front':0,'middle':(len(valid)-n)//2,'back':len(valid)-n}[position]
                ix=valid[start:start+n];feats[m][ix]=0;masks[m][ix]=0
            new=replace(s,**feats,mask=masks)
        for m in U.parse_modalities(mt):
            drop=(s.mask[m]>0)&(new.mask[m]==0);n=int((s.mask[m]>0).sum())
            audits.append({'id':s.sid,'modality':m,'target_rate':rate,'actual_rate':float(drop.sum()/n) if n else 0.,'n_before':n,'n_dropped':int(drop.sum()),'segments_requested':segments,'segments_storage':U.count_runs(drop)})
        out.append(new)
    return out,audits

def main():
    torch.set_num_threads(4);samples=U.load_a2_all(splits=('valid',))['valid'];net=load_model('p2')
    bias=json.loads((C.MODEL_DIR/'decision_calibration.json').read_text())['p2']
    cases=[]
    for mt in C.MISS_TYPES:
        for rate in [0.,.2,.4,.6,.8]:cases.append(('rate',mt,rate,1,'random'))
    for mt in ['text','audio','vision']:
        for k in [2,4,8]:cases.append(('segments',mt,.4,k,'random'))
        for pos in ['front','middle','back']:cases.append(('position',mt,.4,1,pos))
    rows=[];audit_summary=[]
    zero_metrics=calibrated(predict(net,samples),bias)['metrics']
    for family,mt,rate,k,pos in cases:
        scores=[]
        for seed in [11,22,33]:
            masked,aud=corruption(samples,mt,rate,k,seed,pos)
            # Zero corruption and fixed-location corruption are deterministic.
            if rate==0:metrics=zero_metrics
            elif pos!='random' and scores:metrics=scores[0]
            else:metrics=calibrated(predict(net,masked),bias)['metrics']
            scores.append(metrics)
            da=pd.DataFrame(aud)
            audit_summary.append({'family':family,'modality':mt,'target_rate':rate,'segments':k,'position':pos,'seed':seed,'mean_actual_rate':da.actual_rate.mean(),'mean_actual_segments':da.segments_storage.mean(),'max_rounding_error':float((da.actual_rate-da.target_rate).abs()[da.n_before>0].max())})
        row={'family':family,'modality':mt,'rate':rate,'segments':k,'position':pos}
        for metric in scores[0]:row[metric]=float(np.mean([s[metric] for s in scores]));row[f'{metric}_std']=float(np.std([s[metric] for s in scores],ddof=1))
        rows.append(row);print('ROBUST',family,mt,rate,k,pos,round(row['f1_macro'],4),flush=True)
    pd.DataFrame(rows).to_csv(C.RESULT_DIR/'robustness_valid.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(audit_summary).to_csv(C.RESULT_DIR/'robustness_mask_audit.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':main()
