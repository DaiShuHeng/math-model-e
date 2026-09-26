"""Leakage-free, grouped CV for the revised P1 features."""
import json
import config as C
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
import trainer as T

def main():
    df=pd.read_csv(C.DATA_DIR/'p1_summary_v2.csv',dtype={'clip_id':str});rows=[]
    y=df.label.to_numpy();yc=np.sign(y).astype(int)+1;groups=df.video_id.to_numpy();features={m:[] for m in C.MODALITIES}
    for sid in df.sample_id:
        with np.load(C.P1_DIR/f'{sid.replace(C.SEP,"__")}.npz') as z:
            for m in C.MODALITIES:
                mask=z[f'mask_{m}'];features[m].append((z[m]*mask[:,None]).sum(0)/max(mask.sum(),1))
    for name,mods in [('text',['text']),('audio',['audio']),('vision',['vision']),('fusion',list(C.MODALITIES))]:
        X=np.concatenate([np.stack(features[m]) for m in mods],1)
        for seed in [2026,2027,2028]:
            pc=np.zeros(len(y),int);pr=np.zeros(len(y))
            for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(X,yc,groups):
                assert not set(groups[tr])&set(groups[te])
                clf=make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=2000,class_weight='balanced'))
                reg=make_pipeline(StandardScaler(),Ridge(alpha=10.))
                pc[te]=clf.fit(X[tr],yc[tr]).predict(X[te]);pr[te]=reg.fit(X[tr],y[tr]).predict(X[te]).clip(-3,3)
            rows.append({'feature':name,'seed':seed,**T.compute_metrics(yc,pc,y,pr)})
    pd.DataFrame(rows).to_csv(C.RESULT_DIR/'p1_grouped_cv.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(rows).groupby('feature')[['acc','f1_macro','mae','pearson']].agg(['mean','std']).to_string(),flush=True)

if __name__=='__main__':main()
