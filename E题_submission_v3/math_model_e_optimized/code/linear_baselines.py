"""Train-only fitted pooled linear models, hyperparameters chosen on valid only."""
import json
import config as C
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
import io_utils as U,trainer as T

def pooled(samples,mods):
    return np.stack([np.concatenate([(getattr(s,m)*s.mask[m][:,None]).sum(0)/max(1,s.mask[m].sum()) for m in mods]) for s in samples])

def main():
    data=U.load_a2_all(splits=('train','valid'));rows=[]
    for name,mods in [('text',['text']),('audio',['audio']),('vision',['vision']),('fusion',list(C.MODALITIES))]:
        X={sp:pooled(samples,mods) for sp,samples in data.items()}
        yc={sp:np.array([s.label_cls for s in samples]) for sp,samples in data.items()};yr={sp:np.array([s.label_reg for s in samples]) for sp,samples in data.items()}
        best_cls=None;best_reg=None
        for c in [.01,.1,1.]:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1500,class_weight='balanced'))
            model.fit(X['train'],yc['train']);pc=model.predict(X['valid'])
            f=T.compute_metrics(yc['valid'],pc,yr['valid'],np.zeros(len(pc)))['f1_macro']
            if best_cls is None or f>best_cls[0]:best_cls=(f,c,pc,model)
        for alpha in [1.,10.,100.,1000.]:
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha));model.fit(X['train'],yr['train']);pr=model.predict(X['valid']).clip(-3,3);mae=float(np.abs(pr-yr['valid']).mean())
            if best_reg is None or mae<best_reg[0]:best_reg=(mae,alpha,pr,model)
        row={'model':name,'C':best_cls[1],'ridge_alpha':best_reg[1],**T.compute_metrics(yc['valid'],best_cls[2],yr['valid'],best_reg[2])};rows.append(row)
        import joblib
        joblib.dump({'classifier':best_cls[3],'regressor':best_reg[3],'modalities':mods},C.MODEL_DIR/f'linear_{name}.joblib')
        print(row,flush=True)
    pd.DataFrame(rows).to_csv(C.RESULT_DIR/'linear_baselines_valid.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':main()
