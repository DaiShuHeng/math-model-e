"""Integrity checks and uncertainty summaries from completed outputs."""
import json,importlib.metadata as md,platform
import config as C
import numpy as np,pandas as pd
from sklearn.metrics import f1_score
def main():
    checks={};summary=pd.read_csv(C.DATA_DIR/'p1_summary_v2.csv');assert len(summary)==100 and summary.sample_id.nunique()==100
    for row in summary.itertuples():
        with np.load(C.P1_DIR/f'{row.sample_id.replace(C.SEP,"__")}.npz') as z:
            assert str(z['schema_version'])=='physical_time_bins_v2'
            assert len(z['time_edges'])==51 and np.all(np.diff(z['time_edges'])>0)
            for m in C.MODALITIES:
                assert z[m].shape==(50,C.MOD_DIM[m]) and np.isfinite(z[m]).all()
                assert (z[m][z[f'mask_{m}']==0]==0).all()
    checks['p1_samples']=len(summary);checks['p1_media_failures']=int((summary.processing_status!='ok').sum())
    for name,n in [('附件3_优化预测.csv',30),('附件4_优化预测与解释.csv',20)]:
        df=pd.read_csv(C.RESULT_DIR/name,dtype={'sample_id':str});assert len(df)==n and df.sample_id.nunique()==n
        assert df.pred_intensity.between(-3,3).all();assert df.pred_polarity.isin(['Negative','Neutral','Positive']).all();checks[name]=len(df)
    evidence_count=0;timed_count=0
    for row in df.to_dict('records'):
        duration=json.loads((C.DATA_DIR/'alignment/a4'/f"{row['sample_id']}.json").read_text(encoding='utf-8'))['duration']
        norms=[row[m+'_normalized_abs_effect'] for m in C.MODALITIES]
        assert abs(sum(norms)-1)<.0001 or sum(norms)==0
        for m in C.MODALITIES:
            assert np.isfinite(row[m+'_probability_drop_on_removal'])
            for entry in json.loads(row[m+'_evidence']):
                evidence_count+=1;assert 0<=entry['position']<50 and np.isfinite(entry['probability_drop'])
                if 'start_s' in entry:
                    assert 0<=entry['start_s']<entry['end_s']<=duration+.001;timed_count+=1
    checks['evidence_entries']=evidence_count;checks['evidence_entries_with_times']=timed_count
    alignment=[]
    for group in ['a1','a4']:
        for path in sorted((C.DATA_DIR/'alignment'/group).glob('*.json')):
            d=json.loads(path.read_text(encoding='utf-8'));last=0
            for w in d['words']:
                assert 0<=w['start']<w['end']<=d['duration']+.001 and w['start']>=last-.001;last=w['end']
            alignment.append(dict(dataset=group,sample_id=path.stem,confidence=d['mean_confidence'],transcript_similarity=d['transcript_similarity'],needs_review=d['transcript_similarity']<.65))
    pd.DataFrame(alignment).to_csv(C.RESULT_DIR/'alignment_quality.csv',index=False,encoding='utf-8-sig')
    intervals=[]
    for tag in ['p2','p3']:
        z=np.load(C.RESULT_DIR/f'{tag}_test_predictions.npz');rng=np.random.default_rng(2026);values=[]
        for _ in range(1000):
            ix=rng.integers(0,len(z['y_cls']),len(z['y_cls']))
            values.append([np.mean(z['y_cls'][ix]==z['p_cls'][ix]),f1_score(z['y_cls'][ix],z['p_cls'][ix],labels=[0,1,2],average='macro',zero_division=0),np.mean(abs(z['y_reg'][ix]-z['p_reg'][ix])),np.corrcoef(z['y_reg'][ix],z['p_reg'][ix])[0,1]])
        for j,m in enumerate(['acc','f1_macro','mae','pearson']):
            lo,hi=np.quantile(np.array(values)[:,j],[.025,.975]);intervals.append(dict(model=tag,metric=m,ci95_low=lo,ci95_high=hi,method='sample bootstrap; does not account for shared source-video correlation'))
    pd.DataFrame(intervals).to_csv(C.RESULT_DIR/'test_bootstrap_intervals.csv',index=False,encoding='utf-8-sig')
    packages=['torch','numpy','pandas','scikit-learn','transformers','librosa','opencv-python','imageio-ffmpeg','openpyxl','scipy','matplotlib','joblib','huggingface-hub','markdown','tabulate']
    versions={p:md.version(p) for p in packages};versions['python']=platform.python_version()
    (C.ROOT/'environment.json').write_text(json.dumps(versions,indent=2))
    (C.ROOT/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items() if p!='python')+'\n')
    (C.RESULT_DIR/'integrity_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8');print(checks,flush=True)
if __name__=='__main__':main()
