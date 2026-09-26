"""Freeze validation-only decision calibration, then evaluate and export once."""
import json,itertools
from dataclasses import replace
import config as C
import numpy as np,pandas as pd,torch
from sklearn.metrics import confusion_matrix,classification_report
import io_utils as U,trainer as T
from inference import load_model,predict

def calibrate(prob,y):
    best=None
    for bn,bu in itertools.product([0.,-.2,.2],repeat=2):
        bias=np.array([bn,bu,0.]);p=(np.log(prob.clip(1e-9))+bias).argmax(1)
        from sklearn.metrics import f1_score
        score=f1_score(y,p,average='macro')
        if best is None or score>best[0]+1e-12:best=(score,bias)
    return best[1]

def calibrated(out,bias):
    out=dict(out);logp=np.log(out['prob'].clip(1e-9))+np.asarray(bias)
    prob=np.exp(logp-logp.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
    out['prob']=prob;out['p_cls']=prob.argmax(1)
    labelled=out['y_cls']>=0
    out['metrics']=T.compute_metrics(out['y_cls'][labelled],out['p_cls'][labelled],out['y_reg'][labelled],out['p_reg'][labelled]) if labelled.any() else {}
    return out

def main():
    torch.set_num_threads(4)
    data=U.load_a2_all(splits=('valid','test'));report={};rows=[];cal={}
    for tag in ['p2','p3']:
        net=load_model(tag);va=predict(net,data['valid']);bias=calibrate(va['prob'],va['y_cls']);cal[tag]=bias.tolist()
        va_cal=calibrated(va,bias)
        report[tag]={'validation_uncalibrated':va['metrics'],'validation':va_cal['metrics'],'log_probability_bias':bias.tolist()}
        pd.DataFrame({'sample_id':va_cal['sid'],'true_class':va_cal['y_cls'],'pred_class':va_cal['p_cls'],'true_intensity':va_cal['y_reg'],'pred_intensity':va_cal['p_reg'],'absolute_error':np.abs(va_cal['y_reg']-va_cal['p_reg']),'text':[s.raw_text for s in data['valid']]}).to_csv(C.RESULT_DIR/f'{tag}_validation_details.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(confusion_matrix(va_cal['y_cls'],va_cal['p_cls'],labels=[0,1,2]),index=['Negative','Neutral','Positive'],columns=['Negative','Neutral','Positive']).to_csv(C.RESULT_DIR/f'{tag}_confusion_valid.csv',encoding='utf-8-sig')
        # Bias frozen before test predictions.
        res=calibrated(predict(net,data['test']),bias)
        report[tag]['test']=res['metrics']
        report[tag]['per_class']=classification_report(res['y_cls'],res['p_cls'],labels=[0,1,2],target_names=['Negative','Neutral','Positive'],output_dict=True,zero_division=0)
        pd.DataFrame(confusion_matrix(res['y_cls'],res['p_cls'],labels=[0,1,2]),index=['Negative','Neutral','Positive'],columns=['Negative','Neutral','Positive']).to_csv(C.RESULT_DIR/f'{tag}_confusion_test.csv',encoding='utf-8-sig')
        np.savez_compressed(C.RESULT_DIR/f'{tag}_test_predictions.npz',y_cls=res['y_cls'],p_cls=res['p_cls'],y_reg=res['y_reg'],p_reg=res['p_reg'],prob=res['prob'])
        for split,metrics in [('valid',report[tag]['validation']),('test',res['metrics'])]:rows.append({'model':tag,'split':split,**metrics})
        print(tag,json.dumps(report[tag]),flush=True)
    (C.MODEL_DIR/'decision_calibration.json').write_text(json.dumps(cal,indent=2))
    (C.RESULT_DIR/'final_metrics.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame(rows).to_csv(C.RESULT_DIR/'model_metrics.csv',index=False,encoding='utf-8-sig')
    net=load_model('p2');a3=U.load_a3();res=calibrated(predict(net,a3),cal['p2'])
    pd.DataFrame({'sample_id':res['sid'],'pred_polarity':np.array(['Negative','Neutral','Positive'])[res['p_cls']],'pred_class':res['p_cls'],'pred_intensity':res['p_reg'].round(5),'confidence':res['prob'].max(1).round(5)}).to_csv(C.RESULT_DIR/'附件3_优化预测.csv',index=False,encoding='utf-8-sig')
    stats=[]
    for group,samples in [('a3',a3),('a4',U.load_a4())]:
        for s in samples:
            for m in C.MODALITIES:
                expected=s.support[m].sum();observed=s.mask[m].sum()
                stats.append({'dataset':group,'id':s.sid,'modality':m,'storage_positions':50,'content_positions':int(expected),'observed_positions':int(observed),'padding_or_special_positions':int(50-expected),'zero_within_content':int(expected-observed),'missing_rate_within_available_token_sequence':float(1-observed/max(expected,1)),'note':'Original deleted text token count unknown for attachment3' if group=='a3' else 'Aligned BERT token support'})
    pd.DataFrame(stats).to_csv(C.RESULT_DIR/'padding_and_observation_audit.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':main()
