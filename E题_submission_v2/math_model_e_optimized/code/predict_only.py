"""Inference using saved models, training statistics and frozen validation bias."""
import json
import config as C
import numpy as np,pandas as pd,torch
import io_utils as U
from inference import load_model,predict
from evaluate_final import calibrated
def main():
    torch.set_num_threads(4)
    bias=json.loads((C.MODEL_DIR/'decision_calibration.json').read_text())['p2']
    out=calibrated(predict(load_model('p2'),U.load_a3()),bias)
    pd.DataFrame({'sample_id':out['sid'],'pred_polarity':np.array(['Negative','Neutral','Positive'])[out['p_cls']],'pred_class':out['p_cls'],'pred_intensity':out['p_reg'].round(5),'confidence':out['prob'].max(1).round(5)}).to_csv(C.RESULT_DIR/'附件3_优化预测.csv',index=False,encoding='utf-8-sig')
    from explain_v2 import main as explain
    explain()
if __name__=='__main__':main()
