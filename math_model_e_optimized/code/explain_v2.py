"""Feature-position occlusion; NOT raw-word deletion.
Text effects retain contextual information in unmasked BERT features.
Audio/vision timestamps inferred from token alignment need independent review.
"""
import json
from dataclasses import replace
import config as C
import numpy as np,pandas as pd,torch
import io_utils as U
from inference import load_model,predict
from evaluate_final import calibrated

def remove(s,m,idx):
    feats={k:getattr(s,k).copy() for k in C.MODALITIES};masks={k:s.mask[k].copy() for k in C.MODALITIES}
    feats[m][idx]=0.;masks[m][idx]=0.
    return replace(s,**feats,mask=masks)

def position_mapping(s,tokenizer):
    alignment=json.loads((C.DATA_DIR/'alignment/a4'/f'{s.sid}.json').read_text(encoding='utf-8'))
    enc=tokenizer(s.raw_text,padding='max_length',truncation=True,max_length=50,return_offsets_mapping=True)
    stored=np.asarray(s.meta['text_bert'])[0].astype(int)
    matched=np.array_equal(np.asarray(enc['input_ids']),stored)
    if not matched:
        # Some supplied arrays omit a final special token after truncation. Map
        # only an exactly matching prefix; never manufacture correspondence.
        prefix=next((i for i,(a,b) in enumerate(zip(enc['input_ids'],stored)) if a!=b),50)
    else:prefix=50
    mapping={}
    for pos,(a,b) in enumerate(enc['offset_mapping']):
        if b<=a or pos>=prefix:continue
        overlapping=[w for w in alignment['words'] if a<w['char_end'] and b>w['char_start']]
        if overlapping:
            w=overlapping[0];mapping[pos]={'word':s.raw_text[w['char_start']:w['char_end']], 'start_s':w['start'],'end_s':w['end'],'alignment_confidence':w['confidence'],'char_start':w['char_start']}
        else:
            mapping[pos]={'word':s.raw_text[a:b],'char_start':a,'mapping_status':'verified_text_fragment_no_speech_timestamp'}
    return mapping,alignment,matched,prefix

def main():
    torch.set_num_threads(4);samples=U.load_a4();net=load_model('p3')
    bias=json.loads((C.MODEL_DIR/'decision_calibration.json').read_text())['p3']
    base=calibrated(predict(net,samples),bias);cls=base['p_cls'];prob=base['prob'][np.arange(len(samples)),cls]
    contribution=np.zeros((len(samples),3));intensity=np.zeros_like(contribution)
    for mi,m in enumerate(C.MODALITIES):
        changed=[remove(s,m,np.flatnonzero(s.mask[m])) for s in samples]
        alt=calibrated(predict(net,changed),bias)
        contribution[:,mi]=prob-alt['prob'][np.arange(len(samples)),cls]
        intensity[:,mi]=base['p_reg']-alt['p_reg']
    tasks=[];locations=[]
    for i,s in enumerate(samples):
        for mi,m in enumerate(C.MODALITIES):
            valid=np.flatnonzero(s.mask[m])
            for j in valid:tasks.append(remove(s,m,[j]));locations.append((i,mi,int(j)))
    perturbed=calibrated(predict(net,tasks),bias)
    effects=np.full((len(samples),3,50),np.nan)
    for n,(i,mi,j) in enumerate(locations):effects[i,mi,j]=prob[i]-perturbed['prob'][n,cls[i]]
    del tasks,perturbed
    from text_encoder import load_tokenizer
    tokenizer=load_tokenizer();rows=[];evidence=[];mapping_audit=[]
    for i,s in enumerate(samples):
        mapping,alignment,exact,prefix=position_mapping(s,tokenizer)
        mapping_audit.append({'sample_id':s.sid,'token_ids_exact_match':bool(exact),'matched_prefix_length':prefix,'mapped_positions':sum('start_s' in v for v in mapping.values()),'text_mapped_positions':len(mapping),'content_positions':int(s.support['text'].sum()),'alignment_similarity':alignment['transcript_similarity'],'alignment_confidence':alignment['mean_confidence']})
        abs_contrib=np.abs(contribution[i]);norm=abs_contrib/max(abs_contrib.sum(),1e-9);dom=int(np.argmax(abs_contrib))
        row={'sample_id':s.sid,'pred_polarity':['Negative','Neutral','Positive'][cls[i]],'pred_intensity':round(float(base['p_reg'][i]),5),'confidence':round(float(prob[i]),5),'main_reference_modality':C.MODALITIES[dom],'text':s.raw_text,'alignment_status':'automatic_needs_review' if alignment['transcript_similarity']<.65 else 'automatic','head_sign_agreement':int(cls[i]==U.polarity_to_cls(base['p_reg'][i]))}
        for mi,m in enumerate(C.MODALITIES):
            row[f'{m}_gate_weight']=round(float(base['alpha'][i,mi]),5)
            row[f'{m}_probability_drop_on_removal']=round(float(contribution[i,mi]),5)
            row[f'{m}_normalized_abs_effect']=round(float(norm[mi]),5)
            row[f'{m}_intensity_change_on_removal']=round(float(intensity[i,mi]),5)
            valid=np.flatnonzero(s.mask[m]);order=valid[np.argsort(-effects[i,mi,valid])];chosen=[];seen=set()
            for j in order:
                span=mapping.get(int(j));key=span['char_start'] if span else f'unmapped{j}'
                if key in seen:continue
                seen.add(key);entry={'position':int(j),'explanation_scope':'feature_position_occlusion','time_mapping_status':('ctc_text_anchor' if m=='text' else 'inferred_from_text_anchor_unverified_av_timing'),'probability_drop':round(float(effects[i,mi,j]),6),'pooling_weight':round(float(base['beta'][i,mi,j]),6)}
                if span:entry.update(span)
                else:entry['mapping_status']='unverified_position_no_timestamp'
                chosen.append(entry)
                if len(chosen)==3:break
            row[f'{m}_evidence']=json.dumps(chosen,ensure_ascii=False)
            for rank,entry in enumerate(chosen,1):evidence.append({'sample_id':s.sid,'modality':m,'rank':rank,**entry})
        rows.append(row)
    pd.DataFrame(rows).to_csv(C.RESULT_DIR/'revised_附件4_特征位置解释.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(evidence).to_csv(C.RESULT_DIR/'revised_evidence_time_mapping.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(mapping_audit).to_csv(C.RESULT_DIR/'revised_evidence_mapping_audit.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(C.RESULT_DIR/'revised_explanation_effects.npz',modality_effect=contribution,local_effect=effects,pooling_beta=base['beta'],gate_alpha=base['alpha'])
    print('EXPLANATIONS',len(rows),'mapped',pd.DataFrame(mapping_audit).to_dict('records'),flush=True)

if __name__=='__main__':main()
