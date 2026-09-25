"""Read-only training input check, plus an optional real-train optimizer step.
Does not evaluate test performance, fit special-test data or write checkpoints.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from . import config
from .data_adapter import read_pickle,Standardizer,SplitData
from .models import Q2Model,q2_loss
from .datasets import Q2Dataset
from .augment import MixedBlockAugment
from .consistency import confident_consistency


def inspect_data(path):
    data=read_pickle(path)
    report={};video_sets={}
    for split in ['train','valid','test']:
        d=data[split];n=len(d['id']);tb=np.asarray(d['text_bert']);audio=np.asarray(d['audio']);vision=np.asarray(d['vision'])
        for name,a,shape in [('text_bert',tb,(n,3,50)),('audio',audio,(n,50,74)),('vision',vision,(n,50,35))]:
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f'{split}/{name}: invalid shape or nonfinite')
        labels=np.asarray(d['classification_labels']).reshape(-1).astype(int);reg=np.asarray(d['regression_labels']).reshape(-1)
        if labels.shape!=(n,) or reg.shape!=(n,) or not np.isfinite(reg).all() or (np.abs(reg)>3).any():raise ValueError('Invalid labels')
        if not np.array_equal(labels,np.where(reg<0,0,np.where(reg>0,2,1))):raise ValueError('Class/regression label mapping mismatch')
        mask=tb[:,1,:];lengths=mask.sum(-1)
        if not np.isin(mask,[0,1]).all() or not np.array_equal(mask,np.arange(50)[None]<lengths[:,None]):raise ValueError('Non-prefix attention mask')
        video_sets[split]={str(x).split('$_$')[0] for x in d['id']}
        report[split]={'samples':n,'class_counts':np.bincount(labels,minlength=3).tolist()}
    for a,b in [('train','valid'),('train','test'),('valid','test')]:
        if video_sets[a]&video_sets[b]:raise ValueError(f'Video leakage: {a}/{b}')
    return data,report


def smoke(model,d):
    # Build train statistics on the full training split, then take a small
    # fixed training batch solely for a finite-loss/gradient portability check.
    tb=np.asarray(d['text_bert'],dtype=np.int64);a=np.asarray(d['audio'],np.float32);v=np.asarray(d['vision'],np.float32)
    sd=SplitData('train',tb[:,0],tb[:,1],tb[:,2],a,v,~np.isclose(a,0).all(-1),~np.isclose(v,0).all(-1),np.asarray(d['regression_labels'],np.float32).reshape(-1),np.asarray(d['classification_labels'],np.int64).reshape(-1),list(d['id']))
    ds=Q2Dataset(sd,Standardizer.fit(sd),augmenter=MixedBlockAugment(seed=2026),paired_clean=True)
    batch=next(iter(torch.utils.data.DataLoader(ds,batch_size=4,num_workers=0)))
    clean={k.removeprefix('clean_'):v for k,v in batch.items() if k.startswith('clean_')};clean.update(y_cls=batch['y_cls'],y_reg=batch['y_reg'])
    model.train();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    out=model(batch);full=model(clean)
    l1,_=q2_loss(out,batch);l2,_=q2_loss(full,clean);kl,_=confident_consistency(full,out,batch['y_cls'])
    loss=.5*(l1+l2)+.1*kl
    loss.backward()
    if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Nonfinite loss/gradient')
    optimizer.step()
    return {'status':'passed','samples':4,'meaning':'real TRAIN execution check only; not accuracy evidence'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bert-dir',type=Path,default=config.BERT_DIR);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    data,splits=inspect_data(config.ALIGNED_PKL)
    model=Q2Model(bert_dir=args.bert_dir)
    if int(np.max(data['train']['text_bert'][:,0]))>=model.bert.config.vocab_size:raise ValueError('Token IDs exceed pretrained vocabulary')
    count=sum(p.numel() for p in model.parameters())
    q3=config.ROOT/'math_model_e_optimized/data/cache/a2_stats_v2.npz'
    q3_valid=False
    if q3.exists():
        with np.load(q3,allow_pickle=False) as z:
            q3_valid=all(z[f'{m}_{k}'].shape==(dim,) and np.isfinite(z[f'{m}_{k}']).all() and (k!='std' or (z[f'{m}_{k}']>0).all()) for m,dim in [('text',768),('audio',74),('vision',35)] for k in ['mean','std'])
    result={'training_ready':True,'aligned_bytes':config.ALIGNED_PKL.stat().st_size,'splits':splits,
        'bert_dir':str(args.bert_dir),'model_parameter_bytes_fp32':count*4,
        'cuda_available':torch.cuda.is_available(),'torch':torch.__version__,
        'q1_features':len(list((config.SOLUTION/'data/p1_features_v3').glob('*.npz'))),
        'attachment3_aligned_count':len(list(config.ATT3_DIR.glob('*.pkl'))),
        'attachment4_aligned_count':len(list(config.ATT4_DIR.glob('*.pkl'))),
        'q3_normalizer_valid':bool(q3_valid),
        'optional_for_q2_training':{'bert_base_present':(config.ROOT/'models/bert-base-uncased/config.json').exists(),
        'unaligned_present':(config.DATA_DIR/'附件2-数据集特征文件/unaligned_50.pkl').exists()}}
    if args.smoke:result['training_smoke']=smoke(model,data['train'])
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
