"""Checkpoint configuration is loaded explicitly; fixed equal-weight seed ensemble."""
from pathlib import Path
import types
import config as C
import numpy as np,torch
from torch.utils.data import DataLoader
import models as M,trainer as T

class Ensemble(torch.nn.Module):
    def __init__(self,members):super().__init__();self.members=torch.nn.ModuleList(members)
    def forward(self,X,masks,need_recon=False,support=None):
        outs=[m(X,masks,need_recon=False,support=support) for m in self.members]
        return {'logits':torch.stack([o['logits'].softmax(-1) for o in outs]).mean(0).clamp_min(1e-10).log(),
                'pred_reg':torch.stack([o['pred_reg'] for o in outs]).mean(0),
                'alpha':torch.stack([o['alpha'] for o in outs]).mean(0),
                'beta':torch.stack([o['beta'] for o in outs]).mean(0)}

def load_model(tag,seeds=(2026,2027,2028)):
    members=[]
    for seed in seeds:
        ck=torch.load(C.MODEL_DIR/f'{tag}_{seed}'/'model.pt',map_location='cpu',weights_only=True)
        if ck.get('interface')!='bert_direct_content_mask_v2':raise ValueError('Incompatible input interface')
        model=M.RobustModel(types.SimpleNamespace(**ck['model_cfg']));model.load_state_dict(ck['state']);members.append(model.eval())
    return Ensemble(members).eval()

def predict(model,samples,batch_size=128):
    loader=DataLoader(T.MoseiDataset(samples),batch_size=batch_size,shuffle=False,collate_fn=T.collate,num_workers=0)
    return T.predict(model,loader,torch.device('cpu'))
