"""Aligned BERT-position interface; padding support and observed masks are distinct."""
from __future__ import annotations
import pickle,random
from dataclasses import dataclass,field,replace
from functools import lru_cache
from typing import Dict,List,Optional
import numpy as np
import config as C

def set_seed(seed=C.TrainCfg.seed):
    random.seed(seed);np.random.seed(seed)
    import torch
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def _load_pkl(path):
    with open(path,'rb') as f:return pickle.load(f,encoding='latin1')

def valid_mask(arr):
    a=np.asarray(arr)
    if a.ndim==3:a=a[0]
    return (np.isfinite(a).all(-1)&(np.abs(a).sum(-1)>1e-12)).astype(np.float32)

def token_support(text_bert):
    tb=np.asarray(text_bert).squeeze()
    if tb.shape!=(3,C.N_POS):raise ValueError(f'Unexpected text_bert shape {tb.shape}')
    return ((tb[1]>0)&~np.isin(tb[0],[0,101,102])).astype(np.float32)

@dataclass
class Sample:
    sid:str
    text:np.ndarray
    audio:np.ndarray
    vision:np.ndarray
    mask:Dict[str,np.ndarray]
    support:Dict[str,np.ndarray]=field(default_factory=dict)
    label_cls:Optional[int]=None
    label_reg:Optional[float]=None
    raw_text:str=''
    video_path:Optional[str]=None
    meta:dict=field(default_factory=dict)
    def feats(self):return {m:getattr(self,m) for m in C.MODALITIES}
    def __post_init__(self):
        if not self.support:self.support={m:np.ones_like(self.mask[m]) for m in C.MODALITIES}

def polarity_to_cls(y):return 2 if y>0 else (0 if y<0 else 1)

def compute_stats(samples):
    stats={}
    for m in C.MODALITIES:
        total=np.zeros(C.MOD_DIM[m],np.float64);sq=total.copy();n=0
        for s in samples:
            v=np.asarray(getattr(s,m)[s.mask[m]>0],np.float64)
            total+=v.sum(0);sq+=(v*v).sum(0);n+=len(v)
        if not n:raise ValueError(f'No observed training data for {m}')
        mean=total/n;std=np.sqrt(np.maximum(sq/n-mean*mean,0))
        stats[m]={'mean':mean.astype(np.float32),'std':np.where(std<1e-6,1,std).astype(np.float32)}
    return stats

def apply_stats(samples,stats):
    out=[]
    for s in samples:
        feats={m:((getattr(s,m)-stats[m]['mean'])/stats[m]['std'])*s.mask[m][:,None] for m in C.MODALITIES}
        if not all(np.isfinite(v).all() for v in feats.values()):raise ValueError(f'Nonfinite features {s.sid}')
        out.append(replace(s,**feats,mask={m:v.copy() for m,v in s.mask.items()},support={m:v.copy() for m,v in s.support.items()},meta=dict(s.meta)))
    return out

def save_stats(stats,path):
    np.savez_compressed(path,**{f'{m}_{k}':v for m,d in stats.items() for k,v in d.items()})
def load_stats(path):
    with np.load(path) as z:return {m:{k:z[f'{m}_{k}'] for k in ('mean','std')} for m in C.MODALITIES}

@lru_cache(maxsize=1)
def _a2_raw():return _load_pkl(C.A2_DIR/'aligned_50.pkl')

def load_a2(split):
    d=_a2_raw()[split];out=[]
    for i,sid in enumerate(d['id']):
        feats={m:np.asarray(d[m][i],np.float32) for m in C.MODALITIES}
        sup=token_support(d['text_bert'][i]);y=float(np.asarray(d['regression_labels'][i]).item())
        out.append(Sample(str(sid),**feats,mask={m:valid_mask(feats[m])*sup for m in C.MODALITIES},support={m:sup.copy() for m in C.MODALITIES},label_cls=polarity_to_cls(y),label_reg=y,raw_text=str(d['raw_text'][i]),meta={'text_bert':np.asarray(d['text_bert'][i],np.int64)}))
    return out

STATS_PATH=C.CACHE_DIR/'a2_stats_v2.npz'
def load_a2_all(normalize=True,splits=('train','valid','test')):
    data={sp:load_a2(sp) for sp in splits}
    if not normalize:return data
    if STATS_PATH.exists():stats=load_stats(STATS_PATH)
    else:
        stats=compute_stats(data.get('train') or load_a2('train'));save_stats(stats,STATS_PATH)
    out={sp:apply_stats(v,stats) for sp,v in data.items()}
    _a2_raw.cache_clear()
    return out

def load_a3(samples_text=True):
    paths=sorted(C.A3_ALIGNED_DIR.glob('*.pkl'))
    if not paths:raise FileNotFoundError(C.A3_ALIGNED_DIR)
    cache=C.CACHE_DIR/'a3_text_bert_direct_v2.npz';text_map={}
    if cache.exists():
        with np.load(cache) as z:text_map={k:z[k] for k in z.files}
    ds={p.stem:_load_pkl(p)['test'] for p in paths}
    missing=[p.stem for p in paths if p.stem not in text_map]
    if missing:
        if not samples_text:raise ValueError('Text cache is required; zero text cannot be marked available.')
        from text_encoder import encode_text_bert_batch
        embs=encode_text_bert_batch([np.asarray(ds[s]['text_bert']).squeeze().T for s in missing])
        text_map.update(dict(zip(missing,embs)));np.savez_compressed(cache,**text_map)
    out=[]
    for p in paths:
        d=ds[p.stem];sup=token_support(d['text_bert'])
        feats={m:np.asarray(d[m],np.float32).squeeze() for m in ('audio','vision')};feats['text']=text_map[p.stem]
        out.append(Sample(p.stem,**feats,mask={m:valid_mask(feats[m])*sup for m in C.MODALITIES},support={m:sup.copy() for m in C.MODALITIES},meta={'text_bert':np.asarray(d['text_bert']).squeeze(),'length_status':'observed_token_sequence; deleted original token count unknown'}))
    return apply_stats(out,load_stats(STATS_PATH))

def load_a4():
    paths=sorted(C.A4_ALIGNED_DIR.glob('*.pkl'));out=[]
    if not paths:raise FileNotFoundError(C.A4_ALIGNED_DIR)
    for p in paths:
        d=_load_pkl(p);sup=token_support(d['text_bert']);feats={m:np.asarray(d[m],np.float32).squeeze() for m in C.MODALITIES}
        vid=C.A4_VIDEO_DIR/f'{p.stem}.mp4'
        out.append(Sample(p.stem,**feats,mask={m:valid_mask(feats[m])*sup for m in C.MODALITIES},support={m:sup.copy() for m in C.MODALITIES},raw_text=str(d.get('raw_text','')),video_path=str(vid) if vid.exists() else None,meta={'text_bert':np.asarray(d['text_bert']).squeeze()}))
    return apply_stats(out,load_stats(STATS_PATH))

def parse_modalities(name):
    aliases={'a':('audio',),'v':('vision',),'t':('text',),'audio':('audio',),'vision':('vision',),'text':('text',),'av':('audio','vision'),'tv':('text','vision'),'ta':('text','audio'),'tav':('text','audio','vision')}
    if name not in aliases:raise ValueError(f'Unknown missing modality {name}')
    return aliases[name]

def count_runs(mask):
    b=np.asarray(mask,dtype=bool)
    return int((b&~np.r_[False,b[:-1]]).sum())

def sample_blocks(length,n_drop,n_segments,rng):
    """Exactly k nonadjacent blocks; cap k only when geometrically impossible."""
    if not n_drop:return np.zeros(length,bool)
    k=min(n_segments,n_drop,length-n_drop+1)
    if k<1:raise ValueError('n_segments must be positive')
    def composition(total, parts):
        if parts == 1:return np.array([total])
        bars=np.sort(rng.choice(total+parts-1,parts-1,replace=False))
        return np.diff(np.r_[-1,bars,total+parts-1])-1
    lengths=composition(n_drop-k,k)+1
    gaps=composition(length-n_drop-(k-1),k+1);gaps[1:k]+=1
    out=np.zeros(length,bool);cursor=int(gaps[0])
    for j,l in enumerate(lengths):
        out[cursor:cursor+l]=True;cursor+=int(l+gaps[j+1])
    return out

def apply_missing(sample,miss_type,miss_rate,rng,n_segments=1):
    if not 0<=miss_rate<=1:raise ValueError('miss_rate must be in [0,1]')
    if n_segments<1:raise ValueError('n_segments must be positive')
    feats={m:getattr(sample,m).copy() for m in C.MODALITIES};masks={m:sample.mask[m].copy() for m in C.MODALITIES};meta=dict(sample.meta);audit={}
    for m in parse_modalities(miss_type):
        valid=np.flatnonzero(masks[m]>0);L=len(valid);nd=int(round(miss_rate*L))
        drop=sample_blocks(L,nd,n_segments,rng) if L else np.zeros(0,bool)
        ix=valid[drop];feats[m][ix]=0;masks[m][ix]=0
        physical=np.zeros(C.N_POS,bool);physical[ix]=True
        audit[m]={'target_rate':miss_rate,'observed_before':L,'n_dropped':len(ix),'actual_rate':len(ix)/L if L else 0.,'requested_segments':n_segments,'segments_valid_index':count_runs(drop),'segments_storage_axis':count_runs(physical),'indices':ix.tolist()}
    meta['synthetic_missing']=audit
    return replace(sample,**feats,mask=masks,support={m:v.copy() for m,v in sample.support.items()},meta=meta)

def stack(samples):
    return ({m:np.stack([getattr(s,m) for s in samples]) for m in C.MODALITIES},
            {m:np.stack([s.mask[m] for s in samples]) for m in C.MODALITIES},
            np.array([s.label_cls if s.label_cls is not None else -1 for s in samples]),
            np.array([s.label_reg if s.label_reg is not None else np.nan for s in samples]),[s.sid for s in samples])
