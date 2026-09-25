"""Synthetic protocol regressions; no attachment 3/4 fitting or real training needed."""
import tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from src.augment import WordLevelTAV, MixedBlockAugment, assert_augmentation_invariants
from src.models import Q2Model, MaskedAttnPool
from src.checkpoints import load_revised
from src import config

class RevisionTests(unittest.TestCase):
    def arrays(self):
        ids=np.r_[101,np.arange(200,218),102,np.zeros(30)].astype('int64')
        return ids, np.r_[np.ones(20),np.zeros(30)].astype('int64'),np.ones((50,74),'float32'),np.ones((50,35),'float32')
    def test_block_locations_same_count(self):
        for region,start in [('head',1),('mid',6),('tail',12)]:
            ids,attn,a,v=self.arrays();original=ids.copy();before=attn.copy()
            damage=WordLevelTAV(p_file=1,rate_pool=[.4],region=region)(ids,attn,a,v)
            self.assertEqual(np.flatnonzero(damage).tolist(),list(range(start,start+7)))
            assert_augmentation_invariants(original,before,ids,a,v,damage,attn)
            bad=attn.copy();bad[-1]=1
            with self.assertRaises(AssertionError):assert_augmentation_invariants(original,before,ids,a,v,damage,bad)
    def test_subsets(self):
        for mods in MixedBlockAugment.SUBSETS:
            ids,attn,a,v=self.arrays();old=[x.copy() for x in (ids,a,v)]
            d=WordLevelTAV(p_file=1,rate_pool=[.5],modalities=mods)(ids,attn,a,v)
            for m,x,y in zip('tav',(ids,a,v),old):
                self.assertTrue(np.array_equal(x[~d],y[~d]))
                self.assertEqual(np.array_equal(x,y),m not in mods)
    def test_scatter_capacity(self):
        with self.assertRaises(ValueError):WordLevelTAV(p_file=1,rate_pool=[.7],contiguous=False,region='head')(*self.arrays())
    def test_empty_pool(self):
        p=MaskedAttnPool(4);out=p(torch.randn(2,5,4),torch.zeros(2,5))
        self.assertTrue(torch.equal(out,torch.zeros_like(out)))
    def model(self):
        return Q2Model(bert_config=dict(vocab_size=400,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,max_position_embeddings=64)).eval()
    def batch(self,n):
        torch.manual_seed(4)
        b={'ids':torch.tensor([[101,210,100,212,102]+[0]*(n-5)]),'attn':torch.tensor([[1]*5+[0]*(n-5)])}
        for name,dim in [('audio',74),('vision',35)]:
            x=torch.randn(1,5,dim);b[name]=torch.cat([x,torch.zeros(1,n-5,dim)],1);b[name+'_obs']=b['attn'].float()
        return b
    def test_padding_invariance(self):
        model=self.model()
        with torch.no_grad():a=model(self.batch(7));b=model(self.batch(12))
        # Text gate still uses occupancy n/L: identical logical content must not depend on padding length.
        for k in ['logits','reg','gate']:
            torch.testing.assert_close(a[k],b[k],atol=2e-6,rtol=2e-6)
    def test_checkpoint_roundtrip_and_legacy_rejection(self):
        m=self.model();batch=self.batch(8)
        ck={'protocol_version':config.PROTOCOL_VERSION,'args':{},'bert_config':m.bert.config.to_dict(),'state_dict':m.state_dict(),'normalizer':{f'{mod}_{stat}':([0.]*d if stat=='mean' else [1.]*d) for mod,d in [('a',74),('v',35)] for stat in ['mean','std']}}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'new.pt';torch.save(ck,p);reloaded,_,_=load_revised(p)
            with torch.no_grad():torch.testing.assert_close(m(batch)['logits'],reloaded(batch)['logits'])
            ck.pop('protocol_version');torch.save(ck,p)
            with self.assertRaises(ValueError):load_revised(p)

if __name__=='__main__':unittest.main()
