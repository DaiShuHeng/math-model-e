import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import config as C
import numpy as np,torch
import io_utils as U,models as M,trainer as T

class ProtocolTests(unittest.TestCase):
    def sample(self):
        return U.Sample('synthetic',**{m:np.ones((50,C.MOD_DIM[m]),np.float32) for m in C.MODALITIES},mask={m:np.ones(50,np.float32) for m in C.MODALITIES})
    def test_aliases_and_rates(self):
        for rate in [0,.2,.4,.8,1]:
            a=U.apply_missing(self.sample(),'text',rate,np.random.default_rng(4))
            b=U.apply_missing(self.sample(),'t',rate,np.random.default_rng(4))
            np.testing.assert_array_equal(a.mask['text'],b.mask['text'])
            self.assertEqual(int((a.mask['text']==0).sum()),round(50*rate))
    def test_exact_blocks(self):
        for length in [1,5,13,50]:
            for n in range(length+1):
                for k in [1,2,4,8]:
                    b=U.sample_blocks(length,n,k,np.random.default_rng(100+n+k))
                    self.assertEqual(b.sum(),n)
                    self.assertEqual(U.count_runs(b),min(k,n,length-n+1) if n else 0)
    def test_padding_not_missing(self):
        tb=np.zeros((3,50));tb[0,:7]=[101,10,11,12,13,14,102];tb[1,:7]=1
        self.assertEqual(U.token_support(tb).sum(),5)
        net=M.RobustModel().eval()
        x={m:torch.ones(2,50,C.MOD_DIM[m]) for m in C.MODALITIES}
        masks={m:torch.tensor(np.stack([U.token_support(tb)]*2)) for m in C.MODALITIES}
        with torch.no_grad():out=net(x,masks,support=masks,need_recon=False)
        np.testing.assert_allclose(out['avail'],1)
    def test_all_absent_gradients(self):
        net=M.RobustModel()
        x={m:torch.zeros(2,50,C.MOD_DIM[m]) for m in C.MODALITIES}
        masks={m:torch.zeros(2,50) for m in C.MODALITIES}
        out=net(x,masks,support=masks)
        loss=out['logits'].sum()+out['pred_reg'].sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
        self.assertTrue((out['beta']==0).all())
    def test_source_unchanged(self):
        s=self.sample();a=U.apply_missing(s,'tav',.8,np.random.default_rng(2))
        self.assertEqual(s.mask['text'].sum(),50)
        self.assertEqual(a.support['text'].sum(),50)
    def test_reconstruction_dimensions(self):
        x={m:torch.zeros(1,50,C.MOD_DIM[m]) for m in C.MODALITIES}
        rec={m:torch.ones_like(x[m]) for m in C.MODALITIES}
        nat={m:torch.ones(1,50) for m in C.MODALITIES};obs={m:torch.zeros(1,50) for m in C.MODALITIES}
        out={'logits':torch.zeros(1,3),'pred_reg':torch.zeros(1),'recon':rec}
        _,parts=M.compute_loss(out,torch.tensor([1]),torch.tensor([0.]),x,nat,obs)
        self.assertAlmostEqual(parts['rec'],.75,places=6)
    def test_ctc_repeated_character(self):
        from align_media import ctc_path
        # Known emission: blank, A, A, blank, A, B -> transcript AAB.
        logp=np.full((6,3),-20.,np.float32)
        logp[np.arange(6),[0,1,1,0,1,2]]=0.
        states,extended=ctc_path(logp,[1,1,2],0)
        path=extended[states];collapsed=[];previous=None
        for symbol in path:
            if symbol!=previous and symbol!=0:collapsed.append(int(symbol))
            previous=symbol
        self.assertEqual(collapsed,[1,1,2])
        self.assertTrue((np.diff(states)>=0).all())

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
