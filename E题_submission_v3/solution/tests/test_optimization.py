import unittest
import numpy as np
import torch
from src.consistency import confident_consistency
from src.datasets import Q2Dataset
from src.data_adapter import SplitData,Standardizer
from src.augment import WordLevelTAV

class OptimizationTests(unittest.TestCase):
    def test_teacher_detached_student_gradients(self):
        a=torch.tensor([[4.,0.,0.]],requires_grad=True);b=torch.zeros((1,3),requires_grad=True)
        loss,fraction=confident_consistency({'logits':a},{'logits':b},torch.tensor([0]))
        loss.backward();self.assertIsNone(a.grad);self.assertGreater(b.grad.abs().sum().item(),0);self.assertEqual(fraction.item(),1)
    def test_wrong_teacher_not_transferred(self):
        a=torch.tensor([[4.,0.,0.]],requires_grad=True);b=torch.zeros((1,3),requires_grad=True)
        loss,fraction=confident_consistency({'logits':a},{'logits':b},torch.tensor([2]));loss.backward()
        self.assertEqual(loss.item(),0);self.assertEqual(fraction.item(),0);self.assertEqual(b.grad.abs().sum().item(),0)
    def test_matching_predictions_zero_kl(self):
        a=torch.tensor([[4.,0.,0.]])
        loss,_=confident_consistency({'logits':a},{'logits':a.clone()},torch.tensor([0]))
        self.assertLess(abs(loss.item()),1e-6)
    def test_clean_copy_unchanged(self):
        ids=np.array([[101,210,211,212,102,0]],dtype=np.int64);attn=np.array([[1,1,1,1,1,0]])
        audio=np.ones((1,6,74),np.float32);vision=np.ones((1,6,35),np.float32)
        sd=SplitData('train',ids,attn,np.zeros_like(ids),audio,vision,np.ones((1,6),bool),np.ones((1,6),bool),np.array([1.],np.float32),np.array([2]),['v$_$0'])
        std=Standardizer(np.zeros(74),np.ones(74),np.zeros(35),np.ones(35))
        ds=Q2Dataset(sd,std,paired_clean=True,augmenter=WordLevelTAV(p_file=1,rate_pool=[1]))
        result=ds[0]
        np.testing.assert_array_equal(result['clean_ids'],ids[0]);np.testing.assert_array_equal(result['clean_audio'],audio[0])
        self.assertTrue((result['ids'][1:4]==100).all());self.assertTrue((result['audio'][1:4]==0).all())
        self.assertTrue((sd.ids[0,1:4]!=100).all())

if __name__=='__main__':unittest.main()
