import unittest
from types import SimpleNamespace
import tempfile
from pathlib import Path
import numpy as np
import torch
from src.distillation import teacher_loss, TeacherDataset, split_fingerprint


class DistillationTests(unittest.TestCase):
    def test_corrupted_or_wrong_teacher_has_zero_gradient(self):
        logits = torch.tensor([[2.,0.,-1.], [2.,0.,-1.]], requires_grad=True)
        reg = torch.zeros(2, requires_grad=True)
        batch = dict(teacher_logits=torch.tensor([[0.,2.,0.],[2.,0.,0.]]),
                     teacher_reg=torch.ones(2), teacher_clean=torch.tensor([True, False]),
                     y_cls=torch.tensor([0,0]))
        loss, rate = teacher_loss(dict(logits=logits,reg=reg), batch)
        loss.backward()
        self.assertEqual(float(loss.detach()),0.)
        self.assertEqual(float(rate),0.)
        self.assertEqual(float(logits.grad.abs().sum()),0.)
        self.assertEqual(float(reg.grad.abs().sum()),0.)

    def test_clean_correct_teacher_trains_student_not_teacher(self):
        logits = torch.zeros(1,3,requires_grad=True)
        teacher = torch.tensor([[4.,0.,0.]],requires_grad=True)
        reg = torch.zeros(1,requires_grad=True)
        batch = dict(teacher_logits=teacher, teacher_reg=torch.ones(1),
                     teacher_clean=torch.tensor([True]),y_cls=torch.tensor([0]))
        loss, rate = teacher_loss(dict(logits=logits,reg=reg),batch)
        loss.backward()
        self.assertGreater(float(loss.detach()),0)
        self.assertLess(float(logits.grad[0,0]),0)
        self.assertLess(float(reg.grad[0]),0)
        self.assertIsNone(teacher.grad)

    def test_cache_rejects_reordered_or_changed_training_inputs(self):
        sd=SimpleNamespace(**{k:np.zeros((2,2),dtype=np.float32) for k in
                              ('ids','attn','audio','vision','y_cls','y_reg')},sample_id=['a','b'])
        base=SimpleNamespace(sd=sd,augmenter=None)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'teacher.npz'
            np.savez(path,split='train',fingerprint=split_fingerprint(sd),sample_id=['a','b'])
            sd.ids[0,0]=1
            with self.assertRaisesRegex(ValueError,'TRAIN split'):
                TeacherDataset(base,path)

    def test_reordered_cache_rejected_even_with_valid_fingerprint(self):
        sd=SimpleNamespace(**{k:np.zeros((2,2),dtype=np.float32) for k in
                              ('ids','attn','audio','vision','y_cls','y_reg')},sample_id=['a','b'])
        base=SimpleNamespace(sd=sd,augmenter=None)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'teacher.npz'
            np.savez(path,split='train',fingerprint=split_fingerprint(sd),sample_id=['b','a'])
            with self.assertRaisesRegex(ValueError,'order mismatch'):
                TeacherDataset(base,path)

    def test_augmented_views_are_not_distilled(self):
        sd=SimpleNamespace(ids=np.array([[101,5,102]]),attn=np.ones((1,3)),
                           audio=np.ones((1,3,2)),vision=np.ones((1,3,2)),
                           audio_obs=np.ones((1,3)),vision_obs=np.ones((1,3)),
                           y_cls=np.array([0]),y_reg=np.array([-1.]),sample_id=['a'])
        class Base:
            augmenter=None
            def __len__(self): return 1
            def __getitem__(self,j):
                return dict(ids=torch.tensor(sd.ids[j].copy()),audio_obs=torch.ones(3),vision_obs=torch.ones(3))
        base=Base();base.sd=sd
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'teacher.npz'
            np.savez(path,split='train',fingerprint=split_fingerprint(sd),sample_id=['a'],
                     logits=np.array([[2,0,0]]),reg=np.array([-1.]))
            ds=TeacherDataset(base,path)
            self.assertTrue(ds[0]['teacher_clean'])
            class Damaged(Base):
                def __getitem__(self,j):
                    result=super().__getitem__(j);result['ids'][1]=100
                    return result
            ds.base=Damaged();ds.base.sd=sd
            self.assertFalse(ds[0]['teacher_clean'])

if __name__=='__main__': unittest.main()
