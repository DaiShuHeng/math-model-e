"""Identity-checked, clean-view-only teacher supervision; no inference dependency."""
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def split_fingerprint(sd):
    h = hashlib.sha256()
    for name in ('ids', 'attn', 'audio', 'vision', 'y_cls', 'y_reg'):
        a = np.ascontiguousarray(getattr(sd, name))
        h.update(name.encode()); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    for sid in sd.sample_id:
        h.update(str(sid).encode()); h.update(b'\0')
    return h.hexdigest()


class TeacherDataset(Dataset):
    def __init__(self, base, path):
        self.base = base
        self.augmenter = base.augmenter
        with np.load(path, allow_pickle=False) as z:
            if str(z['split']) != 'train' or str(z['fingerprint']) != split_fingerprint(base.sd):
                raise ValueError('Teacher cache does not match this TRAIN split')
            if not np.array_equal(z['sample_id'], np.asarray(base.sd.sample_id).astype(str)):
                raise ValueError('Teacher sample order mismatch')
            self.logits = z['logits'].astype(np.float32)
            self.reg = z['reg'].astype(np.float32)
        if self.logits.shape != (len(base), 3) or self.reg.shape != (len(base),):
            raise ValueError('Invalid teacher output shape')
        if not np.isfinite(self.logits).all() or not np.isfinite(self.reg).all():
            raise ValueError('Nonfinite teacher outputs')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, j):
        b = self.base[j]
        sd = self.base.sd
        # Augmentation only changes token IDs or zeroes A/V rows. If these
        # observations are unchanged, the student sees the teacher's clean view.
        clean = (np.array_equal(b['ids'].numpy(), sd.ids[j])
                 and np.array_equal(b['audio_obs'].numpy(), sd.audio_obs[j])
                 and np.array_equal(b['vision_obs'].numpy(), sd.vision_obs[j]))
        b.update(teacher_logits=torch.from_numpy(self.logits[j]),
                 teacher_reg=torch.tensor(self.reg[j]),
                 teacher_clean=torch.tensor(clean))
        return b


def teacher_loss(out, batch, temperature=2.0):
    """Only clean examples whose teacher predicts the supplied class correctly.

    Mean over the whole batch keeps KD magnitude proportional to eligible data.
    No confidence threshold tuned on special samples; targets are detached.
    """
    target = batch['teacher_logits'].detach().float()
    use = batch['teacher_clean'] & (target.argmax(-1) == batch['y_cls'])
    kl = F.kl_div(F.log_softmax(out['logits'].float()/temperature, -1),
                  F.softmax(target/temperature, -1), reduction='none').sum(-1) * temperature**2
    reg = F.smooth_l1_loss(out['reg'].float(), batch['teacher_reg'].detach().float(),
                           beta=1.0, reduction='none')
    return ((kl + reg) * use.float()).mean(), use.float().mean()
