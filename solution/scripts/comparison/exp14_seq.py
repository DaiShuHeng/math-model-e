"""Exp14: integrate their FUSION-SEQUENCE mechanism into my model, symmetrically.

Teammate fusion: alpha_m (softmax of score + log avail) mixes per-modality
SEQUENCES into one fused sequence F, and the reconstruction decoder consumes F.
Mine: alpha_m mixes pooled vectors z_m, and pooling happens per modality first.

Variants (identical augmentation, budget, selection score, standardiser):
  pooled          : my current design (control)
  seq             : fused sequence + transformer layer + attention pooling
  seq_rec         : seq + reconstruction decoder reading F (their full mechanism)
  seq_rec_miss    : seq_rec + learnable missing token
"""
import sys, time, json, argparse
sys.path.insert(0, '/tmp/eexp')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from exp2_strong import DEV
from exp12_integrate import augment, OBSKEY, DIM, RATES
from exp13_calib import mk, stats, metrics, fit_bias, apply_bias

torch.set_num_threads(8)


class BranchSeq(nn.Module):
    """Per-modality projection + a learnable missing token; keeps the sequence."""

    def __init__(self, dim, dh, drop, missing_token=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dh), nn.GELU())
        self.drop = nn.Dropout(drop)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, dh)) if missing_token else None
        if self.missing_token is not None:
            nn.init.normal_(self.missing_token, std=0.02)
        self.dh = dh

    def forward(self, x, m):
        mm = m[..., None]
        h = self.proj(x) * mm
        if self.missing_token is not None:
            h = h + self.missing_token * (1.0 - mm)
        return self.drop(h), m.sum(1, keepdim=True) / m.size(1)


class NetSeq(nn.Module):
    def __init__(self, dh=96, dmodel=96, drop=0.25, nhead=4, nlayer=1, dff=192,
                 missing_token=False, rec=False, seq=True):
        super().__init__()
        self.cfg = dict(dh=dh, dmodel=dmodel, drop=drop, nhead=nhead, nlayer=nlayer,
                        dff=dff, missing_token=missing_token, rec=rec, seq=seq)
        self.bs = nn.ModuleDict({m: BranchSeq(DIM[m], dh, drop, missing_token) for m in "TAV"})
        self.score = nn.Sequential(nn.Linear(dh, dh // 2), nn.GELU(), nn.Dropout(drop), nn.Linear(dh // 2, 1))
        self.norm = nn.LayerNorm(dmodel)
        self.dmodel = dmodel
        if dmodel != dh:
            self.to_model = nn.Linear(dh, dmodel)
        else:
            self.to_model = nn.Identity()
        if seq:
            layer = nn.TransformerEncoderLayer(dmodel, nhead, dff, drop, batch_first=True,
                                               norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, nlayer)
        self.q = nn.Parameter(torch.randn(dmodel) / np.sqrt(dmodel))
        self.hc = nn.Linear(dmodel, 3)
        self.hr = nn.Linear(dmodel, 1)
        if rec:
            self.dec = nn.ModuleDict({m: nn.Sequential(nn.Linear(dmodel, dmodel), nn.GELU(),
                                                       nn.Linear(dmodel, DIM[m])) for m in "TAV"})

    def forward(self, b):
        h, av = {}, {}
        for m in "TAV":
            h[m], av[m] = self.bs[m](b[m], b[OBSKEY[m]])
        A = torch.cat([av[m] for m in "TAV"], -1)                  # (B,3) availability
        # modality score must be a per-sample scalar: masked mean over observed time
        sc = []
        for m in "TAV":
            o = b[OBSKEY[m]][..., None]
            sm = self.score(h[m])                                  # (B,T,1)
            sc.append((sm * o).sum(1) / o.sum(1).clamp_min(1e-6))   # (B,1)
        s = torch.cat(sc, -1)                                      # (B,3)
        logits = (s + torch.log(A.clamp_min(1e-3))).masked_fill(A <= 0, -1e4)
        alpha = torch.softmax(logits, -1)                          # (B,3)
        F_seq = (alpha[:, 0:1, None] * h['T'] + alpha[:, 1:2, None] * h['A']
                 + alpha[:, 2:3, None] * h['V'])
        F_seq = self.to_model(F_seq)
        if self.cfg['seq']:
            pad = b['t_obs'] <= 0
            pad = pad.clone()
            pad[pad.all(1), 0] = False
            F_seq = self.enc(F_seq, src_key_padding_mask=pad)
        F_seq = self.norm(F_seq)
        mask = b['t_obs']
        mm = mask[..., None]
        lg = torch.einsum('btd,d->bt', F_seq, self.q) / np.sqrt(F_seq.size(-1))
        lg = lg.masked_fill(mm.squeeze(-1) <= 0, -1e4)
        w = torch.softmax(lg, -1) * (mask > 0).float()
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
        z = torch.einsum('bt,btd->bd', w, F_seq)
        out = dict(logits=self.hc(z), reg=3 * torch.tanh(self.hr(z)).squeeze(-1),
                   gate=alpha, avail=A, F_seq=F_seq)
        if self.cfg['rec']:
            out['recon'] = {m: self.dec[m](F_seq) for m in "TAV"}
        return out


def train(seed, variant, epochs=140, bs=64, patience=30, verbose=False, lr=6e-4,
          wd=1e-4, lam_rec=0.1):
    from harness import get
    d = get()
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    mu_t, sd_t, mu_a, sd_a, mu_v, sd_v = stats()
    tr = mk('train', mu_t, sd_t, mu_a, sd_a, mu_v, sd_v)
    va = mk('valid', mu_t, sd_t, mu_a, sd_a, mu_v, sd_v)
    te = mk('test', mu_t, sd_t, mu_a, sd_a, mu_v, sd_v)
    cnt = np.bincount(d['train_ycls'], minlength=3)
    cw = torch.tensor(len(d['train_ycls']) / (3 * np.maximum(cnt, 1)), dtype=torch.float32, device=DEV)
    kw = dict(rec='rec' in variant, missing_token='miss' in variant, seq='pooled' not in variant)
    model = NetSeq(**kw).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = max(1, (len(d['train_ycls']) // bs + 1) * epochs)
    warm = max(1, int(steps * 0.1))

    def lr_lambda(step):
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, steps - warm)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    n = len(d['train_ycls']); best = (-1, None); state = None; bad = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            sub = {k: v[idx] for k, v in tr.items()}
            clean = {k: v.clone() for k, v in sub.items()}
            sub, keep = augment(sub, rng)
            sub['y'] = tr['y'][idx]; sub['r'] = tr['r'][idx]
            out = model(sub)
            loss = F.cross_entropy(out['logits'], sub['y'], weight=cw, label_smoothing=0.05) + \
                   F.smooth_l1_loss(out['reg'], sub['r'], beta=0.5)
            if kw['rec'] and keep is not None:
                tot = 0
                for m in "TAV":
                    tm = ((clean[OBSKEY[m]] > 0) & (sub[OBSKEY[m]] <= 0))
                    if tm.sum() == 0:
                        continue
                    pred = out['recon'][m]
                    diff = F.smooth_l1_loss(pred[tm], clean[m][tm], beta=0.5, reduction='sum')
                    tot = tot + diff / (tm.sum().clamp_min(1) * DIM[m])
                loss = loss + lam_rec * tot
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            o = model(va)
            pc = o['logits'].argmax(-1).cpu().numpy()
        from harness import cm
        cl = cm(d['valid_ycls'], pc)
        vb, _ = augment({k: v.clone() for k, v in va.items()}, np.random.default_rng(777), p_aug=1.0)
        with torch.no_grad():
            o2 = model(vb)
        r20 = cm(d['valid_ycls'], o2['logits'].argmax(-1).cpu().numpy())
        score = 0.5 * (cl['MacroF1'] + r20['MacroF1'])
        if score > best[0]:
            best = (score, dict(score=score, ep=ep + 1, clean=cl, r20=r20)); bad = 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (ep < 3 or (ep + 1) % 30 == 0):
            print(f'    ep{ep+1:3d} clean {cl["MacroF1"]:.4f} r20 {r20["MacroF1"]:.4f} score {score:.4f}', flush=True)
        if bad >= patience:
            break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        pv = torch.softmax(model(va)['logits'], -1).cpu().numpy()
        pt = torch.softmax(model(te)['logits'], -1).cpu().numpy()
    b = fit_bias(pv, d['valid_ycls'])
    return model, dict(best=best[1], bias=[float(x) for x in b],
                       valid_raw=metrics(d['valid_ycls'], pv.argmax(1)),
                       valid_cal=metrics(d['valid_ycls'], apply_bias(pv, b).argmax(1)),
                       test_raw=metrics(d['test_ycls'], pt.argmax(1)),
                       test_cal=metrics(d['test_ycls'], apply_bias(pt, b).argmax(1)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='2026')
    ap.add_argument('--variants', default='pooled,seq,seq_rec,seq_rec_miss')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    res = {}
    for v in a.variants.split(','):
        rows = []
        for sd in seeds:
            t0 = time.time()
            _, r = train(sd, v, verbose=True)
            rows.append(r)
            print(f"  [{v} s{sd}] val {r['valid_raw']['f1']:.4f}->{r['valid_cal']['f1']:.4f} | "
                  f"TEST {r['test_raw']['acc']:.4f}/{r['test_raw']['f1']:.4f} -> "
                  f"{r['test_cal']['acc']:.4f}/{r['test_cal']['f1']:.4f} | bias {np.round(r['bias'],2).tolist()} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        mean = {k: float(np.mean([x[k]['f1'] for x in rows])) for k in ('valid_raw','valid_cal','test_raw','test_cal')}
        mean_acc = {k: float(np.mean([x[k]['acc'] for x in rows])) for k in ('valid_raw','test_raw','test_cal')}
        res[v] = dict(rows=rows, mean_f1=mean, mean_acc=mean_acc)
        print(f"[{v:14s}] MEAN val {mean['valid_raw']:.4f}->{mean['valid_cal']:.4f} | "
              f"TEST f1 {mean['test_raw']:.4f}->{mean['test_cal']:.4f}  acc {mean_acc['test_raw']:.4f}->{mean_acc['test_cal']:.4f}\n", flush=True)
        json.dump(res, open('/tmp/eexp/exp14_seq.json','w'), indent=2, default=float)
