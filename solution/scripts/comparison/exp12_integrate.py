"""Exp12: integrate the teammate's ideas into my feature-level model.

Ablation (all on the same data, same budget, same selection score):
  base   : my v5 recipe
  +rec   : reconstruction of artificially masked feature elements (their lambda_rec)
  +miss  : learnable missing token instead of a zeroed row
  +ls    : label smoothing 0.05 in CE
  full   : all of the above + mask-aware gate prior log(avail)
"""
import sys, time, json, argparse, itertools
sys.path.insert(0, '/tmp/eexp')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from harness import get
from exp2_strong import C, DEV

torch.set_num_threads(8)
RATES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
SUBSETS = (("T",), ("A",), ("V",), ("T", "A"), ("T", "V"), ("A", "V"), ("T", "A", "V"))
DIM = {"T": 768, "A": 74, "V": 35}


class Branch(nn.Module):
    def __init__(self, dim, dh, drop, missing_token=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dh), nn.GELU())
        self.q = nn.Parameter(torch.randn(dh) / np.sqrt(dh))
        self.drop = nn.Dropout(drop)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, dh)) if missing_token else None
        if self.missing_token is not None:
            nn.init.normal_(self.missing_token, std=0.02)
        self.dh = dh

    def forward(self, x, m):
        mm = m[..., None]
        if self.missing_token is not None:
            h = self.proj(x) * mm + self.missing_token * (1.0 - mm)
        else:
            h = self.proj(x) * mm
        denom = mm.sum(1).clamp_min(1e-6)
        mean = (h * mm).sum(1) / denom
        mx = h.masked_fill(mm <= 0, -1e4).max(1).values
        mx = torch.where(m.sum(1, keepdim=True) > 0, mx, torch.zeros_like(mx))
        lg = torch.einsum('btd,d->bt', h, self.q) / np.sqrt(h.size(-1))
        lg = lg.masked_fill(mm.squeeze(-1) <= 0, -1e4)
        w = torch.softmax(lg, -1) * (m > 0).float()
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
        ap = torch.einsum('bt,btd->bd', w, h)
        rep = self.drop(torch.cat([mean, mx, ap], -1))
        avail = m.sum(1, keepdim=True) / m.size(1)
        return rep, avail, h


class Net2(nn.Module):
    def __init__(self, dh=64, dmodel=128, drop=0.4, rec=False, missing_token=False,
                 label_smooth=0.0, gate_prior=False):
        super().__init__()
        self.cfg = dict(dh=dh, dmodel=dmodel, drop=drop, rec=rec,
                        missing_token=missing_token, label_smooth=label_smooth,
                        gate_prior=gate_prior)
        self.bt = Branch(768, dh, drop, missing_token)
        self.ba = Branch(74, dh, drop, missing_token)
        self.bv = Branch(35, dh, drop, missing_token)
        total = 3 * 3 * dh
        self.gate = nn.Sequential(nn.Linear(total + 3, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, 3))
        self.fuse = nn.Sequential(nn.LayerNorm(total), nn.Linear(total, dmodel), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(dmodel, dmodel), nn.GELU())
        self.hc = nn.Linear(dmodel, 3)
        self.hr = nn.Linear(dmodel, 1)
        if rec:
            # decode the fused SEQUENCE back to each modality's feature space
            # per-position decoder: fused sequence is not available here, so the
            # decoder consumes the pooled fused vector broadcast over time and
            # predicts the modality's per-position feature row.
            self.dec = nn.ModuleDict({m: nn.Sequential(nn.Linear(total, dmodel), nn.GELU(),
                                                       nn.Linear(dmodel, DIM[m]))
                                      for m in "TAV"})

    def forward(self, b):
        rt, at, ht = self.bt(b['T'], b['t_obs'])
        ra, aa, ha = self.ba(b['A'], b['a_obs'])
        rv, av, hv = self.bv(b['V'], b['v_obs'])
        reps = torch.cat([rt, ra, rv], -1)
        avs = torch.cat([at, aa, av], -1)
        scores = self.gate(torch.cat([reps, avs], -1))
        if self.cfg['gate_prior']:
            scores = scores + torch.log(avs.clamp_min(1e-3))
        scores = scores.masked_fill(avs <= 0, -1e4)
        g = torch.softmax(scores, -1)
        fused = (reps.view(reps.size(0), 3, -1) * g[..., None]).reshape(reps.size(0), -1)
        z = self.fuse(fused)
        out = dict(logits=self.hc(z), reg=3 * torch.tanh(self.hr(z)).squeeze(-1), gate=g, avail=avs)
        if self.cfg['rec']:
            out['recon'] = {m: self.dec[m](fused) for m in "TAV"}   # (B, DIM[m])
        return out


OBSKEY = {'T': 't_obs', 'A': 'a_obs', 'V': 'v_obs'}


def augment(b, rng, p_aug=0.5):
    """Their augmentation family: random subset x rate x 1-3 segments, random placement."""
    if rng.random() >= p_aug:
        return b, None
    B, Tn = b['T'].shape[0], b['T'].shape[1]
    ar = torch.arange(Tn, device=b['T'].device)[None, :]
    obs = {'T': b['t_obs'], 'A': b['a_obs'], 'V': b['v_obs']}
    keep = {m: torch.ones(B, Tn, device=b['T'].device) for m in "TAV"}
    for m in "TAV":
        if rng.random() >= 0.5:
            continue
        rate = float(rng.choice(RATES))
        w = obs[m].sum(1, keepdim=True)
        k = (rate * w).round().clamp(min=1)
        n_seg = int(rng.integers(1, 4))
        for _ in range(n_seg):
            st = (torch.rand(B, 1, device=b['T'].device) * w.clamp_min(1)).floor()
            dist = torch.where(obs[m] > 0, (ar - st) % Tn, Tn + 1)
            keep[m] *= (dist.argsort(1).argsort(1) >= (k / n_seg).round().clamp_min(1)).float()
    out = dict(b)
    for m in "TAV":
        new_obs = obs[m] * keep[m]
        out[OBSKEY[m]] = new_obs
        out[m] = b[m] * keep[m][..., None]
    return out, keep


def train(seed=2026, epochs=140, lr=5e-5, wd=0.05, drop=0.4, dh=64, dmodel=128,
          rec=False, missing_token=False, label_smooth=0.0, gate_prior=False,
          lam_rec=0.1, bs=64, patience=30, verbose=False):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    d = get()
    def mk(s):
        return dict(T=torch.tensor(d[f'{s}_text'], device=DEV),
                    t_obs=torch.tensor(d[f'{s}_attn'], device=DEV),
                    A=torch.tensor(d[f'{s}_audio'], device=DEV),
                    a_obs=torch.tensor((~np.isclose(d[f'{s}_audio'], 0).all(-1)).astype(np.float32), device=DEV),
                    V=torch.tensor(d[f'{s}_vision'], device=DEV),
                    v_obs=torch.tensor((~np.isclose(d[f'{s}_vision'], 0).all(-1)).astype(np.float32), device=DEV),
                    y=torch.tensor(d[f'{s}_ycls'], device=DEV), r=torch.tensor(d[f'{s}_yreg'], device=DEV))
    # standardise with the v5 convention (prefix rows for text)
    mu = d['train_text'][d['train_attn'].astype(bool)].mean(0)
    sd = d['train_text'][d['train_attn'].astype(bool)].std(0) + 1e-6
    tr, va = mk('train'), mk('valid')
    for b in (tr, va):
        b['T'] = ((b['T'] - torch.tensor(mu, device=DEV)) / torch.tensor(sd, device=DEV)) * b['t_obs'][..., None]
    for m, k in (('audio', 'A'), ('vision', 'V')):
        o = (~np.isclose(d[f'train_{m}'], 0).all(-1))
        mu2 = d[f'train_{m}'][o].mean(0); sd2 = d[f'train_{m}'][o].std(0) + 1e-6
        for b in (tr, va):
            b[k] = ((b[k] - torch.tensor(mu2, device=DEV)) / torch.tensor(sd2, device=DEV)) * b[OBSKEY[k]][..., None]
    cnt = np.bincount(d['train_ycls'], minlength=3)
    cw = torch.tensor(len(d['train_ycls']) / (3 * np.maximum(cnt, 1)), dtype=torch.float32, device=DEV)
    model = Net2(dh=dh, dmodel=dmodel, drop=drop, rec=rec, missing_token=missing_token,
                 label_smooth=label_smooth, gate_prior=gate_prior).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
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
            loss = F.cross_entropy(out['logits'], sub['y'], weight=cw,
                                   label_smoothing=label_smooth) + \
                   F.smooth_l1_loss(out['reg'], sub['r'], beta=1.0)
            if rec and keep is not None:
                lr_ = 0
                for m in "TAV":
                    tm = ((clean[OBSKEY[m]] > 0) & (sub[OBSKEY[m]] <= 0))
                    if tm.sum() == 0:
                        continue
                    # per-sample mean over the artificially masked positions
                    tgt = (clean[m] * tm[..., None]).sum(1) / tm.sum(1, keepdim=True).clamp_min(1)
                    diff = F.smooth_l1_loss(out['recon'][m], tgt, beta=0.5, reduction='none').mean(-1)
                    lr_ = lr_ + (diff * (tm.sum(1) > 0)).sum() / (tm.sum(1) > 0).sum().clamp_min(1)
                loss = loss + lam_rec * lr_
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            o = model(va)
            pc = o['logits'].argmax(-1).cpu().numpy(); pr = o['reg'].cpu().numpy()
        from harness import cm, rm
        clean_m = {**cm(d['valid_ycls'], pc), **rm(d['valid_yreg'], pr)}
        torch.manual_seed(777)
        vb, _ = augment({k: v.clone() for k, v in va.items()}, np.random.default_rng(777), p_aug=1.0)
        with torch.no_grad():
            o2 = model(vb)
        r20 = cm(d['valid_ycls'], o2['logits'].argmax(-1).cpu().numpy())
        score = 0.5 * (clean_m['MacroF1'] + r20['MacroF1'])
        if score > best[0]:
            best = (score, dict(score=score, ep=ep + 1, clean=clean_m, r20=r20)); bad = 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (ep < 3 or (ep + 1) % 20 == 0):
            print(f'    ep{ep+1:3d} clean {clean_m["MacroF1"]:.4f} r20 {r20["MacroF1"]:.4f} score {score:.4f}', flush=True)
        if bad >= patience:
            break
    model.load_state_dict(state)
    return model, best[1]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', default='abl')
    ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args()
    cfgs = {
        'base':  dict(),
        'rec':   dict(rec=True),
        'miss':  dict(missing_token=True),
        'ls':    dict(label_smooth=0.05),
        'prior': dict(gate_prior=True),
        'full':  dict(rec=True, missing_token=True, label_smooth=0.05, gate_prior=True),
    }
    res = {}
    for name, kw in cfgs.items():
        t0 = time.time()
        _, r = train(seed=a.seed, epochs=140, verbose=True, **kw)
        res[name] = r
        print(f"[{name:6s}] score {r['score']:.4f} | clean {r['clean']['MacroF1']:.4f}/"
              f"{r['clean']['Accuracy']:.4f} MAE {r['clean']['MAE']:.4f} | r20 {r['r20']['MacroF1']:.4f} "
              f"| ep{r['ep']} [{time.time()-t0:.0f}s]", flush=True)
        json.dump(res, open(f'/tmp/eexp/exp12_s{a.seed}.json', 'w'), indent=2, default=float)
