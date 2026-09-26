"""Exp16: does the SELECTION CRITERION explain the gap, and can I close it?

Test A: early-stop my architecture by THEIR score = valid_f1 - 0.5*(valid_mae/3).
Test B: keep my score, but train a configuration whose validation MAE is much lower.
Both use identical data, augmentation and budget (their augmentation family).
"""
import sys, time, json, argparse
sys.path.insert(0, '/tmp/eexp')
import numpy as np, torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from exp2_strong import DEV
from exp12_integrate import Net2, augment, OBSKEY, DIM
from exp13_calib import mk, stats, metrics, fit_bias, apply_bias


def train(seed, criterion="mine", lam_reg=1.0, hidden_rec=True, epochs=140, bs=64,
          patience=30, lr=5e-5, wd=0.05, drop=0.4, dh=64, dmodel=128, rec=True,
          missing_token=True, verbose=False):
    from harness import get
    d = get()
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    mu_t, sd_t, mu_a, sd_a, mu_v, sd_v = stats()
    tr = mk('train', mu_t, sd_t, mu_a, sd_a, mu_v, sd_v)
    va = mk('valid', mu_t, sd_t, mu_a, sd_a, mu_v, sd_v)
    te = mk('test', mu_t, sd_t, mu_a, sd_a, mu_v, sd_v)
    cnt = np.bincount(d['train_ycls'], minlength=3)
    cw = torch.tensor(len(d['train_ycls']) / (3 * np.maximum(cnt, 1)), dtype=torch.float32, device=DEV)
    model = Net2(dh=dh, dmodel=dmodel, drop=drop, rec=rec, missing_token=missing_token).to(DEV)
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
            loss = F.cross_entropy(out['logits'], sub['y'], weight=cw) + \
                   lam_reg * F.smooth_l1_loss(out['reg'], sub['r'], beta=0.5)
            if rec and keep is not None:
                tot = 0
                for m in "TAV":
                    tm = ((clean[OBSKEY[m]] > 0) & (sub[OBSKEY[m]] <= 0))
                    if tm.sum() == 0:
                        continue
                    tgt = (clean[m] * tm[..., None]).sum(1) / tm.sum(1, keepdim=True).clamp_min(1)
                    dif = F.smooth_l1_loss(out['recon'][m], tgt, beta=0.5, reduction='none').mean(-1)
                    tot = tot + (dif * (tm.sum(1) > 0)).sum() / (tm.sum(1) > 0).sum().clamp_min(1)
                loss = loss + 0.1 * tot
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            o = model(va)
            pc = o['logits'].argmax(-1).cpu().numpy(); pr = o['reg'].cpu().numpy()
        vf1 = f1_score(d['valid_ycls'], pc, average='macro')
        vmae = float(np.mean(np.abs(d['valid_yreg'] - pr)))
        vb, _ = augment({k: v.clone() for k, v in va.items()}, np.random.default_rng(777), p_aug=1.0)
        with torch.no_grad():
            r20 = f1_score(d['valid_ycls'], model(vb)['logits'].argmax(-1).cpu().numpy(), average='macro')
        s_mine = 0.5 * (vf1 + r20)
        s_theirs = vf1 - 0.5 * (vmae / 3.0)
        score = s_mine if criterion == "mine" else s_theirs
        if score > best[0]:
            best = (score, dict(score=score, ep=ep + 1, vf1=vf1, vmae=vmae,
                                s_mine=s_mine, s_theirs=s_theirs)); bad = 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (ep < 3 or (ep + 1) % 30 == 0):
            print(f'    ep{ep+1:3d} vf1 {vf1:.4f} vmae {vmae:.4f} r20 {r20:.4f} '
                  f'mine {s_mine:.4f} theirs {s_theirs:.4f}', flush=True)
        if bad >= patience:
            break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        pv = torch.softmax(model(va)['logits'], -1).cpu().numpy()
        pt = torch.softmax(model(te)['logits'], -1).cpu().numpy()
        rv = model(va)['reg'].cpu().numpy(); rt = model(te)['reg'].cpu().numpy()
    b = fit_bias(pv, d['valid_ycls'])
    return model, dict(best=best[1], bias=[float(x) for x in b],
                       valid=dict(f1=vf1, mae=vmae,
                                  test_f1=float(f1_score(d['test_ycls'], pt.argmax(1), average='macro')),
                                  test_acc=float((pt.argmax(1) == d['test_ycls']).mean())),
                       valid_cal=metrics(d['valid_ycls'], apply_bias(pv, b).argmax(1)),
                       test_raw=metrics(d['test_ycls'], pt.argmax(1)),
                       test_cal=metrics(d['test_ycls'], apply_bias(pt, b).argmax(1)),
                       test_mae=float(np.mean(np.abs(d['test_yreg'] - rt))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='2026,7,42')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    cfgs = {
        'crit_mine':   dict(criterion='mine'),
        'crit_theirs': dict(criterion='theirs'),
        'mae_focus':   dict(criterion='theirs', lam_reg=2.0, dh=96, dmodel=192, drop=0.3),
    }
    res = {}
    for name, kw in cfgs.items():
        rows = []
        for sd in seeds:
            t0 = time.time()
            _, r = train(sd, verbose=True, **kw)
            rows.append(r)
            print(f"  [{name} s{sd}] valF1 {r['best']['vf1']:.4f} valMAE {r['best']['vmae']:.4f} "
                  f"| TEST raw {r['test_raw']['acc']:.4f}/{r['test_raw']['f1']:.4f} MAE {r['test_mae']:.4f} "
                  f"| cal {r['test_cal']['acc']:.4f}/{r['test_cal']['f1']:.4f} [ep{r['best']['ep']}] "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        res[name] = dict(rows=rows,
                         mean={k: float(np.mean([x['test_raw'][k] if k in x['test_raw'] else
                                                 x['test_cal'][k] if k in x['test_cal'] else
                                                 x['test_mae'] for x in rows]))
                               for k in ('f1', 'acc', 'mae')} if False else {
                             'test_f1_raw': float(np.mean([x['test_raw']['f1'] for x in rows])),
                             'test_f1_cal': float(np.mean([x['test_cal']['f1'] for x in rows])),
                             'test_acc_raw': float(np.mean([x['test_raw']['acc'] for x in rows])),
                             'test_acc_cal': float(np.mean([x['test_cal']['acc'] for x in rows])),
                             'test_mae': float(np.mean([x['test_mae'] for x in rows])),
                             'valid_f1': float(np.mean([x['best']['vf1'] for x in rows])),
                             'valid_mae': float(np.mean([x['best']['vmae'] for x in rows]))})
        m = res[name]['mean']
        print(f"[{name:12s}] MEAN validF1 {m['valid_f1']:.4f} validMAE {m['valid_mae']:.4f} | "
              f"TEST f1 {m['test_f1_raw']:.4f}->{m['test_f1_cal']:.4f} acc {m['test_acc_raw']:.4f}->{m['test_acc_cal']:.4f} "
              f"MAE {m['test_mae']:.4f}\n", flush=True)
        json.dump(res, open('/tmp/eexp/exp16_criterion.json', 'w'), indent=2, default=float)
