"""Exp13: does decision-threshold calibration (the teammate's biggest single lever)
transfer to my model, and does the missing token add on top?

All selection is VALID-only; TEST is used only for reporting.
"""
import sys, time, json, argparse, itertools
sys.path.insert(0, '/tmp/eexp')
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from exp2_strong import C, DEV
from exp12_integrate import Net2, augment, OBSKEY

torch.set_num_threads(8)


def mk(split, mu_t, sd_t, mu_a, sd_a, mu_v, sd_v):
    from harness import get
    d = get()
    b = dict(T=torch.tensor(d[f'{split}_text'], device=DEV),
             t_obs=torch.tensor(d[f'{split}_attn'], device=DEV),
             A=torch.tensor(d[f'{split}_audio'], device=DEV),
             a_obs=torch.tensor((~np.isclose(d[f'{split}_audio'], 0).all(-1)).astype(np.float32), device=DEV),
             V=torch.tensor(d[f'{split}_vision'], device=DEV),
             v_obs=torch.tensor((~np.isclose(d[f'{split}_vision'], 0).all(-1)).astype(np.float32), device=DEV),
             y=torch.tensor(d[f'{split}_ycls'], device=DEV), r=torch.tensor(d[f'{split}_yreg'], device=DEV))
    b['T'] = ((b['T'] - mu_t) / sd_t) * b['t_obs'][..., None]
    b['A'] = ((b['A'] - mu_a) / sd_a) * b['a_obs'][..., None]
    b['V'] = ((b['V'] - mu_v) / sd_v) * b['v_obs'][..., None]
    return b


def stats():
    from harness import get
    d = get()
    m = d['train_attn'].astype(bool)
    mu_t = torch.tensor(d['train_text'][m].mean(0), device=DEV)
    sd_t = torch.tensor(d['train_text'][m].std(0) + 1e-6, device=DEV)
    out = [mu_t, sd_t]
    for key in ('audio', 'vision'):
        o = ~np.isclose(d[f'train_{key}'], 0).all(-1)
        out += [torch.tensor(d[f'train_{key}'][o].mean(0), device=DEV),
                torch.tensor(d[f'train_{key}'][o].std(0) + 1e-6, device=DEV)]
    return out


def metrics(y, p):
    return dict(acc=float(accuracy_score(y, p)), f1=float(f1_score(y, p, average='macro')))


def fit_bias(prob, y):
    best = None
    for bn, bu in itertools.product([0., -.2, .2, -.4, .4], repeat=2):
        b = np.array([bn, bu, 0.])
        p = (np.log(prob.clip(1e-9)) + b).argmax(1)
        s = f1_score(y, p, average='macro')
        if best is None or s > best[0] + 1e-12:
            best = (s, b)
    return best[1]


def apply_bias(prob, b):
    lp = np.log(prob.clip(1e-9)) + np.asarray(b)
    p = np.exp(lp - lp.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def train(seed, epochs=140, lr=5e-5, wd=0.05, drop=0.4, dh=64, dmodel=128,
          missing_token=True, rec=False, cls_pow=1.0, lam_rec=0.1, bs=64, patience=30,
          verbose=False):
    from harness import get
    d = get()
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    mu_t, sd_t, mu_a, sd_a, mu_v, sd_v = stats()
    tr = mk('train', mu_t, sd_t, mu_a, sd_a, mu_v, sd_v)
    va = mk('valid', mu_t, sd_t, mu_a, sd_a, mu_v, sd_v)
    te = mk('test', mu_t, sd_t, mu_a, sd_a, mu_v, sd_v)
    cnt = np.bincount(d['train_ycls'], minlength=3)
    cw = torch.tensor(len(d['train_ycls']) / (3 * np.maximum(cnt, 1)), dtype=torch.float32, device=DEV)
    cw = (cw ** cls_pow); cw = cw / cw.mean()
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
                   F.smooth_l1_loss(out['reg'], sub['r'], beta=1.0)
            if rec and keep is not None:
                tot = 0
                for m in "TAV":
                    tm = ((clean[OBSKEY[m]] > 0) & (sub[OBSKEY[m]] <= 0))
                    if tm.sum() == 0:
                        continue
                    tgt = (clean[m] * tm[..., None]).sum(1) / tm.sum(1, keepdim=True).clamp_min(1)
                    diff = F.smooth_l1_loss(out['recon'][m], tgt, beta=0.5, reduction='none').mean(-1)
                    tot = tot + (diff * (tm.sum(1) > 0)).sum() / (tm.sum(1) > 0).sum().clamp_min(1)
                loss = loss + lam_rec * tot
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            o = model(va)
            pc = o['logits'].argmax(-1).cpu().numpy(); pr = o['reg'].cpu().numpy()
            prob = torch.softmax(o['logits'], -1).cpu().numpy()
        from harness import cm, rm
        cl = {**cm(d['valid_ycls'], pc), **rm(d['valid_yreg'], pr)}
        rng2 = np.random.default_rng(777)
        vb, _ = augment({k: v.clone() for k, v in va.items()}, rng2, p_aug=1.0)
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
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pv = torch.softmax(model(va)['logits'], -1).cpu().numpy()
        pt = torch.softmax(model(te)['logits'], -1).cpu().numpy()
    b = fit_bias(pv, d['valid_ycls'])
    res = dict(best=best[1], bias=[float(x) for x in b],
               valid_raw=metrics(d['valid_ycls'], pv.argmax(1)),
               valid_cal=metrics(d['valid_ycls'], apply_bias(pv, b).argmax(1)),
               test_raw=metrics(d['test_ycls'], pt.argmax(1)),
               test_cal=metrics(d['test_ycls'], apply_bias(pt, b).argmax(1)))
    return model, res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='2026')
    a = ap.parse_args()
    cfgs = {
        'base_pow1':      dict(missing_token=False, cls_pow=1.0),
        'miss_pow1':      dict(missing_token=True, cls_pow=1.0),
        'miss_pow05':     dict(missing_token=True, cls_pow=0.5),
        'miss_rec_pow1':  dict(missing_token=True, rec=True, cls_pow=1.0),
        'base_pow05':     dict(missing_token=False, cls_pow=0.5),
    }
    seeds = [int(s) for s in a.seeds.split(',')]
    res = {}
    for name, kw in cfgs.items():
        rows = []
        for sd in seeds:
            t0 = time.time()
            _, r = train(sd, verbose=(sd == seeds[0]), **kw)
            rows.append(r)
            print(f"  [{name} s{sd}] val {r['valid_raw']['f1']:.4f}->{r['valid_cal']['f1']:.4f} | "
                  f"TEST {r['test_raw']['acc']:.4f}/{r['test_raw']['f1']:.4f} -> "
                  f"{r['test_cal']['acc']:.4f}/{r['test_cal']['f1']:.4f} | bias {np.round(r['bias'],2).tolist()} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        agg = {k: float(np.mean([x[k]['f1'] for x in rows])) for k in ('valid_raw','valid_cal','test_raw','test_cal')}
        res[name] = dict(rows=rows, mean=agg)
        print(f"[{name:16s}] MEAN val {agg['valid_raw']:.4f}->{agg['valid_cal']:.4f} | "
              f"TEST f1 {agg['test_raw']:.4f}->{agg['test_cal']:.4f}\n", flush=True)
        json.dump(res, open('/tmp/eexp/exp13_calib.json','w'), indent=2, default=float)
