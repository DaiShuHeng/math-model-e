"""Q2 training: frozen-BERT-Tiny gated fusion model with word-level TAV damage.

Selection protocol (题目要求): parameters learned on TRAIN only; structure/
hyper-parameters/early stopping chosen on VALID only. TEST is never touched
here. Valid is evaluated both clean and under a FIXED corruption (identical
damaged samples for every model/config: per-sample seed = fixed_corrupt_seed(0)
+ index; the constructor seed is inert by design, see datasets.py).

Run:
  cd solution && python -m src.train_q2 --seed 2026 --out weights/q2/s2026
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .augment import WordLevelTAV, MixedBlockAugment
from .data_adapter import Standardizer, load_splits
from .datasets import Q2Dataset
from .metrics import cls_metrics, head_agreement, reg_metrics
from .models import Q2Model, q2_loss
from .consistency import confident_consistency


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    ys_c, ps_c, ys_r, ps_r = [], [], [], []
    gates = []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(batch)
        ps_c.append(out["logits"].argmax(-1).cpu().numpy())
        ps_r.append(out["reg"].float().cpu().numpy())
        ys_c.append(batch["y_cls"].cpu().numpy())
        ys_r.append(batch["y_reg"].cpu().numpy())
        gates.append(out["gate"].float().cpu().numpy())
    y_c, p_c = np.concatenate(ys_c), np.concatenate(ps_c)
    y_r, p_r = np.concatenate(ys_r), np.concatenate(ps_r)
    m = cls_metrics(y_c, p_c)
    m.update(reg_metrics(y_r, p_r))
    m["head_agreement"] = head_agreement(p_c, p_r)
    m["gate_mean"] = np.concatenate(gates).mean(axis=0).tolist()
    return m


def _worker_init(worker_id: int):
    # Top-level function is picklable under macOS/Windows spawn.
    info = torch.utils.data.get_worker_info()
    aug = getattr(info.dataset, "augmenter", None) if info else None
    if aug is not None: aug.reseed(int(torch.initial_seed()) % (2**31))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lambda-cls", type=float, default=1.0)
    ap.add_argument("--p-file", type=float, default=config.P_FILE)
    ap.add_argument("--freeze-text", type=int, default=1)
    ap.add_argument("--ft-lr", type=float, default=5e-5,
                    help="encoder LR when --freeze-text 0 (heads keep --lr)")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--no-aug", action="store_true",
                    help="ablation: train without corruption augmentation")
    ap.add_argument("--cls-weight", type=int, default=0,
                    help="1 = inverse-frequency class weights on CE (macro-F1 oriented)")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--bert-dir", type=str, default=str(config.BERT_DIR),
                    help="text encoder dir (bert-tiny self-contained / bert-base frozen)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument('--paired-clean', action='store_true', help='Supervise clean and corrupt views equally')
    ap.add_argument('--consistency-weight', type=float, default=0.0)
    ap.add_argument('--consistency-warmup', type=int, default=5)
    ap.add_argument('--cls-weight-power', type=float, default=1.0,
                    help='0.5 is square-root inverse frequency; 1 preserves prior baseline')
    args = ap.parse_args()
    if args.consistency_weight < 0 or args.consistency_warmup < 1 or not 0 <= args.cls_weight_power <= 1:
        ap.error('Invalid optimization parameters')
    if args.consistency_weight and not args.paired_clean:
        ap.error('--consistency-weight requires --paired-clean')

    set_seed(args.seed)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "best.pt").exists():
        raise FileExistsError("Choose a new run directory; existing checkpoint will not be overwritten")

    splits = load_splits()
    std = Standardizer.fit(splits["train"])

    train_ds = Q2Dataset(splits["train"], std, train=True, paired_clean=args.paired_clean,
                         augmenter=None if args.no_aug else
                         MixedBlockAugment(p_file=args.p_file, seed=args.seed))
    valid_clean = Q2Dataset(splits["valid"], std, train=False, augmenter=None)
    valid_corr = Q2Dataset(splits["valid"], std, train=False,
                           augmenter=WordLevelTAV(p_file=1.0, rate_pool=[0.2], seed=777),
                           fixed_corrupt_seed=0)

    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda",
                          drop_last=False, generator=g,
                          worker_init_fn=_worker_init)
    vc_dl = DataLoader(valid_clean, batch_size=128, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    vr_dl = DataLoader(valid_corr, batch_size=128, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = Q2Model(bert_dir=args.bert_dir,
                    freeze_text=bool(args.freeze_text)).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    if args.freeze_text:
        groups = [{"params": params, "lr": args.lr}]
    else:
        bert_ids = {id(p) for p in model.bert.parameters()}
        enc = [p for p in params if id(p) in bert_ids]
        heads = [p for p in params if id(p) not in bert_ids]
        groups = [{"params": enc, "lr": args.ft_lr}, {"params": heads, "lr": args.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = {"selection_score": -1.0}
    history = []
    cls_w = None
    if args.cls_weight:
        counts = np.bincount(splits["train"].y_cls, minlength=config.N_CLASSES)
        cls_w = torch.tensor(len(splits["train"]) / (config.N_CLASSES * np.maximum(counts, 1)),
                             dtype=torch.float32, device=device)
        cls_w = cls_w.pow(args.cls_weight_power)
        cls_w = cls_w / cls_w.mean()
        print(f"class weights: {cls_w.cpu().numpy().round(3).tolist()}", flush=True)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        if args.freeze_text:
            model.bert.eval()
        tot, nb = 0.0, 0
        consistency_total, reliable_total = 0.0, 0.0
        for batch in train_dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
                out = model(batch)
                loss, _ = q2_loss(out, batch, lambda_cls=args.lambda_cls, cls_weight=cls_w)
                if args.paired_clean:
                    clean_batch = {k.removeprefix('clean_'): v for k,v in batch.items() if k.startswith('clean_')}
                    clean_batch.update(y_cls=batch['y_cls'], y_reg=batch['y_reg'])
                    clean_out = model(clean_batch)
                    clean_loss, _ = q2_loss(clean_out, clean_batch, lambda_cls=args.lambda_cls, cls_weight=cls_w)
                    loss = 0.5 * (loss + clean_loss)
                    if args.consistency_weight:
                        consistency, reliable = confident_consistency(clean_out, out, batch['y_cls'])
                        coefficient = args.consistency_weight * min(1.0, ep / args.consistency_warmup)
                        loss = loss + coefficient * consistency
                        consistency_total += float(consistency.detach())
                        reliable_total += float(reliable.detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()

        mc = evaluate(model, vc_dl, device)
        mr = evaluate(model, vr_dl, device)
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1),
               "valid_clean": mc, "valid_corrupt20": mr,
               "consistency_loss": consistency_total / max(nb,1),
               "reliable_teacher_fraction": reliable_total / max(nb,1)}
        history.append(rec)
        marker = ""
        selection_score = 0.5 * (mc["MacroF1"] + mr["MacroF1"])
        if selection_score > best["selection_score"]:
            best = dict(mc)
            best["selection_score"] = selection_score
            best["epoch"] = ep
            torch.save({"protocol_version": config.PROTOCOL_VERSION, "state_dict": model.state_dict(), "args": vars(args),
                        "bert_config": model.bert.config.to_dict(),
                        "normalizer": {k: getattr(std, k).tolist() for k in ("a_mean","a_std","v_mean","v_std")},
                        "valid_clean": mc, "valid_corrupt20": mr, "epoch": ep},
                       out_dir / "best.pt")
            marker = " *"
        print(f"[ep {ep:02d}] loss {tot/max(nb,1):.4f} | clean F1 {mc['MacroF1']:.4f} "
              f"Acc {mc['Accuracy']:.4f} MAE {mc['MAE']:.4f} r {mc['Pearson']:.4f} | "
              f"corrupt F1 {mr['MacroF1']:.4f} MAE {mr['MAE']:.4f}{marker}", flush=True)
        if ep - best.get("epoch", 0) >= args.patience:
            print(f"early stop at ep {ep} (best ep {best['epoch']})", flush=True)
            break

    # reload best and final report
    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    final_clean = evaluate(model, vc_dl, device)
    final_corr = evaluate(model, vr_dl, device)
    report = {
        "protocol_version": config.PROTOCOL_VERSION, "args": vars(args), "trainable_params": n_trainable, "total_params": n_total,
        "wall_time_sec": round(time.time() - t0, 1),
        "best_epoch": ck["epoch"],
        "valid_clean": final_clean, "valid_corrupt20": final_corr,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"valid_clean": final_clean, "valid_corrupt20": final_corr},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
