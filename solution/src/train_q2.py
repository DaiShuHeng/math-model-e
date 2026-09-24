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
from .augment import WordLevelTAV
from .data_adapter import Standardizer, load_splits
from .datasets import Q2Dataset
from .metrics import cls_metrics, head_agreement, reg_metrics
from .models import Q2Model, q2_loss


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
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--bert-dir", type=str, default=str(config.BERT_DIR),
                    help="text encoder dir (bert-tiny self-contained / bert-base frozen)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(config.DEVICE)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits()
    std = Standardizer.fit(splits["train"])

    train_ds = Q2Dataset(splits["train"], std, train=True,
                         augmenter=None if args.no_aug else
                         WordLevelTAV(p_file=args.p_file, seed=args.seed))
    valid_clean = Q2Dataset(splits["valid"], std, train=False, augmenter=None)
    valid_corr = Q2Dataset(splits["valid"], std, train=False,
                           augmenter=WordLevelTAV(p_file=1.0, rate_pool=[0.2], seed=777),
                           fixed_corrupt_seed=0)

    def _worker_init(worker_id: int):
        # Independent augmentation RNG per worker/epoch: torch.initial_seed()
        # inside a worker = base_seed + worker_id, and base_seed is redrawn on
        # every epoch's iterator creation, so damage streams never repeat.
        info = torch.utils.data.get_worker_info()
        aug = getattr(info.dataset, "augmenter", None) if info else None
        if aug is not None and hasattr(aug, "reseed"):
            aug.reseed(int(torch.initial_seed()) % (2**31))

    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=False, generator=g,
                          worker_init_fn=_worker_init)
    vc_dl = DataLoader(valid_clean, batch_size=128, num_workers=2, pin_memory=True)
    vr_dl = DataLoader(valid_corr, batch_size=128, num_workers=2, pin_memory=True)

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

    best = {"MacroF1": -1.0}
    history = []
    cls_w = None
    if args.cls_weight:
        counts = np.bincount(splits["train"].y_cls, minlength=config.N_CLASSES)
        cls_w = torch.tensor(len(splits["train"]) / (config.N_CLASSES * counts),
                             dtype=torch.float32, device=device)
        print(f"class weights: {cls_w.cpu().numpy().round(3).tolist()}", flush=True)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        if args.freeze_text:
            model.bert.eval()
        tot, nb = 0.0, 0
        for batch in train_dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch)
                loss, _ = q2_loss(out, batch, lambda_cls=args.lambda_cls, cls_weight=cls_w)
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
               "valid_clean": mc, "valid_corrupt20": mr}
        history.append(rec)
        marker = ""
        if mc["MacroF1"] > best["MacroF1"]:
            best = dict(mc)
            best["epoch"] = ep
            torch.save({"state_dict": model.state_dict(), "args": vars(args),
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
        "args": vars(args), "trainable_params": n_trainable, "total_params": n_total,
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
