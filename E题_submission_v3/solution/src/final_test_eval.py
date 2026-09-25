"""Evaluate a frozen revised model on TEST after validation selection.
Requires explicit checkpoint arguments; will not overwrite a previous report.
This script does not establish that historical test access occurred only once.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.augment import WordLevelTAV
from src.data_adapter import Standardizer, load_splits
from src.datasets import Q2Dataset
from src.metrics import cls_metrics, head_agreement, reg_metrics
from src.checkpoints import load_ensemble

SOL = Path(__file__).resolve().parents[1]
CKPTS = sorted((SOL / "weights" / "q2_base_cw").glob("s*/best.pt"))
CONDITIONS = ("clean", "pool", "r20", "r40")


def make_aug(cond: str):
    if cond == "clean":
        return None
    if cond == "pool":
        return WordLevelTAV(p_file=config.P_FILE, seed=777)   # seed inert with fixed_corrupt_seed
    r = 0.2 if cond == "r20" else 0.4
    return WordLevelTAV(p_file=1.0, rate_pool=[r], seed=777)


@torch.no_grad()
def ensemble_metrics(models, ds, device):
    dl = DataLoader(ds, batch_size=128, num_workers=0)
    probs, regs, ys_c, ys_r = [], [], [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        p = torch.stack([torch.softmax(m(batch)["logits"].float(), -1) for m in models]).mean(0)
        r = torch.stack([m(batch)["reg"].float() for m in models]).mean(0)
        probs.append(p.cpu()); regs.append(r.cpu())
        ys_c.append(batch["y_cls"].cpu()); ys_r.append(batch["y_reg"].cpu())
    prob = torch.cat(probs).numpy(); p_reg = torch.cat(regs).numpy()
    y_c = torch.cat(ys_c).numpy(); y_r = torch.cat(ys_r).numpy()
    m = cls_metrics(y_c, prob.argmax(1))
    m.update(reg_metrics(y_r, p_reg))
    m["head_agreement"] = head_agreement(prob.argmax(1), p_reg)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Test report exists; choose an explicit new path")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, std = load_ensemble(args.ckpt, device)
    splits = load_splits()

    results = {}
    for cond in CONDITIONS:
        ds = Q2Dataset(splits["test"], std, train=False,
                       augmenter=make_aug(cond), fixed_corrupt_seed=0)
        m = ensemble_metrics(models, ds, device)
        results[cond] = m
        print(f"TEST [{cond:>5}] Acc {m['Accuracy']:.4f} mF1 {m['MacroF1']:.4f} "
              f"MAE {m['MAE']:.4f} r {m['Pearson']:.4f}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"protocol_version": config.PROTOCOL_VERSION,
        "checkpoints": args.ckpt, "n_test": len(splits["test"]), "test": results},
        ensure_ascii=False, indent=2))
    print(f"wrote {out}")



if __name__ == "__main__":
    main()
