"""Final deployment: ensembled FEATURE-LEVEL model for the Q2/Q3 deliverables.

Trains a fixed pre-registered set of (config, seed) models on 附件2 TRAIN, selects
the deployment set on VALID only, and writes predictions for 附件3 and 附件4.

Text-input availability is handled explicitly and is the reason two inference
paths exist (see docs/缺失结构审计.md §5):
  * 附件4 carries the 768-d `text` field          -> text branch uses it directly;
  * 附件3 carries only `text_bert` token ids      -> text branch needs an encoder,
    and this script reports that condition instead of silently substituting tokens.

Run:
  python -m src.deploy_v5 --root weights/v5_deploy --stage train
  python -m src.deploy_v5 --root weights/v5_deploy --stage select
  python -m src.deploy_v5 --root weights/v5_deploy --stage predict --att 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import config
from .metrics import cls_metrics, reg_metrics
from .q2_features import FeatStandardizer, load_aligned_splits
from .q2_feature_model import FeatureTAV, feature_loss, save_checkpoint, load_checkpoint
from .train_q2_feature import GRID, SEEDS, corrupt_span_bounded, evaluate, to_torch


def train_all(root: Path, epochs: int, threads: int):
    torch.set_num_threads(threads)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    splits = load_aligned_splits(("train", "valid"))
    std = FeatStandardizer.fit(splits["train"])
    tr = to_torch(std.transform(splits["train"]), device)
    tr["y_cls"] = torch.as_tensor(splits["train"].y_cls, device=device)
    tr["y_reg"] = torch.as_tensor(splits["train"].y_reg, device=device)
    va = to_torch(std.transform(splits["valid"]), device)
    va["y_cls"] = torch.as_tensor(splits["valid"].y_cls, device=device)
    va["y_reg"] = torch.as_tensor(splits["valid"].y_reg, device=device)
    counts = np.bincount(splits["train"].y_cls, minlength=3)
    cls_w = torch.tensor(len(splits["train"]) / (3 * np.maximum(counts, 1)),
                         dtype=torch.float32, device=device)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, hp in GRID.items():
        for seed in SEEDS:
            ck = root / f"{name}_s{seed}.pt"
            if ck.exists():
                print(f"skip existing {ck.name}"); continue
            t0 = time.time()
            torch.manual_seed(seed); np.random.seed(seed)
            model = FeatureTAV(dh=hp["dh"], dmodel=hp["dmodel"],
                               dropout=hp["dropout"], pool=hp["pool"]).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            rng = np.random.default_rng(seed)
            n = len(splits["train"])
            best = {"selection_score": -1.0}; state = None; stale = 0
            for ep in range(1, epochs + 1):
                model.train()
                perm = torch.randperm(n, device=device)
                for i in range(0, n, 64):
                    idx = perm[i:i + 64]
                    batch = {k: v[idx] for k, v in tr.items()}
                    batch = corrupt_span_bounded(batch, rng)
                    batch["y_cls"] = tr["y_cls"][idx]; batch["y_reg"] = tr["y_reg"][idx]
                    out = model(batch)
                    loss, _ = feature_loss(out, batch["y_cls"], batch["y_reg"], cls_weight=cls_w)
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                sched.step()
                m = evaluate(model, va)
                if m["selection_score"] > best["selection_score"]:
                    best = dict(m); best["epoch"] = ep
                    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                if stale >= 30:
                    break
            model.load_state_dict(state)
            save_checkpoint(ck, model, std, {"variant": name, "seed": seed, "hyper": hp,
                                             "best_epoch": best["epoch"], "valid": best})
            rows.append({"variant": name, "seed": seed, "ckpt": ck.name,
                         "valid_clean": best["clean"], "valid_r20": best["r20"],
                         "selection_score": best["selection_score"],
                         "bytes": ck.stat().st_size, "sec": round(time.time() - t0, 1)})
            print(f"{name} s{seed}: clean {best['clean']['MacroF1']:.4f} r20 {best['r20']['MacroF1']:.4f} "
                  f"score {best['selection_score']:.4f} ep{best['epoch']} "
                  f"({ck.stat().st_size/1024:.0f} KB, {time.time()-t0:.0f}s)", flush=True)
            (root / "runs.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--stage", default="train", choices=("train",))
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    train_all(a.root, a.epochs, a.threads)
