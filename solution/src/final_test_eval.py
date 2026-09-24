"""ONE-TIME final test evaluation of the frozen Q2 champion (q2_base_cw).

Config was selected on valid ONLY (10-config x 3-seed ablation, see
logs/ablation_v2_table.json); test is touched exactly once here and never
again. Evaluates the 3-seed ensemble on test under: clean, the adjudicated
附件3 rate-pool protocol (P_FILE=0.9 + empirical rates), and fixed r20/r40 —
identical protocol to eval_q2 (fixed per-sample damage seeds 0+index), so
numbers are directly comparable with logs/rate_curve_base_cw.json and
logs/head2head.json.

Run:
  cd solution && CUDA_VISIBLE_DEVICES=0 python -m src.final_test_eval
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.augment import WordLevelTAV
from src.data_adapter import Standardizer, load_splits
from src.datasets import Q2Dataset
from src.metrics import cls_metrics, head_agreement, reg_metrics
from src.models import Q2Model

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = []
    for c in CKPTS:
        ck = torch.load(c, map_location="cpu", weights_only=False)
        m = Q2Model(bert_dir=ck["args"].get("bert_dir", str(config.BERT_DIR)),
                    freeze_text=bool(ck["args"].get("freeze_text", 1))).to(device).eval()
        m.load_state_dict(ck["state_dict"])
        models.append(m)
    print(f"final config: q2_base_cw ensemble {[c.parent.name for c in CKPTS]}", flush=True)

    splits = load_splits()
    std = Standardizer.fit(splits["train"])

    results = {}
    for cond in CONDITIONS:
        ds = Q2Dataset(splits["test"], std, train=False,
                       augmenter=make_aug(cond), fixed_corrupt_seed=0)
        m = ensemble_metrics(models, ds, device)
        results[cond] = m
        print(f"TEST [{cond:>5}] Acc {m['Accuracy']:.4f} mF1 {m['MacroF1']:.4f} "
              f"MAE {m['MAE']:.4f} r {m['Pearson']:.4f}", flush=True)

    out = SOL / "logs" / "final_test_eval.json"
    out.write_text(json.dumps({"config": "q2_base_cw", "seeds": [c.parent.name for c in CKPTS],
                               "n_test": len(splits["test"]), "test": results},
                              ensure_ascii=False, indent=2))
    print(f"wrote {out} — test is now FROZEN, do not retune", flush=True)


if __name__ == "__main__":
    main()
