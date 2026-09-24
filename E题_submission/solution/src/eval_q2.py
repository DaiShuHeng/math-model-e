"""Q2 evaluation sweeps for the paper: robustness vs missing rate, gate stats.

Protocol: every checkpoint is evaluated on VALID under a grid of word-level
damage rates {0, .1, .2, .3, .5, .7} with p_file=1.0 and a FIXED per-sample
seed (fixed_corrupt_seed(0) + index; the constructor's seed arg is inert —
see datasets.py), so all rates/configs see comparable corrupted copies.
TEST split is never touched here; the paper's test numbers come from a single
final-config run only after model selection is frozen.

Run:
  cd solution && python -m src.eval_q2 --ckpt weights/q2_ft/s2026/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .augment import WordLevelTAV
from .data_adapter import Standardizer, load_splits
from .datasets import Q2Dataset
from .metrics import cls_metrics, head_agreement, reg_metrics
from .models import Q2Model
from .train_q2 import evaluate

RATES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
BASE_SEED = 777


def rate_curve(ckpt_path: Path, rates=RATES, base_seed: int = BASE_SEED) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck["args"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Q2Model(bert_dir=args.get("bert_dir", str(config.BERT_DIR)),
                    freeze_text=bool(args.get("freeze_text", 1))).to(device)
    model.load_state_dict(ck["state_dict"])

    splits = load_splits()
    std = Standardizer.fit(splits["train"])
    out = {"ckpt": str(ckpt_path), "rates": rates, "points": []}
    for r in rates:
        ds = Q2Dataset(splits["valid"], std, train=False, augmenter=None if r == 0 else
                       WordLevelTAV(p_file=1.0, rate_pool=[r], seed=base_seed),
                       fixed_corrupt_seed=0)
        dl = DataLoader(ds, batch_size=128, num_workers=2, pin_memory=True)
        m = evaluate(model, dl, device)
        out["points"].append({"rate": r, **{k: v for k, v in m.items()}})
        print(f"  rate {r:.1f}: Acc {m['Accuracy']:.4f} mF1 {m['MacroF1']:.4f} "
              f"MAE {m['MAE']:.4f} r {m['Pearson']:.4f} "
              f"gate {np.round(m['gate_mean'], 3).tolist()}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="+", required=True)
    ap.add_argument("--out", type=str, default=None, help="output json (default: per-ckpt rate_curve.json)")
    args = ap.parse_args()
    results = {}
    for c in args.ckpt:
        p = Path(c)
        print(f"== {p}", flush=True)
        curve = rate_curve(p)
        if args.out:
            results[str(p)] = curve
        else:
            (p.parent / "rate_curve.json").write_text(
                json.dumps({str(p): curve}, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
