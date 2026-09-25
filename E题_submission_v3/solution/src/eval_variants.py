"""Q2 influence study: missing type / position / duration impact patterns.

Evaluates checkpoints on VALID under a fixed-seed corruption grid:
  - type:   which modalities are hit (t / a / v / a+v / t+a+v) at rate 0.2
  - position: head / mid / tail third of the word span at rate 0.2
  - duration: contiguous block vs scattered positions (rate 0.2, 0.4)
All variants produce identical damaged copies across configs: per-sample
seed = fixed_corrupt_seed(0) + index (the WordLevelTAV seed arg is inert).

Run:
  cd solution && python -m src.eval_variants --ckpt weights/q2_ft_cw/s*/best.pt
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
from .checkpoints import load_ensemble
from .train_q2 import evaluate

VARIANTS = (
    # (group, name, kwargs)  — rate pool fixed at 0.2 unless overridden
    ("clean", "clean", None),
    ("type", "T_only", dict(modalities=("t",))),
    ("type", "A_only", dict(modalities=("a",))),
    ("type", "V_only", dict(modalities=("v",))),
    ("type", "TA_only", dict(modalities=("t", "a"))),
    ("type", "TV_only", dict(modalities=("t", "v"))),
    ("type", "AV_only", dict(modalities=("a", "v"))),
    ("type", "TAV (default)", dict(modalities=("t", "a", "v"))),
    ("position", "head_block", dict(region="head")),
    ("position", "mid_block", dict(region="mid")),
    ("position", "tail_block", dict(region="tail")),
    ("duration", "scattered_r20", dict(contiguous=False)),
    ("duration", "contiguous_r20", dict(contiguous=True)),
    ("duration", "scattered_r40", dict(rate_pool=[0.4], contiguous=False)),
    ("duration", "contiguous_r40", dict(contiguous=True, rate_pool=[0.4])),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="+", required=True)
    ap.add_argument("--out", type=str, default="logs/revised_variant_grid.json")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    splits = load_splits()
    models, std = load_ensemble(args.ckpt, device)

    results = []
    for group, name, kw in VARIANTS:
        per_model = []
        gate_means = []
        for m in models:
            if kw is None:
                ds = Q2Dataset(splits["valid"], std, train=False, augmenter=None)
            else:
                # never pop() from the shared module-level VARIANTS dicts —
                # with >1 checkpoint that silently downgraded r40 variants to
                # r20 for models 2..N and corrupted the averaged rows.
                rate_pool = kw.get("rate_pool", [0.2])
                kwargs = {k: v for k, v in kw.items() if k != "rate_pool"}
                aug = WordLevelTAV(p_file=1.0, rate_pool=rate_pool, seed=777, **kwargs)
                ds = Q2Dataset(splits["valid"], std, train=False,
                               augmenter=aug, fixed_corrupt_seed=0)
            dl = DataLoader(ds, batch_size=128, num_workers=0, pin_memory=device.type == "cuda")
            mm = evaluate(m, dl, device)
            per_model.append(mm)
            gate_means.append(mm["gate_mean"])
        keys = ["Accuracy", "MacroF1", "WeightedF1", "MAE", "Pearson"]
        agg = {k: round(float(np.mean([d[k] for d in per_model])), 4) for k in keys}
        agg_std = {k: round(float(np.std([d[k] for d in per_model])), 4) for k in keys}
        agg["gate_mean"] = np.mean(gate_means, axis=0).round(4).tolist()
        results.append({"group": group, "variant": name, **agg, "std": agg_std})
        print(f"[{group:>9}] {name:<14} Acc {agg['Accuracy']:.4f} mF1 {agg['MacroF1']:.4f} "
              f"MAE {agg['MAE']:.4f} r {agg['Pearson']:.4f} gate {agg['gate_mean']}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
