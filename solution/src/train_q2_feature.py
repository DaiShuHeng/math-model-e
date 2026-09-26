"""v5 Q2 training entry point (feature-level model).

Reads TRAIN/VALID from 附件2 aligned_50.pkl only; TEST and 附件3/4 are never
loaded here.  Model selection uses the pre-registered score
    mean(clean MacroF1, continuous TAV r20 MacroF1)
exactly as in the v3/v4 protocol, so the numbers are comparable across rounds.

Corruption protocol (`span_bounded_v1`)
---------------------------------------
Contiguous blocks are drawn *inside the observed support* of each modality.
This is not a stylistic choice: the audit in `docs/缺失结构审计.md` shows that in
the provided aligned files 100% of zero audio/vision rows lie outside the text
token span and ~96% of samples have audio/vision support equal to that span.
Deleting positions outside the support would train the model on inputs that
cannot occur, and would under-sample the deletions that do occur.

Run (from `solution/`):
  python -m src.train_q2_feature --out weights/v5/s2026 --seed 2026
  python -m src.train_q2_feature --plan            # show the fixed grid, no training
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import config
from .q2_features import FeatStandardizer, load_aligned_splits
from .q2_feature_model import FeatureTAV, feature_loss, save_checkpoint

RATES = (0.1, 0.2, 0.3, 0.4, 0.5)
SEEDS = (2026, 7, 42)
GRID = {  # pre-registered candidate set for this round
    "v5a": dict(dh=64, dmodel=128, dropout=0.4, pool="a", lr=5e-5, wd=0.05),
    "v5b": dict(dh=64, dmodel=128, dropout=0.5, pool="a", lr=5e-5, wd=0.10),
    "v5c": dict(dh=64, dmodel=128, dropout=0.4, pool="sma", lr=5e-5, wd=0.05),
    "v5d": dict(dh=96, dmodel=192, dropout=0.5, pool="a", lr=3e-5, wd=0.10),
    # v5e/v5f: same as v5a/v5c with the learnable missing token.  Adopted after
    # the cross-team ablation (docs/两队方案对比与整合分析.md §4.2) showed it is
    # the only teammate idea that is consistently positive on this architecture.
    "v5e": dict(dh=64, dmodel=128, dropout=0.4, pool="a", lr=5e-5, wd=0.05, missing_token=True),
    "v5f": dict(dh=64, dmodel=128, dropout=0.4, pool="sma", lr=5e-5, wd=0.05, missing_token=True),
}


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_torch(arr: dict, device) -> dict:
    return {k: torch.as_tensor(v, device=device) for k, v in arr.items()}


def corrupt_span_bounded(batch: dict, rng: np.random.Generator, p_mod: float = 0.45,
                         rate_choice=None):
    """Delete a contiguous run / a scattered subset *inside the observed support*.

    The deletion budget keeps the original meaning: ``k = round(r * |W|)`` where
    ``W`` is the sample's word-token span (the text observation mask, which equals
    the aligned valid-token count).  The positions actually removed are drawn from
    the modality's observed support intersected with ``W`` — for audio/vision that
    intersection is where the features physically exist (audit §2.1), so no
    deletion is wasted on structural padding.
    """
    out = dict(batch)
    B, Tn = batch["T"].shape[0], batch["T"].shape[1]
    ar = torch.arange(Tn, device=batch["T"].device)[None, :]
    span = batch["t_obs"]
    w_span = span.sum(1, keepdim=True)      # |W|: word-token span length
    for mod, okey in (("T", "t_obs"), ("A", "a_obs"), ("V", "v_obs")):
        if rng.random() >= p_mod:
            continue
        obs = batch[okey]
        valid = (obs > 0)
        if mod != "T":
            valid = valid & (span > 0)
        if valid.sum() == 0:
            continue
        rate = float(rng.choice(rate_choice if rate_choice is not None else RATES))
        # k is the same for every modality of a sample, which is what makes the
        # TAV condition a synchronised continuous-block intervention.
        k = (rate * w_span).round().clamp(min=1)
        if rng.random() < 0.5:
            rank = torch.rand(B, Tn, device=batch["T"].device).masked_fill(~valid, 2.0)
            del_mask = valid & (rank.argsort(1).argsort(1) < k)
        else:
            st = (torch.rand(B, 1, device=batch["T"].device) * w_span.clamp_min(1)).floor()
            dist = torch.where(valid, (ar - st) % Tn, Tn + 1)
            del_mask = valid & (dist.argsort(1).argsort(1) < k)
        keep = (~del_mask).to(batch[mod].dtype)
        out[okey] = obs * keep
        out[mod] = batch[mod] * keep[..., None]
    return out


@torch.no_grad()
def evaluate(model: FeatureTAV, valid_t: dict, rounds: int = 5, seed: int = 777,
             rate: float = 0.2) -> dict:
    """Clean metrics + the continuous-TAV-r20 condition used by the selection score.

    ``r20`` averages `rounds` independent corruption draws with the rate forced to
    ``rate`` on all three modalities, so the condition is the same for every model
    and every seed; the per-round standard deviation is reported alongside.
    """
    from .metrics import cls_metrics, reg_metrics
    model.eval()
    o = model(valid_t)
    prob = torch.softmax(o["logits"], -1)
    y_cls = valid_t["y_cls"].cpu().numpy(); y_reg = valid_t["y_reg"].cpu().numpy()
    clean = {**cls_metrics(y_cls, prob.argmax(-1).cpu().numpy()),
             **reg_metrics(y_reg, o["reg"].cpu().numpy())}

    forced = _force_rate(rate)
    f1s, accs, maes = [], [], []
    for r in range(rounds):
        rng = np.random.default_rng(seed + r)
        b = corrupt_span_bounded(valid_t, rng, p_mod=1.0, rate_choice=forced)
        ob = model(b)
        pb = torch.softmax(ob["logits"], -1).cpu().numpy()
        f1s.append(cls_metrics(y_cls, pb.argmax(-1))["MacroF1"])
        accs.append(cls_metrics(y_cls, pb.argmax(-1))["Accuracy"])
        maes.append(reg_metrics(y_reg, ob["reg"].cpu().numpy())["MAE"])
    r20 = {"MacroF1": float(np.mean(f1s)), "MacroF1_std": float(np.std(f1s)),
           "Accuracy": float(np.mean(accs)), "Accuracy_std": float(np.std(accs)),
           "MAE": float(np.mean(maes))}
    return {"clean": clean, "r20": r20,
            "selection_score": 0.5 * (clean["MacroF1"] + r20["MacroF1"])}


def _force_rate(rate: float):
    """Marker recognised by corrupt_span_bounded to pin the sampled rate."""
    return (float(rate),)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v5a", choices=sorted(GRID))
    ap.add_argument("--seed", type=int, default=config.SEED, choices=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--p-aug", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=str)
    ap.add_argument("--plan", action="store_true", help="print the fixed grid and exit")
    args = ap.parse_args()

    if args.plan:
        print(json.dumps({"protocol": "span_bounded_v1", "grid": GRID, "seeds": SEEDS,
                          "rates": RATES, "selection": "mean(clean MacroF1, TAV r20 MacroF1)",
                          "scope": "TRAIN/VALID only"}, ensure_ascii=False, indent=2))
        return
    if not args.out:
        ap.error("--out is required unless --plan")
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "best.pt").exists():
        raise FileExistsError("Refusing to overwrite an existing checkpoint; use a new --out")

    splits = load_aligned_splits(("train", "valid"))
    std = FeatStandardizer.fit(splits["train"])
    tr = std.transform(splits["train"]); va = std.transform(splits["valid"])
    tr_t = to_torch(tr, device)
    tr_t["y_cls"] = torch.as_tensor(splits["train"].y_cls, device=device)
    tr_t["y_reg"] = torch.as_tensor(splits["train"].y_reg, device=device)
    va_t = to_torch(va, device)
    va_t["y_cls"] = torch.as_tensor(splits["valid"].y_cls, device=device)
    va_t["y_reg"] = torch.as_tensor(splits["valid"].y_reg, device=device)

    hp = GRID[args.variant]
    model = FeatureTAV(dh=hp["dh"], dmodel=hp["dmodel"], dropout=hp["dropout"],
                       pool=hp["pool"], missing_token=hp.get("missing_token", False)).to(device)
    counts = np.bincount(splits["train"].y_cls, minlength=3)
    cls_w = torch.tensor(len(splits["train"]) / (3 * np.maximum(counts, 1)),
                         dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(args.seed)

    n = len(splits["train"]); best = {"selection_score": -1.0}; best_state = None
    stale = 0; history = []; t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot, nb = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch = {k: v[idx] for k, v in tr_t.items()}
            if rng.random() < args.p_aug:
                batch = corrupt_span_bounded(batch, rng)
                batch["y_cls"] = tr_t["y_cls"][idx]; batch["y_reg"] = tr_t["y_reg"][idx]
            out = model(batch)
            loss, parts = feature_loss(out, batch["y_cls"], batch["y_reg"], cls_weight=cls_w)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        m = evaluate(model, va_t)
        history.append({"epoch": ep, "train_loss": tot / max(nb, 1), **m})
        if m["selection_score"] > best["selection_score"]:
            best = dict(m); best["epoch"] = ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
            mark = " *"
        else:
            stale += 1; mark = ""
        print(f"[ep {ep:03d}] loss {tot/max(nb,1):.4f} | clean F1 {m['clean']['MacroF1']:.4f} "
              f"Acc {m['clean']['Accuracy']:.4f} MAE {m['clean']['MAE']:.4f} | "
              f"r20 F1 {m['r20']['MacroF1']:.4f} | score {m['selection_score']:.4f}{mark}", flush=True)
        if stale >= args.patience:
            print(f"early stop at ep {ep} (best ep {best['epoch']})", flush=True)
            break

    model.load_state_dict(best_state)
    save_checkpoint(out_dir / "best.pt", model, std, {
        "variant": args.variant, "seed": args.seed, "hyper": hp,
        "selection": "mean(clean MacroF1, continuous TAV r20 MacroF1)",
        "corruption": "span_bounded_v1", "best_epoch": best["epoch"],
        "valid": {"clean": best["clean"], "r20": best["r20"]},
        "data_fingerprint": hashlib.sha256(
            np.ascontiguousarray(splits["train"].text[0]).tobytes()).hexdigest()[:16],
    })
    report = {"variant": args.variant, "seed": args.seed, "hyper": hp,
              "wall_time_sec": round(time.time() - t0, 1), "best_epoch": best["epoch"],
              "valid_clean": best["clean"], "valid_r20": best["r20"],
              "selection_score": best["selection_score"], "history": history}
    (out_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"valid_clean": best["clean"], "valid_r20": best["r20"],
                      "selection_score": best["selection_score"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
