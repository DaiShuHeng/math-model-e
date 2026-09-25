"""训练 / 评估 / 指标 通用流程。"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset

import config as C
import io_utils as U
import models as M


# --------------------------------------------------------------------------
# 数据集
# --------------------------------------------------------------------------
class MoseiDataset(Dataset):
    """按需做缺失增广；同时保留"自然掩码"以支持重建辅助损失。"""

    def __init__(self, samples: List[U.Sample], train: bool = False,
                 aug_prob: float = 0.0, miss_rates: Sequence[float] = C.MISS_RATES[1:],
                 miss_types: Sequence[str] = ("a", "v", "t", "av", "tv", "ta", "tav"),
                 n_segments_range=(1, 3), seed: int = 0):
        self.samples = samples
        self.train = train
        self.aug_prob = aug_prob
        self.miss_rates = list(miss_rates)
        self.miss_types = list(miss_types)
        self.n_segments_range = n_segments_range
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        nat = {m: s.mask[m].copy() for m in C.MODALITIES}
        Xt = {m: s.feats()[m].copy() for m in C.MODALITIES}
        obs = {m: nat[m].copy() for m in C.MODALITIES}

        if self.train and self.aug_prob > 0 and self.rng.random() < self.aug_prob:
            mt = str(self.rng.choice(self.miss_types))
            mr = float(self.rng.choice(self.miss_rates))
            ns = int(self.rng.integers(self.n_segments_range[0], self.n_segments_range[1] + 1))
            aug = U.apply_missing(s, mt, mr, self.rng, n_segments=ns)
            for m in C.MODALITIES:
                Xt[m] = aug.feats()[m]
                obs[m] = aug.mask[m]

        item = {
            "support": {m: torch.from_numpy(s.support[m].copy()) for m in C.MODALITIES},
            "X": {m: torch.from_numpy(np.ascontiguousarray(Xt[m])) for m in C.MODALITIES},
            "X_true": {m: torch.from_numpy(np.ascontiguousarray(s.feats()[m])) for m in C.MODALITIES},
            "m_obs": {m: torch.from_numpy(obs[m]) for m in C.MODALITIES},
            "m_nat": {m: torch.from_numpy(nat[m]) for m in C.MODALITIES},
        }
        item["y_cls"] = torch.tensor(int(s.label_cls) if s.label_cls is not None else -1, dtype=torch.long)
        item["y_reg"] = torch.tensor(float(s.label_reg) if s.label_reg is not None else np.nan, dtype=torch.float32)
        item["sid"] = s.sid
        return item


def collate(batch: List[dict]) -> dict:
    out = {}
    for key in ("X", "X_true", "m_obs", "m_nat", "support"):
        out[key] = {m: torch.stack([b[key][m] for b in batch]) for m in C.MODALITIES}
    out["y_cls"] = torch.stack([b["y_cls"] for b in batch])
    out["y_reg"] = torch.stack([b["y_reg"] for b in batch])
    out["sid"] = [b["sid"] for b in batch]
    return out


def to_device(batch: dict, dev) -> dict:
    out = {}
    for key in ("X", "X_true", "m_obs", "m_nat", "support"):
        out[key] = {m: v.to(dev) for m, v in batch[key].items()}
    out["y_cls"] = batch["y_cls"].to(dev)
    out["y_reg"] = batch["y_reg"].to(dev)
    out["sid"] = batch["sid"]
    return out


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def compute_metrics(y_cls: np.ndarray, p_cls: np.ndarray,
                    y_reg: np.ndarray, p_reg: np.ndarray) -> Dict[str, float]:
    m = {
        "acc": float(accuracy_score(y_cls, p_cls)),
        "f1_macro": float(f1_score(y_cls, p_cls, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_cls, p_cls, average="weighted", zero_division=0)),
        "mae": float(np.mean(np.abs(y_reg - p_reg))),
    }
    if len(y_reg) > 2 and np.std(y_reg) > 1e-9 and np.std(p_reg) > 1e-9:
        m["pearson"] = float(np.corrcoef(y_reg, p_reg)[0, 1])
    else:
        m["pearson"] = float("nan")
    # 二分类（非负 vs 负）辅助指标，便于与 MOSEI 文献对齐
    yb = (y_cls > 0).astype(int)
    pb = (p_cls > 0).astype(int)
    m["acc2"] = float(accuracy_score(yb, pb))
    m["f1_2"] = float(f1_score(yb, pb, average="binary", zero_division=0))
    return m


@torch.no_grad()
def predict(model: nn.Module, loader, dev, need_recon: bool = False) -> dict:
    model.eval()
    P_cls, P_reg, Y_cls, Y_reg, sids = [], [], [], [], []
    P_prob = []
    A_all, B_all = [], []
    for batch in loader:
        b = to_device(batch, dev)
        out = model(b["X"], b["m_obs"], support=b["support"], need_recon=need_recon)
        P_cls.append(out["logits"].argmax(-1).cpu().numpy())
        P_prob.append(out["logits"].softmax(-1).cpu().numpy())
        P_reg.append(out["pred_reg"].cpu().numpy())
        Y_cls.append(b["y_cls"].cpu().numpy())
        Y_reg.append(b["y_reg"].cpu().numpy())
        A_all.append(out["alpha"].cpu().numpy())
        B_all.append(out["beta"].cpu().numpy())
        sids.extend(b["sid"])
    res = dict(
        y_cls=np.concatenate(Y_cls), p_cls=np.concatenate(P_cls),
        y_reg=np.concatenate(Y_reg), p_reg=np.concatenate(P_reg),
        alpha=np.concatenate(A_all), beta=np.concatenate(B_all), sid=sids,
        prob=np.concatenate(P_prob),
    )
    labelled = (res["y_cls"] >= 0) & np.isfinite(res["y_reg"])
    res["metrics"] = compute_metrics(res["y_cls"][labelled], res["p_cls"][labelled], res["y_reg"][labelled], res["p_reg"][labelled]) if labelled.any() else {}
    return res


# --------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------
def class_weights(samples: List[U.Sample], dev) -> torch.Tensor:
    y = np.array([s.label_cls for s in samples])
    cnt = np.bincount(y, minlength=C.ModelCfg.n_classes).astype(np.float64)
    w = cnt.sum() / (C.ModelCfg.n_classes * np.maximum(cnt, 1))
    return torch.tensor(w, dtype=torch.float32, device=dev)


def train_model(model: nn.Module, train_samples, valid_samples, dev,
                cfg=C.TrainCfg, aug_prob: float = 0.5, tag: str = "model",
                log=print) -> dict:
    from torch.utils.data import DataLoader

    U.set_seed(cfg.seed)
    tr = MoseiDataset(train_samples, train=True, aug_prob=aug_prob, seed=cfg.seed)
    va = MoseiDataset(valid_samples, train=False)
    dl_tr = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                       num_workers=0, drop_last=False)
    dl_va = DataLoader(va, batch_size=128, shuffle=False, collate_fn=collate, num_workers=0)

    cw = class_weights(train_samples, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = max(1, len(dl_tr) * cfg.epochs)
    warm = max(1, int(steps * cfg.warmup_ratio))

    def lr_lambda(step):
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, steps - warm)
        return 0.5 * (1 + np.cos(np.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best = {"score": -1e9, "epoch": -1, "state": None, "metrics": None}
    hist = []
    patience = 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        agg = {"cls": 0.0, "reg": 0.0, "rec": 0.0, "n": 0}
        for batch in dl_tr:
            b = to_device(batch, dev)
            out = model(b["X"], b["m_obs"], support=b["support"], need_recon=cfg.lambda_rec > 0)
            loss, parts = M.compute_loss(out, b["y_cls"], b["y_reg"],
                                         b["X_true"], b["m_nat"], b["m_obs"],
                                         cfg, class_weight=cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            for k in ("cls", "reg", "rec"):
                agg[k] += parts[k]
            agg["n"] += 1

        res = predict(model, dl_va, dev)
        m = res["metrics"]
        # 选择准则：F1_macro 与 (-MAE) 的标准化组合
        score = m["f1_macro"] - 0.5 * (m["mae"] / C.ModelCfg.reg_range)
        hist.append({"epoch": ep, "train": {k: agg[k] / max(1, agg["n"]) for k in ("cls", "reg", "rec")},
                     "valid": m, "score": score, "sec": round(time.time() - t0, 1)})
        if score > best["score"]:
            best = {"score": score, "epoch": ep, "metrics": m,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            patience = 0
        else:
            patience += 1
        if ep % 5 == 0 or ep == 1 or patience == 0:
            log(f"  [{tag}] ep{ep:3d} loss={agg['cls']/max(1,agg['n']):.4f} "
                f"| val acc={m['acc']:.4f} f1={m['f1_macro']:.4f} mae={m['mae']:.4f} "
                f"r={m['pearson']:.3f} score={score:.4f} ({time.time()-t0:.1f}s)")
        if patience >= cfg.early_stop_patience:
            log(f"  [{tag}] 早停于 epoch {ep}（最优 epoch {best['epoch']}）")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"best": best, "history": hist}
