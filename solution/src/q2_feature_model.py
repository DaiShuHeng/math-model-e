"""v5 Q2/Q3 model: evidence-pooled modality branches + availability-aware gated fusion.

Design notes
------------
* Each modality is reduced to a fixed-length representation by three *masked*
  statistics over the 50 aligned positions: masked mean, masked max and a learned
  attention pool.  All three respect the observation mask, so padding and missing
  rows never enter the statistics (`MISSING_SENTINEL` is used only to keep
  `max` finite, and rows with no observation at all are forced to zero).
* A modality whose observation count is zero is switched off by the gate
  (`-1e4` before softmax) and contributes an explicit learned `null` vector for
  text, so a text-less sample degrades gracefully instead of injecting noise.
* Gate weights are *availability-aware mixture coefficients*; they are reported as
  model internals and are not treated as causal modality importance (that is what
  the Shapley decomposition in `q3_shapley_v5.py` is for).

Everything the checkpoint needs in order to run on a fresh machine is inside the
checkpoint: architecture config, train-fitted normalisation, and the protocol id.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MISSING_SENTINEL = -1e4
PROTOCOL_VERSION = "feature_level_v5_span_bounded"


def masked_stats(h: torch.Tensor, m: torch.Tensor, q: torch.Tensor):
    """h:(B,T,D) m:(B,T) 1=keep. Returns (mean, max, attn_pool, attn_weights)."""
    mm = m[..., None]
    denom = mm.sum(1).clamp_min(1e-6)
    mean = (h * mm).sum(1) / denom
    mx = h.masked_fill(mm <= 0, MISSING_SENTINEL).max(1).values
    mx = torch.where(m.sum(1, keepdim=True) > 0, mx, torch.zeros_like(mx))
    logits = torch.einsum("btd,d->bt", h, q) / np.sqrt(h.size(-1))
    logits = logits.masked_fill(mm.squeeze(-1) <= 0, MISSING_SENTINEL)
    w = torch.softmax(logits, dim=-1) * (m > 0).to(logits.dtype)
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
    pool = torch.einsum("bt,btd->bd", w, h)
    return mean, mx, pool, w


class FeatureBranch(nn.Module):
    """LayerNorm -> linear projection -> masked statistics -> dropout.

    ``missing_token`` (adopted from the teammate package's encoder, see
    docs/两队方案对比与整合分析.md §4.2) replaces the zeroed rows with a learned
    vector instead of leaving them at zero after the mask multiply.  The masked
    statistics still only count observed positions, so the token is a *placeholder*
    the encoder can use to represent "this position is missing" rather than a
    value that leaks into the pooled representation.  Ablation: +0.0095 valid
    selection score, the only consistently positive change among the four
    teammate ideas that were tested.
    """

    def __init__(self, in_dim: int, dh: int, dropout: float, pool: str = "sma",
                 missing_token: bool = False):
        super().__init__()
        self.pool = pool
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dh), nn.GELU())
        self.query = nn.Parameter(torch.randn(dh) / np.sqrt(dh))
        self.drop = nn.Dropout(dropout)
        self.out_dim = len(pool) * dh
        if missing_token:
            self.missing = nn.Parameter(torch.zeros(1, 1, dh))
            nn.init.normal_(self.missing, std=0.02)
        else:
            self.register_parameter("missing", None)

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        h = self.proj(x)
        if self.missing is not None:
            mm = (m > 0)[..., None].to(h.dtype)
            h = h * mm + self.missing * (1.0 - mm)
        mean, mx, pool, w = masked_stats(h, m, self.query)
        parts = {"s": mean, "m": mx, "a": pool}
        rep = torch.cat([parts[c] for c in self.pool], dim=-1)
        avail = m.sum(1, keepdim=True) / m.size(1)          # observed fraction
        return self.drop(rep) * (avail > 0).to(rep.dtype), avail, w


class FeatureTAV(nn.Module):
    def __init__(self, dh: int = 64, dmodel: int = 128, dropout: float = 0.4, pool: str = "a",
                 missing_token: bool = False):
        super().__init__()
        self.cfg = dict(dh=dh, dmodel=dmodel, dropout=dropout, pool=pool,
                        missing_token=missing_token)
        self.text = FeatureBranch(768, dh, dropout, pool, missing_token)
        self.audio = FeatureBranch(74, dh, dropout, pool, missing_token)
        self.vision = FeatureBranch(35, dh, dropout, pool, missing_token)
        total = self.text.out_dim * 3
        self.null_text = nn.Parameter(torch.zeros(self.text.out_dim))
        self.gate = nn.Sequential(nn.Linear(total + 3, 128), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(128, 3))
        self.fuse = nn.Sequential(nn.LayerNorm(total), nn.Linear(total, dmodel), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(dmodel, dmodel), nn.GELU())
        self.head_cls = nn.Linear(dmodel, 3)
        self.head_reg = nn.Linear(dmodel, 1)

    def forward(self, batch: dict, corrupt=None):
        if corrupt is not None:
            batch = corrupt(batch)
        rt, at, wt = self.text(batch["T"], batch["t_obs"])
        ra, aa, _ = self.audio(batch["A"], batch["a_obs"])
        rv, av, _ = self.vision(batch["V"], batch["v_obs"])
        rt = torch.where(at > 0, rt, self.null_text.expand_as(rt))
        reps = torch.cat([rt, ra, rv], dim=-1)
        avails = torch.cat([at, aa, av], dim=-1)
        logits = self.gate(torch.cat([reps, avails], dim=-1))
        logits = logits.masked_fill(avails <= 0, MISSING_SENTINEL)
        gate = torch.softmax(logits, dim=-1)
        fused = (reps.view(reps.size(0), 3, -1) * gate[..., None]).reshape(reps.size(0), -1)
        z = self.fuse(fused)
        return {"logits": self.head_cls(z),
                "reg": 3 * torch.tanh(self.head_reg(z)).squeeze(-1),
                "gate": gate, "avail": avails,
                "text_attn": wt}

    @torch.no_grad()
    def predict(self, arrays: dict, device="cpu", batch: int = 128) -> dict:
        """arrays from FeatStandardizer.transform; returns numpy outputs."""
        self.eval().to(device)
        n = len(arrays["T"])
        out = {"prob": [], "reg": [], "gate": [], "text_attn": []}
        for i in range(0, n, batch):
            b = {k: torch.as_tensor(v[i:i + batch], device=device) for k, v in arrays.items()}
            o = self.forward(b)
            out["prob"].append(torch.softmax(o["logits"], -1).cpu().numpy())
            out["reg"].append(o["reg"].cpu().numpy())
            out["gate"].append(o["gate"].cpu().numpy())
            out["text_attn"].append(o["text_attn"].cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}


def feature_loss(out: dict, y_cls: torch.Tensor, y_reg: torch.Tensor,
                 cls_weight=None, lambda_cls: float = 1.0):
    ce = F.cross_entropy(out["logits"], y_cls, weight=cls_weight)
    hub = F.smooth_l1_loss(out["reg"], y_reg, beta=1.0)
    return hub + lambda_cls * ce, {"ce": float(ce.detach()), "huber": float(hub.detach())}


def save_checkpoint(path, model: FeatureTAV, std, extra: dict):
    payload = {"protocol_version": PROTOCOL_VERSION, "model_cfg": model.cfg,
               "state_dict": model.state_dict(), "normalizer": std.to_json()}
    payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device="cpu") -> tuple[FeatureTAV, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Refusing checkpoint with protocol {ck.get('protocol_version')!r}")
    model = FeatureTAV(**ck["model_cfg"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck
