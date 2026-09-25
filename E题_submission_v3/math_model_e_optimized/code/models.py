"""Multimodal Transformer with support-aware fusion and two prediction heads.
Pooling and gate weights describe model internals; explanation claims require
input-removal experiments implemented in explain_v2.py and explanation_validation.py.
Problem 2 optionally reconstructs artificially hidden feature elements.
No complementarity regularization is used in the optimized experiments."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


# --------------------------------------------------------------------------
# 基础模块
# --------------------------------------------------------------------------
class ModalityEncoder(nn.Module):
    """单模态编码器：缺失令牌 + 位置编码 + Transformer + 证据注意力池化。"""

    def __init__(self, d_in: int, d_model: int, n_pos: int = C.N_POS,
                 n_heads: int = C.ModelCfg.n_heads, n_layers: int = C.ModelCfg.n_layers,
                 d_ff: int = C.ModelCfg.d_ff, dropout: float = C.ModelCfg.dropout):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d_model), nn.LayerNorm(d_model))
        self.missing_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.missing_token, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, n_pos, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        # 证据查询：对序列做注意力池化，权重即证据分布
        self.evidence = nn.Linear(d_model, 1, bias=False)
        self.absent = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.absent, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x:(B,T,D) mask:(B,T) 1=可用
        返回 h:(B,T,d) 序列表示, z:(B,d) 模态表示, beta:(B,T) 证据权重
        """
        m = mask.unsqueeze(-1)
        h = self.proj(x) * m + self.missing_token * (1.0 - m)
        h = self.drop(h + self.pos)
        padding = (mask <= 0).clone()
        empty = padding.all(dim=1)
        padding[empty, 0] = False
        h = self.enc(h, src_key_padding_mask=padding)
        h = self.norm(h)

        logits = self.evidence(h).squeeze(-1)                       # (B,T)
        logits = logits.masked_fill(mask <= 0, -1e4)
        beta = torch.softmax(logits, dim=-1) * mask
        beta = beta / beta.sum(-1, keepdim=True).clamp(min=1e-8)
        z = torch.einsum("bt,btd->bd", beta, h)
        absent = (mask.sum(dim=1, keepdim=True) <= 0)
        z = torch.where(absent, self.absent.unsqueeze(0).expand_as(z), z)
        return h, z, beta


class MaskAwareFusion(nn.Module):
    """掩码感知门控融合：输出模态作用程度 α 与融合表示 / 融合序列。"""

    def __init__(self, d_model: int = C.ModelCfg.d_model, n_mod: int = 3,
                 dropout: float = C.ModelCfg.dropout):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, zs: torch.Tensor, hs: torch.Tensor, avail: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        zs:(B,M,d) hs:(B,M,T,d) avail:(B,M) 各模态有效位置占比
        返回 z:(B,d) F:(B,T,d) alpha:(B,M)
        """
        s = self.score(zs).squeeze(-1)                              # (B,M)
        prior = torch.log(avail.clamp(min=1e-3))
        logits = (s + prior).masked_fill(avail <= 0, -1e4)
        # 全部模态都不可用时退化为均匀权重（避免 NaN）
        degenerate = (avail.sum(dim=1, keepdim=True) <= 0)
        logits = torch.where(degenerate.expand_as(logits), torch.zeros_like(logits), logits)
        alpha = torch.softmax(logits, dim=-1)
        z = torch.einsum("bm,bmd->bd", alpha, zs)
        F_seq = torch.einsum("bm,bmtd->btd", alpha, hs)
        return self.norm(z), F_seq, alpha


# --------------------------------------------------------------------------
# 主干
# --------------------------------------------------------------------------
class MultimodalBackbone(nn.Module):
    def __init__(self, cfg=C.ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict({
            m: ModalityEncoder(C.MOD_DIM[m], cfg.d_model, C.N_POS,
                               cfg.n_heads, cfg.n_layers, cfg.d_ff, cfg.dropout)
            for m in C.MODALITIES
        })
        self.fusion = MaskAwareFusion(cfg.d_model, len(C.MODALITIES), cfg.dropout)

    def forward(self, X: Dict[str, torch.Tensor], M: Dict[str, torch.Tensor], support=None):
        hs, zs, betas, avails = [], [], [], []
        for m in C.MODALITIES:
            h, z, b = self.encoders[m](X[m], M[m])
            hs.append(h); zs.append(z); betas.append(b)
            denominator = support[m].sum(dim=1).clamp(min=1) if support is not None else torch.full_like(M[m].sum(1), C.N_POS)
            avails.append(M[m].sum(dim=1) / denominator)
        H = torch.stack(hs, dim=1)                                  # (B,M,T,d)
        Z = torch.stack(zs, dim=1)                                  # (B,M,d)
        B = torch.stack(betas, dim=1)                               # (B,M,T)
        A = torch.stack(avails, dim=1)                              # (B,M)
        z, F_seq, alpha = self.fusion(Z, H, A)
        return dict(z=z, F_seq=F_seq, alpha=alpha, beta=B, H=H, Z=Z, avail=A)


# --------------------------------------------------------------------------
# 问题2：鲁棒预测模型
# --------------------------------------------------------------------------
class RobustModel(nn.Module):
    def __init__(self, cfg=C.ModelCfg):
        super().__init__()
        self.backbone = MultimodalBackbone(cfg)
        d = cfg.d_model
        self.head_cls = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                      nn.Dropout(cfg.dropout), nn.Linear(d, cfg.n_classes))
        self.head_reg = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                      nn.Dropout(cfg.dropout), nn.Linear(d, 1))
        # 重建解码器：由融合序列反推各模态原始特征
        self.decoders = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, C.MOD_DIM[m]))
            for m in C.MODALITIES
        })

    def forward(self, X, M, need_recon: bool = True, support=None):
        out = self.backbone(X, M, support=support)
        out["logits"] = self.head_cls(out["z"])
        out["pred_reg"] = 3.0 * torch.tanh(self.head_reg(out["z"]).squeeze(-1) / 3.0)
        if need_recon:
            out["recon"] = {m: self.decoders[m](out["F_seq"]) for m in C.MODALITIES}
        return out


# --------------------------------------------------------------------------
# 问题3：可解释预测模型
# --------------------------------------------------------------------------
class InterpretableModel(RobustModel):
    """Shared prediction structure. Weights alone do not establish attribution."""

    def forward(self, X, M, need_recon: bool = False, support=None):
        return super().forward(X, M, need_recon=need_recon, support=support)


# --------------------------------------------------------------------------
# 损失
# --------------------------------------------------------------------------
def compute_loss(out: dict, y_cls: torch.Tensor, y_reg: torch.Tensor,
                 X_true: Dict[str, torch.Tensor], M_natural: Dict[str, torch.Tensor],
                 M_obs: Dict[str, torch.Tensor], cfg=C.TrainCfg,
                 class_weight: Optional[torch.Tensor] = None):
    """总损失 = 分类 + 强度回归 + 重建（仅在被人工遮蔽但原本有效的位置上）
              + 可选正则（本次实验权重为零）"""
    loss_cls = F.cross_entropy(out["logits"], y_cls, weight=class_weight,
                               label_smoothing=cfg.label_smoothing)
    loss_reg = F.smooth_l1_loss(out["pred_reg"], y_reg, beta=0.5)

    loss_rec = torch.zeros((), device=y_cls.device)
    if "recon" in out:
        num = torch.zeros((), device=y_cls.device)
        for m in C.MODALITIES:
            # 只学"原本有效、被人工遮蔽"的位置（自然补零位不参与）
            target_mask = (M_natural[m] > 0) & (M_obs[m] <= 0)
            if target_mask.sum() == 0:
                continue
            pred = out["recon"][m]
            tgt = X_true[m].detach()
            diff = F.smooth_l1_loss(pred[target_mask], tgt[target_mask],
                                    beta=0.5, reduction="sum")
            loss_rec = loss_rec + diff / (target_mask.sum().clamp(min=1) * C.MOD_DIM[m])
            num = num + 1
        if num > 0:
            loss_rec = loss_rec / num

    # 跨模态互补正则：惩罚不同模态证据分布的平均重叠，鼓励各模态关注互补时段
    loss_comp = torch.zeros((), device=y_cls.device)
    if getattr(cfg, "lambda_comp", 0.0) > 0 and "beta" in out:
        B = out["beta"]                                    # (B, M, T)
        bbar = B.mean(dim=0)                               # (M, T) 批内平均证据分布
        bbar = bbar / bbar.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        pairs = [(0, 1), (0, 2), (1, 2)]
        loss_comp = sum((bbar[i] * bbar[j]).sum() for i, j in pairs) / len(pairs)

    total = (cfg.lambda_cls * loss_cls + cfg.lambda_reg * loss_reg
             + cfg.lambda_rec * loss_rec + cfg.lambda_comp * loss_comp)
    return total, {"cls": float(loss_cls.detach()), "reg": float(loss_reg.detach()),
                   "rec": float(loss_rec.detach()), "comp": float(loss_comp.detach())}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
