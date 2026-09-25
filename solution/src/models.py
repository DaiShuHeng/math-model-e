"""Q2 model: frozen BERT-Tiny text branch + visibility-aware A/V branches,
gated fusion, dual heads (3-class polarity + intensity regression).

All pooling and gating respect two mask families:
  - text-valid positions (attention mask, includes CLS/SEP)
  - per-modality observation (non-zero rows; missing/padding rows excluded)
A fully-unavailable modality is gated off; text is always at least CLS/SEP.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import config


class MaskedAttnPool(nn.Module):
    """Attention pooling over time with a float mask (1=keep)."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) / np.sqrt(dim))

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: (B, T, D), mask: (B, T) float
        scores = torch.einsum("btd,d->bt", h, self.query) / np.sqrt(h.size(-1))
        scores = scores.masked_fill(mask <= 0, -1e4)
        w = torch.softmax(scores, dim=-1) * (mask > 0).to(scores.dtype)
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
        return torch.einsum("bt,btd->bd", w, h)


def run_packed(gru, x, span):
    """Exclude right padding from both directions; retain interior missing steps."""
    lengths = span.long().sum(-1)
    expected = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
    if not torch.equal(span.bool(), expected):
        raise ValueError("Attention mask must be a contiguous prefix")
    packed = pack_padded_sequence(x, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
    output, _ = gru(packed)
    h, _ = pad_packed_sequence(output, batch_first=True, total_length=x.shape[1])
    return h * span[..., None]



class ModalityBranch(nn.Module):
    """LayerNorm -> projection -> BiGRU -> masked attention pooling."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.pool = MaskedAttnPool(hidden * 2)

    def forward(self, x: torch.Tensor, obs: torch.Tensor, span: torch.Tensor):
        # x: (B, T, D) standardised (zeros at missing); obs/span: (B, T) float
        h = self.proj(self.norm(x))
        h = run_packed(self.gru, h, span)
        mask = obs * span
        rep = self.pool(h, mask)
        avail = mask.sum(dim=-1, keepdim=True) / span.sum(dim=-1, keepdim=True).clamp_min(1)             # (B, 1)
        rep = rep * (avail > 0).float()                     # zero out if no obs
        return rep, avail


class Q2Model(nn.Module):
    def __init__(self, bert_dir=config.BERT_DIR, freeze_text: bool = True,
                 hidden: int = 64, mlp_hidden: int = 128, dropout: float = 0.3, bert_config=None):
        super().__init__()
        self.bert = (BertModel(BertConfig.from_dict(bert_config)) if bert_config is not None
                     else BertModel.from_pretrained(str(bert_dir), local_files_only=True))
        d_text = self.bert.config.hidden_size          # 128 (Tiny) or 768 (Base)
        self.freeze_text = freeze_text
        if freeze_text:
            for p in self.bert.parameters():
                p.requires_grad_(False)
            self.bert.eval()
        self.text_norm = nn.LayerNorm(d_text)
        self.text_gru = nn.GRU(d_text, hidden, batch_first=True, bidirectional=True)
        self.text_pool = MaskedAttnPool(hidden * 2)

        self.audio_branch = ModalityBranch(74, hidden)
        self.vision_branch = ModalityBranch(35, hidden)

        # every branch outputs hidden*2 (=128 with hidden=64)
        total = hidden * 2 * 3
        self.gate = nn.Sequential(
            nn.Linear(total + 3, total // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(total // 2, 3),
        )
        self.fuse = nn.Sequential(
            nn.Linear(total, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(),
        )
        self.head_cls = nn.Linear(mlp_hidden, config.N_CLASSES)
        self.head_reg = nn.Linear(mlp_hidden, 1)

    def encode_text(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        if self.freeze_text:
            self.bert.eval()
            with torch.no_grad():
                out = self.bert(input_ids=ids, attention_mask=attn)
        else:
            out = self.bert(input_ids=ids, attention_mask=attn)
        return out.last_hidden_state                      # (B, T, 128)

    def forward(self, batch: dict):
        ids = batch["ids"]
        attn = batch["attn"].float()
        span = attn                                        # text-valid positions
        t = self.encode_text(ids, attn.long())
        t = self.text_norm(t)
        th = run_packed(self.text_gru, t, span)
        rep_t = self.text_pool(th, span)
        avail_t = (span.sum(dim=-1, keepdim=True) > 0).float()  # text branch present, NOT word reliability

        rep_a, avail_a = self.audio_branch(batch["audio"], batch["audio_obs"], span)
        rep_v, avail_v = self.vision_branch(batch["vision"], batch["vision_obs"], span)

        reps = torch.cat([rep_t, rep_a, rep_v], dim=-1)    # (B, 3*2h)
        avails = torch.cat([avail_t, avail_a, avail_v], dim=-1)  # (B, 3)
        logits = self.gate(torch.cat([reps, avails], dim=-1))
        logits = logits.masked_fill(avails <= 0, -1e4)     # force off dead modalities
        gate = torch.softmax(logits, dim=-1)
        # gate-modulated concatenation: keeps per-modality information weighted
        # by its availability-aware gate weight -> (B, total)
        fused = (reps.view(reps.size(0), 3, -1) * gate[..., None]).reshape(reps.size(0), -1)
        z = self.fuse(fused)
        return {
            "logits": self.head_cls(z),                    # (B, 3)
            "reg": 3 * torch.tanh(self.head_reg(z)).squeeze(-1),  # (B,) in (-3, 3)
            "gate": gate, "avail": avails,
        }


def q2_loss(out: dict, batch: dict, lambda_cls: float = 1.0,
            huber_delta: float = 1.0, cls_weight=None):
    ce = F.cross_entropy(out["logits"], batch["y_cls"], weight=cls_weight)
    hub = F.smooth_l1_loss(out["reg"], batch["y_reg"], beta=huber_delta)
    return hub + lambda_cls * ce, {"ce": float(ce.item()), "huber": float(hub.item())}
