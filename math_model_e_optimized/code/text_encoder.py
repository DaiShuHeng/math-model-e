"""Frozen BERT encoder. Official aligned features use direct last-layer token states.
Problem 1 uses physical CTC time bins in p1_refine.py; these interfaces differ."""
from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

import numpy as np

import config as C

_TOK = None
_MODEL = None


def load_tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(C.BERT_MODEL)
    return _TOK


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_bert():
    """懒加载 tokenizer 与 BERT（经 HF_ENDPOINT 镜像下载）。"""
    global _TOK, _MODEL
    if _MODEL is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        _TOK = load_tokenizer()
        _MODEL = AutoModel.from_pretrained(C.BERT_MODEL).eval().to(_device())
        for p in _MODEL.parameters():
            p.requires_grad_(False)
        _ = torch  # noqa
    return _TOK, _MODEL


def encode_text_bert_batch(text_bert: Sequence[np.ndarray]) -> List[np.ndarray]:
    """附件3 接口：text_bert (50,3) 整数 -> (50,768) 特征。"""
    ii = np.stack([np.asarray(t, dtype=np.int64)[:, 0] for t in text_bert])
    am = np.stack([np.asarray(t, dtype=np.int64)[:, 1] for t in text_bert])
    tt = np.stack([np.asarray(t, dtype=np.int64)[:, 2] for t in text_bert])
    import torch
    _, model = load_bert()
    with torch.no_grad():
        h = model(input_ids=torch.tensor(ii, device=_device()),
                  attention_mask=torch.tensor(am, device=_device()),
                  token_type_ids=torch.tensor(tt, device=_device())).last_hidden_state
    return list(h.cpu().numpy().astype(np.float32))


