"""Self-contained revised checkpoint loading. No original Linux path or train data needed."""
from pathlib import Path
import numpy as np
import torch
from . import config
from .data_adapter import Standardizer
from .models import Q2Model


def load_revised(path, device="cpu"):
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    if ck.get("protocol_version") != config.PROTOCOL_VERSION:
        raise ValueError("Historical checkpoint: retrain under the revised protocol; do not relabel legacy results")
    required = {"state_dict", "bert_config", "normalizer", "args"}
    if not required <= ck.keys():
        raise ValueError(f"Incomplete checkpoint: missing {required - ck.keys()}")
    arrays = {k: np.asarray(ck["normalizer"][k], dtype=np.float32) for k in ("a_mean", "a_std", "v_mean", "v_std")}
    for k, n in (("a_mean",74),("a_std",74),("v_mean",35),("v_std",35)):
        if arrays[k].shape != (n,) or not np.isfinite(arrays[k]).all():
            raise ValueError(f"Invalid normalizer: {k}")
        if k.endswith("std") and (arrays[k] <= 0).any():
            raise ValueError(f"Non-positive standard deviation: {k}")
    model = Q2Model(bert_config=ck["bert_config"], freeze_text=bool(ck["args"].get("freeze_text",1)))
    model.load_state_dict(ck["state_dict"], strict=True)
    return model.to(device).eval(), Standardizer(**arrays), ck


def load_ensemble(paths, device="cpu"):
    if not paths:
        raise ValueError("No checkpoints provided")
    models, reference = [], None
    for path in paths:
        model, std, _ = load_revised(path, device)
        if reference is not None:
            for k in ("a_mean", "a_std", "v_mean", "v_std"):
                if not np.array_equal(getattr(reference,k),getattr(std,k)):
                    raise ValueError("Ensemble checkpoints have different training normalizers")
        reference = std
        models.append(model)
    return models, reference
