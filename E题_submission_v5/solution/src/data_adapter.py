"""Data adapter: load 附件2 aligned pkl into arrays + visibility masks.

Design decisions (verified by the vocab/interface audit):
- Observation masks are computed on RAW features: a row is "observed" iff it is
  not all-zero. Zeros are an operational visibility convention; a zero row alone does not
  establish the physical cause of missingness.
- Standardization statistics are computed ONCE on the TRAIN split over observed
  rows only, and applied with zero-preservation (missing rows stay exactly 0).
- The adapter never touches labels beyond passing them through; splits keep the
  original order.
"""
from __future__ import annotations

import pickle
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


class _SafeUnpickler(pickle.Unpickler):
    """numpy-whitelist unpickler (same policy as the audit scripts)."""

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
        ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy", "asarray"),
        ("builtins", "slice"), ("collections", "OrderedDict"),
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"Unapproved data class: {module}.{name}")
        return super().find_class(module, name)


def read_pickle(path: Path):
    with open(path, "rb") as f:
        try:
            return _SafeUnpickler(f).load()
        except (EOFError, pickle.UnpicklingError) as exc:
            raise ValueError(f"Cannot read {path}: incomplete/corrupt or unsupported pickle ({exc})") from exc


@dataclass
class SplitData:
    split: str
    ids: np.ndarray          # (N, 50) int64  token ids
    attn: np.ndarray         # (N, 50) int64  attention mask (1=valid)
    seg: np.ndarray          # (N, 50) int64  token type ids (all zero)
    audio: np.ndarray        # (N, 50, 74) float32 raw
    vision: np.ndarray       # (N, 50, 35) float32 raw
    audio_obs: np.ndarray    # (N, 50) bool  observed (non-zero) rows
    vision_obs: np.ndarray   # (N, 50) bool
    y_reg: np.ndarray        # (N,) float32  [-3, 3]
    y_cls: np.ndarray        # (N,) int64    {0,1,2}
    sample_id: list          # N strings video$_$clip

    def __len__(self):
        return len(self.y_reg)


_CACHE_FILE = config.CACHE / "aligned_arrays.npz"


def _build_from_pkl(splits=("train", "valid", "test")) -> dict:
    data = read_pickle(config.ALIGNED_PKL)
    out = {}
    for split in splits:
        d = data[split]
        tb = np.asarray(d["text_bert"], dtype=np.int64)
        audio = np.asarray(d["audio"], dtype=np.float32)
        vision = np.asarray(d["vision"], dtype=np.float32)
        out[split] = dict(
            ids=tb[:, 0, :], attn=tb[:, 1, :], seg=tb[:, 2, :],
            audio=audio, vision=vision,
            audio_obs=~np.isclose(audio, 0).all(axis=2),
            vision_obs=~np.isclose(vision, 0).all(axis=2),
            y_reg=np.asarray(d["regression_labels"], dtype=np.float32).reshape(-1),
            y_cls=np.asarray(d["classification_labels"], dtype=np.int64).reshape(-1),
            sample_id=np.asarray(d["id"]).astype(str),
        )
    return out


def load_splits(use_cache: bool = True) -> dict[str, SplitData]:
    """Return {'train': SplitData, 'valid': ..., 'test': ...} for aligned_50."""
    raw = None
    source = config.ALIGNED_PKL
    stamp_path = _CACHE_FILE.with_suffix(".source.json")
    st = source.stat()
    stamp = {"path": str(source.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    cached_stamp = json.loads(stamp_path.read_text()) if stamp_path.exists() else None
    if use_cache and _CACHE_FILE.exists() and cached_stamp == stamp:
        z = np.load(_CACHE_FILE, allow_pickle=False)
        raw = {}
        for s in ["train", "valid", "test"]:
            prefix = f"{s}__"
            raw[s] = {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}
    if raw is None:
        raw = _build_from_pkl()
        config.CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            _CACHE_FILE,
            **{f"{s}__{k}": v for s, d in raw.items() for k, v in d.items()},
        )
        stamp_path.write_text(json.dumps(stamp))
    return {s: SplitData(split=s, **d) for s, d in raw.items()}


@dataclass
class Standardizer:
    """Per-dimension z-score fitted on TRAIN observed rows; preserves zeros."""

    a_mean: np.ndarray   # (74,)
    a_std: np.ndarray    # (74,)
    v_mean: np.ndarray   # (35,)
    v_std: np.ndarray    # (35,)

    @staticmethod
    def fit(train: SplitData, eps: float = 1e-6) -> "Standardizer":
        def stats(x, obs):
            rows = x[obs]                       # (K, D) observed rows only
            mean = rows.mean(axis=0)
            std = rows.std(axis=0)
            std = np.maximum(std, eps)
            return mean.astype(np.float32), std.astype(np.float32)

        a_mean, a_std = stats(train.audio, train.audio_obs)
        v_mean, v_std = stats(train.vision, train.vision_obs)
        return Standardizer(a_mean, a_std, v_mean, v_std)

    def transform_audio(self, x: np.ndarray, obs: np.ndarray) -> np.ndarray:
        z = (x - self.a_mean) / self.a_std
        return (z * obs[..., None]).astype(np.float32)

    def transform_vision(self, x: np.ndarray, obs: np.ndarray) -> np.ndarray:
        z = (x - self.v_mean) / self.v_std
        return (z * obs[..., None]).astype(np.float32)


def word_span_mask(attn_row: np.ndarray) -> np.ndarray:
    """W = [1, L-2] inclusive: valid positions excluding CLS(0) and SEP(L-1)."""
    L = int(attn_row.sum())
    m = np.zeros_like(attn_row, dtype=bool)
    if L >= 3:
        m[1: L - 1] = True   # positions 1 .. L-2
    return m


def load_training_splits() -> dict[str, SplitData]:
    """Read TRAIN/VALID only into model inputs; no persistent cache mutation.

    The supplied pickle is one container, so deserialization necessarily reads
    the container, but TEST arrays are never adapted, evaluated or optimized.
    """
    return {s: SplitData(split=s, **d) for s, d in
            _build_from_pkl(splits=("train", "valid")).items()}
