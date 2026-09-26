"""Aligned-feature data access for the v5 feature-level Q2/Q3 model.

Why a second adapter: ``data_adapter.py`` feeds the token-level Tiny model
(``ids``/``attn`` -> BERT).  The v5 model consumes the competition's *standardised
modality features* (文本 768 维、语音 74 维、视觉 35 维) exactly as distributed in
`aligned_50.pkl`、附件3、附件4.  Both adapters read the same provided files; no
external data is introduced.

Observation convention (verified against the provided files, see
`docs/缺失结构审计.md`):
  * a row is *observed* iff it is not all-zero;
  * for the aligned files 100% of zero audio/vision rows lie outside the text
    token span, and ~96% of samples have audio/vision support exactly equal to
    the text span.  Missingness is therefore *span-bounded*, not scattered.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config

MODALITIES = ("T", "A", "V")
DIM = {"T": 768, "A": 74, "V": 35}


class _SafeUnpickler(pickle.Unpickler):
    """numpy/dict whitelist; same policy as data_adapter.py."""

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
class FeatSplit:
    """Feature-level view of one split. All arrays are float32, raw (unstandardised)."""
    split: str
    text: np.ndarray        # (N, 50, 768)  competition text features
    audio: np.ndarray       # (N, 50, 74)
    vision: np.ndarray      # (N, 50, 35)
    t_obs: np.ndarray       # (N, 50) float32  1 = word-token span (positions 1..L-2)
    a_obs: np.ndarray       # (N, 50) float32
    v_obs: np.ndarray       # (N, 50) float32
    y_reg: np.ndarray | None       # (N,) or None for unlabeled sets
    y_cls: np.ndarray | None
    sample_id: list
    span_obs: np.ndarray | None = None   # (N, 50) full contiguous valid-token prefix

    def __len__(self):
        return len(self.sample_id)


def _obs(x: np.ndarray) -> np.ndarray:
    return (~np.isclose(x, 0).all(axis=2)).astype(np.float32)


def _row(raw: dict, split: str, with_labels: bool, span_only: bool = True) -> FeatSplit:
    text = np.asarray(raw["text"], dtype=np.float32)
    audio = np.asarray(raw["audio"], dtype=np.float32)
    vision = np.asarray(raw["vision"], dtype=np.float32)
    if with_labels:
        y_reg = np.asarray(raw["regression_labels"], dtype=np.float32).reshape(-1)
        # classification_labels is provided as a float 3-class index
        y_cls = np.asarray(raw["classification_labels"]).reshape(-1).astype(np.int64)
    else:
        y_reg = y_cls = None
    raw_attn = None
    if "text_bert" in raw:
        tb = np.asarray(raw["text_bert"], dtype=np.int64)
        raw_attn = tb[:, 1, :].astype(np.float32) if tb.ndim == 3 else tb[1].astype(np.float32)
    span_obs = raw_attn if raw_attn is not None else _obs(text)
    return FeatSplit(split=split, text=text, audio=audio, vision=vision,
                     t_obs=(token_span_mask(span_obs) if span_only else span_obs),
                     a_obs=_obs(audio), v_obs=_obs(vision),
                     y_reg=y_reg, y_cls=y_cls,
                     sample_id=[str(x) for x in raw.get("id", [])],
                     span_obs=span_obs)


def load_aligned_splits(splits=("train", "valid"), with_labels: bool = True,
                        span_only: bool = True) -> dict[str, FeatSplit]:
    """附件2 aligned_50.pkl. TEST is never loaded here unless explicitly named.

    ``span_only`` restricts the text observation mask to the word-token span
    ``1..L-2`` (the positions that carry aligned ``text`` rows for this sample);
    this is the convention the model pools and standardises over.
    """
    data = read_pickle(config.ALIGNED_PKL)
    return {s: _row(data[s], s, with_labels, span_only) for s in splits}


def load_attachment3() -> FeatSplit:
    """附件3 = 30 single-sample pickles, each {'test': {...}} with no labels."""
    files = sorted(config.ATT3_DIR.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No 附件3 pickles under {config.ATT3_DIR}")
    return _stack_pickles(files, "att3")


def load_attachment4() -> FeatSplit:
    """附件4 = 20 single-sample pickles, flat dicts with 'id', no labels."""
    files = sorted(config.ATT4_DIR.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No 附件4 pickles under {config.ATT4_DIR}")
    return _stack_pickles(files, "att4")


def _stack_pickles(files: list[Path], split: str) -> FeatSplit:
    texts, audios, visions, ids, attns = [], [], [], [], []
    for p in files:
        d = read_pickle(p)
        # 附件3 wraps its record under the 'test' key; 附件4 is already flat.
        if "test" in d and isinstance(d["test"], dict):
            d = d["test"]
        texts.append(np.asarray(d["text"], dtype=np.float32))
        audios.append(np.asarray(d["audio"], dtype=np.float32))
        visions.append(np.asarray(d["vision"], dtype=np.float32))
        tb = np.asarray(d["text_bert"], dtype=np.float32)
        attns.append((tb[1] if tb.ndim == 2 else tb[0, 1]).astype(np.float32))
        sid = d.get("id", p.stem)
        sid = sid[0] if isinstance(sid, (list, tuple, np.ndarray)) else sid
        ids.append(str(sid))
    text = np.stack(texts); audio = np.stack(audios); vision = np.stack(visions)
    span_obs = np.stack(attns)
    return FeatSplit(split=split, text=text, audio=audio, vision=vision,
                     t_obs=token_span_mask(span_obs), a_obs=_obs(audio), v_obs=_obs(vision),
                     y_reg=None, y_cls=None, sample_id=ids, span_obs=span_obs)


@dataclass
class FeatStandardizer:
    """Per-dimension z-score fitted on TRAIN observed rows only, zeros preserved.

    Text is standardised over the **valid-token prefix** (``span_obs``, i.e. the
    attention mask), *not* over all 50 rows.  In the provided files the 768-d
    text row norms decay across positions while staying non-zero everywhere, so
    including the padded tail drags the mean down and the scale up by a
    length-dependent amount — two samples with different ``L`` would end up on
    different scales.  Audio/vision are standardised over their observed rows,
    which already lie inside the prefix (audit §2.1).  Note the distinction from
    ``t_obs``: the *pooling* mask is the word span ``1..L-2`` (CLS/SEP carry no
    word-level evidence), while the *standardisation* mask is the full prefix.
    """
    t_mean: np.ndarray; t_std: np.ndarray
    a_mean: np.ndarray; a_std: np.ndarray
    v_mean: np.ndarray; v_std: np.ndarray

    @staticmethod
    def fit(train: FeatSplit, eps: float = 1e-6) -> "FeatStandardizer":
        def st(x, obs):
            rows = x[obs.astype(bool)]
            if rows.size == 0:
                raise ValueError("No observed rows to fit statistics on")
            return rows.mean(0).astype(np.float32), np.maximum(rows.std(0), eps).astype(np.float32)
        # Text statistics are fitted on the rows inside each sample's valid-token
        # prefix — the same mask that is applied at transform time, so fit and
        # apply agree by construction.  The tail rows (positions >= L) belong to
        # no aligned word for that sample; including them would mix a
        # length-dependent, norm-decaying population into the scale and would put
        # fit and apply on different supports.  AV statistics use observed rows
        # only, since a zero AV row is a genuine gap rather than padding.
        t_mask = train.span_obs if train.span_obs is not None else _obs(train.text)
        tm, ts = st(train.text, t_mask)
        am, as_ = st(train.audio, train.a_obs)
        vm, vs = st(train.vision, train.v_obs)
        return FeatStandardizer(tm, ts, am, as_, vm, vs)

    def to_json(self) -> dict:
        return {k: getattr(self, k).tolist() for k in
                ("t_mean", "t_std", "a_mean", "a_std", "v_mean", "v_std")}

    @staticmethod
    def from_json(d: dict) -> "FeatStandardizer":
        return FeatStandardizer(**{k: np.asarray(v, dtype=np.float32) for k, v in d.items()})

    def transform(self, sp: FeatSplit) -> dict[str, np.ndarray]:
        """Returns standardised arrays with zeros preserved at unobserved rows.

        Text masking uses ``span_obs`` (the valid-token prefix): the padded tail
        carries no aligned content for that sample and must not enter pooling.
        ``word_span`` is a stricter mask (word positions only, CLS/SEP excluded)
        and is kept separately for evidence localisation.
        """
        t_mask = sp.span_obs if sp.span_obs is not None else _obs(sp.text)
        t = (sp.text - self.t_mean) / self.t_std
        a = (sp.audio - self.a_mean) / self.a_std
        v = (sp.vision - self.v_mean) / self.v_std
        return {
            "T": (t * t_mask[..., None]).astype(np.float32),
            "A": (a * sp.a_obs[..., None]).astype(np.float32),
            "V": (v * sp.v_obs[..., None]).astype(np.float32),
            "t_obs": t_mask.astype(np.float32), "a_obs": sp.a_obs, "v_obs": sp.v_obs,
            "word_span": sp.t_obs.astype(np.float32),
        }


def span_of(t_obs: np.ndarray) -> np.ndarray:
    """Contiguous text span (token positions) as a float mask, same shape as t_obs."""
    return t_obs


def token_span_mask(t_obs: np.ndarray, exclude_special: bool = True) -> np.ndarray:
    """Word-token span ``W`` per sample, from the (contiguous) text observation mask.

    The mask is a contiguous prefix; ``W`` is positions ``1..L-2``, excluding the
    CLS/SEP special positions.  Rows are never dropped — samples whose span is
    empty simply contribute no rows to the statistics.
    """
    m = np.asarray(t_obs) > 0
    if not exclude_special:
        return m
    out = np.zeros_like(m)
    for i in range(m.shape[0]):
        L = int(m[i].sum())
        if L >= 3:
            out[i, 1:L - 1] = True
        elif L > 0:
            out[i, :L] = True
    return out
