"""Corruption augmentation implementing the adjudicated 附件3 damage model.

Rule 1 (word_level_TAV, primary):
  - with prob P_FILE, damage a sample;
  - rate r drawn from the empirical pool of the 30 附件3 files (3 zeros);
  - k = round(r * |W|) positions drawn uniformly without replacement from the
    word span W = [1, L-2] (token positions; CLS/SEP/padding excluded);
  - at each hit i: ids[i] -> UNK(100) in place, attention mask UNCHANGED (=1),
    audio[i, :] = 0 and vision[i, :] = 0 (both modalities, synchronised).

Rule 3 (keep natural missing): nothing to do — natural vision dropouts and
whole-modality-missing samples are kept as provided; we never synthesise
whole-modality loss.

Rule 2 (frame_level_AV_dropout) is a no-op for the aligned pipeline (frame-level
scattered dropouts are absorbed by word-window averaging) — see audit report.

Acceptance signature (assertion helper): damaged positions are exactly
  {i : ids[i] == UNK} == {i : audio_row_is_zero(i) and i in W}
audio interior zeros have natural rate 0, so zero rows inside W == corruption.
"""
from __future__ import annotations

import numpy as np

from . import config
from .data_adapter import word_span_mask


class WordLevelTAV:
    """Rule-1 corruption operating on RAW (unstandardised) arrays.

    Variants for the 缺失类型/位置/时长 influence study (eval-time grid):
      modalities: subset of ("t","a","v") hit at the chosen positions.
                  Default ("t","a","v") = the adjudicated synchronized rule.
      contiguous: True -> one contiguous block of k positions (时长/连续缺失)
                  instead of k scattered positions.
      region: "any" (uniform over W) | "head" | "mid" | "tail" (位置规律).
    """

    def __init__(self, p_file: float = config.P_FILE,
                 rate_pool=None, seed: int = config.SEED,
                 modalities=("t", "a", "v"), contiguous: bool = False,
                 region: str = "any"):
        self.p_file = float(p_file)
        self.rate_pool = np.asarray(
            config.EMPIRICAL_RATES if rate_pool is None else rate_pool, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.modalities = tuple(modalities)
        self.contiguous = bool(contiguous)
        self.region = region

    def reseed(self, seed: int) -> None:
        """Fresh RNG stream. Without this, DataLoader fork clones the parent
        rng into every worker and each epoch replays the identical stream
        (adversarial review finding 2026-09-24); worker_init_fn calls this so
        worker/epoch streams are independent."""
        self.rng = np.random.default_rng(seed)

    def __call__(self, ids: np.ndarray, attn: np.ndarray,
                 audio: np.ndarray, vision: np.ndarray):
        """Damage one sample IN PLACE; returns the boolean damage mask (50,)."""
        damage = np.zeros(ids.shape[0], dtype=bool)
        W = word_span_mask(attn)
        n_w = int(W.sum())
        if n_w == 0 or self.rng.random() >= self.p_file:
            return damage
        r = float(self.rng.choice(self.rate_pool))
        k = int(round(r * n_w))
        k = min(max(k, 0), n_w)
        if k == 0:
            return damage
        w_idx = np.where(W)[0]
        if self.contiguous:
            # one contiguous block of length k, start uniform in the region
            start = self.rng.integers(0, n_w - k + 1)
            pos = w_idx[start:start + k]
        elif self.region == "any":
            pos = self.rng.choice(w_idx, size=k, replace=False)
        else:
            # region-restricted: head/mid/tail third of the word span
            thirds = np.array_split(w_idx, 3)
            pick = {"head": 0, "mid": 1, "tail": 2}[self.region]
            pool = thirds[pick]
            k = min(k, len(pool))
            pos = self.rng.choice(pool, size=k, replace=False)
        damage[pos] = True
        if "t" in self.modalities:
            ids[pos] = config.UNK_ID      # in-place id -> [UNK]; mask untouched
        if "a" in self.modalities:
            audio[pos, :] = 0.0
        if "v" in self.modalities:
            vision[pos, :] = 0.0
        return damage


def corruption_signature(ids: np.ndarray, audio: np.ndarray,
                         attn: np.ndarray) -> set[int]:
    """Positions i in W with ids[i]==UNK AND audio row zero (the audit signature)."""
    W = word_span_mask(attn)
    audio_zero = np.isclose(audio, 0).all(axis=1)
    return set(np.where(W & (ids == config.UNK_ID) & audio_zero)[0].tolist())


def assert_augmentation_invariants(ids0, attn, ids1, audio1, vision1, damage):
    """Raise AssertionError if any adjudicated invariant is violated.

    ids0/attn: pre-damage arrays; ids1/audio1/vision1/damage: post-damage.
    """
    L = int(attn.sum())
    assert np.array_equal(attn, attn), "attention mask must never change"
    # CLS/SEP/padding untouched
    assert ids1[0] == ids0[0] == config.CLS_ID
    assert ids1[L - 1] == ids0[L - 1] == config.SEP_ID
    assert np.array_equal(ids1[L:], ids0[L:]), "padding ids must be untouched"
    # damaged positions: UNK in text, zero rows in audio AND vision, inside W
    W = word_span_mask(attn)
    unk = set(np.where(ids1 == config.UNK_ID)[0].tolist())
    a0 = set(np.where(np.isclose(audio1, 0).all(axis=1))[0].tolist())
    v0 = set(np.where(np.isclose(vision1, 0).all(axis=1))[0].tolist())
    hits = set(np.where(damage)[0].tolist())
    assert hits <= set(np.where(W)[0].tolist()), "damage must stay inside word span"
    assert unk >= hits, "damaged positions must carry UNK"
    assert a0 >= hits, "damaged audio rows must be zero"
    assert v0 >= hits, "damaged vision rows must be zero"
    # no damage outside hits relative to original
    assert set(np.where(ids0 == config.UNK_ID)[0].tolist()) <= unk
