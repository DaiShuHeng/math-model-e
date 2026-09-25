"""Train/validation synthetic corruption in raw token/feature space.
Default: a continuous block and a predefined rate grid, independent of special
samples. Region means a head-, centre-, or tail-anchored interval. Scattered
corruption is an explicitly requested secondary control, not the main task.
"""
from __future__ import annotations

import numpy as np

from . import config
from .data_adapter import word_span_mask


class WordLevelTAV:
    """Rule-1 corruption operating on RAW (unstandardised) arrays.

    Variants for the 缺失类型/位置/时长 influence study (eval-time grid):
      modalities: subset of ("t","a","v") hit at the chosen positions.
                  Default ("t","a","v") = a synchronized-block experimental condition.
      contiguous: True -> one contiguous block of k positions (时长/连续缺失)
                  instead of k scattered positions.
      region: "any" (uniform over W) | "head" | "mid" | "tail" (位置规律).
    """

    def __init__(self, p_file: float = config.P_FILE,
                 rate_pool=None, seed: int = config.SEED,
                 modalities=("t", "a", "v"), contiguous: bool = True,
                 region: str = "any"):
        self.p_file = float(p_file)
        if not 0 <= self.p_file <= 1: raise ValueError("p_file outside [0,1]")
        self.rate_pool = np.asarray(
            config.PREDEFINED_RATES if rate_pool is None else rate_pool, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.modalities = tuple(modalities)
        self.contiguous = bool(contiguous)
        self.region = region
        if region not in ("any", "head", "mid", "tail"): raise ValueError(region)
        if not self.modalities or not set(self.modalities) <= {"t", "a", "v"}: raise ValueError("Invalid modalities")
        if self.rate_pool.size == 0 or not np.isfinite(self.rate_pool).all() or ((self.rate_pool < 0) | (self.rate_pool > 1)).any(): raise ValueError("Invalid rate pool")

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
            # Preserve exactly k positions in every location condition.
            starts = {"head": 0, "mid": (n_w-k)//2, "tail": n_w-k}
            start = int(self.rng.integers(0, n_w-k+1)) if self.region == "any" else starts[self.region]
            pos = w_idx[start:start+k]
        elif self.region == "any":
            pos = self.rng.choice(w_idx, size=k, replace=False)
        else:
            thirds = np.array_split(w_idx, 3)
            pool = thirds[{"head": 0, "mid": 1, "tail": 2}[self.region]]
            if k > len(pool):
                raise ValueError("Requested scattered rate exceeds regional capacity; do not silently reduce rate")
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


def assert_augmentation_invariants(ids0, attn_before, ids1, audio1, vision1, damage, attn_after):
    """Raise AssertionError if any synchronized-TAV invariant is violated.

    ids0/attn: pre-damage arrays; ids1/audio1/vision1/damage: post-damage.
    """
    attn = attn_before
    L = int(attn_before.sum())
    assert np.array_equal(attn_before, attn_after), "attention mask changed"
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

class MixedBlockAugment(WordLevelTAV):
    """Uniformly choose one of seven nonempty modality subsets, then one block."""
    SUBSETS = (("t",), ("a",), ("v",), ("t","a"), ("t","v"), ("a","v"), ("t","a","v"))
    def __call__(self, ids, attn, audio, vision):
        self.modalities = self.SUBSETS[int(self.rng.integers(len(self.SUBSETS)))]
        return super().__call__(ids, attn, audio, vision)
