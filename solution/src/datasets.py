"""Torch datasets for Q2: augmentation on the fly, standardisation preserving zeros.

Pipeline per item (training):
  raw arrays -> WordLevelTAV damage (raw space) -> recompute observation masks
  -> standardise (train stats, zeros preserved) -> tensors.
For evaluation, damage is either off (clean) or applied with a FIXED seed per
sample id so every model sees the identical corrupted validation set.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import WordLevelTAV
from .data_adapter import SplitData, Standardizer


class Q2Dataset(Dataset):
    def __init__(self, sd: SplitData, std: Standardizer, *,
                 train: bool = True, augmenter: WordLevelTAV | None = None,
                 fixed_corrupt_seed: int | None = None):
        self.sd = sd
        self.std = std
        self.train = train
        self.augmenter = augmenter
        self.fixed_seed = fixed_corrupt_seed

    def __len__(self):
        return len(self.sd)

    def __getitem__(self, j: int):
        sd = self.sd
        ids = sd.ids[j].copy()
        audio = sd.audio[j].copy()
        vision = sd.vision[j].copy()
        attn = sd.attn[j]

        if self.augmenter is not None:
            if self.fixed_seed is not None:
                # deterministic per-sample corruption for comparable evaluation:
                # a dedicated augmenter seeded per sample index (variant options
                # forwarded so type/position/duration grids share the protocol)
                local = WordLevelTAV(p_file=self.augmenter.p_file,
                                     rate_pool=self.augmenter.rate_pool,
                                     seed=self.fixed_seed + j,
                                     modalities=self.augmenter.modalities,
                                     contiguous=self.augmenter.contiguous,
                                     region=self.augmenter.region)
                local(ids, attn, audio, vision)
            else:
                self.augmenter(ids, attn, audio, vision)

        audio_obs = ~np.isclose(audio, 0).all(axis=1)
        vision_obs = ~np.isclose(vision, 0).all(axis=1)
        audio_z = self.std.transform_audio(audio[None], audio_obs[None])[0]
        vision_z = self.std.transform_vision(vision[None], vision_obs[None])[0]

        return {
            "ids": torch.from_numpy(ids),
            "attn": torch.from_numpy(attn.astype(np.int64)),
            "audio": torch.from_numpy(audio_z),
            "vision": torch.from_numpy(vision_z),
            "audio_obs": torch.from_numpy(audio_obs.astype(np.float32)),
            "vision_obs": torch.from_numpy(vision_obs.astype(np.float32)),
            "y_reg": torch.tensor(float(self.sd.y_reg[j]), dtype=torch.float32),
            "y_cls": torch.tensor(int(self.sd.y_cls[j]), dtype=torch.long),
        }
