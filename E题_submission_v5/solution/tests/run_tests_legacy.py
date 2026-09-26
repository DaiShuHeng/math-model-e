"""Q2 pipeline tests. Plain-assert style; run:
  cd solution && python -m tests.run_tests
Covers:
  T1 a3 real-data signature: our visibility predicate reproduces the audit
     signature {UNK in W} == {audio zero rows in W} on all 30 files.
  T2 damage_sets replay: computed n_unk/rate match the forensic CSV.
  T3 augmentation invariants on real a2 samples (CLS/SEP/padding/mask/rows).
  T4 statistical check: damaged fraction over draws matches p_file*E[pool].
  T5 determinism of fixed-seed evaluation corruption.
  T6 standardizer: zeros preserved, stats from observed rows only.
  T7 word span edge cases (L=3).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

SOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOL))

from src import config                                    # noqa: E402
from src.augment import WordLevelTAV, corruption_signature  # noqa: E402
from src.data_adapter import Standardizer, load_splits, word_span_mask  # noqa: E402

A3_CACHE = (config.DATA_DIR / "analysis" / "vocab_cache" / "a3_aligned_tokens.json")
DAMAGE_CSV = (config.DATA_DIR / "analysis" / "vocab_cache" / "adv" / "zero_row.damage_sets.csv")


def test_a3_signature():
    a3 = json.loads(A3_CACHE.read_text())
    n_match = 0
    for name, d in a3.items():
        tb = np.asarray(d["text_bert"])
        ids, attn = tb[0], tb[1]
        # rebuild raw audio/vision zero-row masks from cache flags
        audio = np.zeros((50, 2), dtype=np.float32)
        audio[~np.asarray(d["audio_zero_rows"], dtype=bool)] = 1.0
        vision = np.zeros((50, 2), dtype=np.float32)
        vision[~np.asarray(d["vision_zero_rows"], dtype=bool)] = 1.0
        sig = corruption_signature(ids, audio, attn)
        W = word_span_mask(attn)
        unk_in_w = set(np.where(W & (ids == config.UNK_ID))[0].tolist())
        assert sig == unk_in_w, f"{name}: signature mismatch"
        n_match += 1
    print(f"T1 PASS: a3 signature check {n_match}/30 files")


def test_damage_sets_replay():
    a3 = json.loads(A3_CACHE.read_text())
    with open(DAMAGE_CSV) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30
    for r in rows:
        name = r["file"]
        d = a3[name]
        tb = np.asarray(d["text_bert"])
        ids, attn = tb[0], tb[1]
        W = word_span_mask(attn)
        unk = np.where(W & (ids == config.UNK_ID))[0]
        assert int(r["n_unk"]) == len(unk), f"{name}: n_unk mismatch"
        csv_pos = set(int(x) for x in r["unk_positions"].split("|") if x)
        assert csv_pos == set(unk.tolist()), f"{name}: unk positions mismatch"
    print("T2 PASS: damage_sets.csv replay 30/30 (n_unk, positions, word span)")


def test_augment_invariants():
    splits = load_splits()
    train = splits["train"]
    aug = WordLevelTAV(seed=123)
    n_damaged = 0
    for j in range(400):
        ids0 = train.ids[j].copy()
        ids = train.ids[j].copy()
        audio = train.audio[j].copy()
        vision = train.vision[j].copy()
        attn = train.attn[j]
        L = int(attn.sum())
        damage = aug(ids, attn, audio, vision)
        if damage.any():
            n_damaged += 1
            unk = set(np.where(ids == config.UNK_ID)[0].tolist())
            a0 = set(np.where(np.isclose(audio, 0).all(axis=1))[0].tolist())
            v0 = set(np.where(np.isclose(vision, 0).all(axis=1))[0].tolist())
            hits = set(np.where(damage)[0].tolist())
            W = set(np.where(word_span_mask(attn))[0].tolist())
            assert hits <= W
            assert hits <= unk and hits <= a0 and hits <= v0
            assert ids[0] == config.CLS_ID and ids[L - 1] == config.SEP_ID
            assert np.array_equal(ids[L:], ids0[L:])
            # no NEW unk outside hits
            pre_unk = set(np.where(ids0 == config.UNK_ID)[0].tolist())
            assert unk - pre_unk == hits
    print(f"T3 PASS: augmentation invariants on 400 train samples "
          f"({n_damaged} damaged)")


def test_damage_statistics():
    aug = WordLevelTAV(seed=7)
    expected = aug.p_file * float(np.mean(aug.rate_pool))
    fracs = []
    for j in range(300):
        ids = np.full(50, 5, dtype=np.int64)
        ids[0], ids[19] = config.CLS_ID, config.SEP_ID
        attn = np.zeros(50, dtype=np.int64)
        attn[:20] = 1
        audio = np.random.default_rng(j).normal(size=(50, 74)).astype(np.float32)
        vision = np.random.default_rng(j + 1).normal(size=(50, 35)).astype(np.float32)
        d = aug(ids, attn, audio, vision)
        fracs.append(d.sum() / 18)  # word span = 18
    m = float(np.mean(fracs))
    assert abs(m - expected) < 0.03, f"damage fraction {m:.3f} vs expected {expected:.3f}"
    print(f"T4 PASS: mean damage fraction {m:.4f} ~= p_file*E[pool]={expected:.4f}")


def test_fixed_seed_determinism():
    splits = load_splits()
    sd = splits["valid"]
    from src.datasets import Q2Dataset
    ds = Q2Dataset(sd, Standardizer.fit(splits["train"]), train=False,
                   augmenter=WordLevelTAV(p_file=1.0, rate_pool=[0.3], seed=99),
                   fixed_corrupt_seed=0)
    a = ds[5]
    b = ds[5]
    assert np.array_equal(a["ids"].numpy(), b["ids"].numpy())
    assert np.array_equal(a["audio"].numpy(), b["audio"].numpy())
    # different sample -> different corruption (probabilistically)
    diffs = sum(int(not np.array_equal(ds[k]["ids"].numpy(), ds[k + 1]["ids"].numpy()))
                for k in range(20))
    assert diffs >= 10
    print(f"T5 PASS: fixed-seed corruption deterministic; {diffs}/20 adjacent pairs differ")


def test_standardizer():
    splits = load_splits()
    std = Standardizer.fit(splits["train"])
    tr = splits["train"]
    x = tr.audio[:64]
    obs = tr.audio_obs[:64]
    z = std.transform_audio(x, obs)
    # zeros preserved at unobserved rows
    assert np.isclose(z[~obs], 0).all()
    # stats came from observed rows only: recompute on 3 dims and compare
    for dim in [0, 17, 73]:
        rows = tr.audio[tr.audio_obs][:, dim]
        assert np.isclose(std.a_mean[dim], rows.mean(), atol=1e-4)
        assert np.isclose(std.a_std[dim], rows.std(), atol=1e-4)
    print("T6 PASS: standardizer preserves zeros; stats from observed train rows only")


def test_word_span_edges():
    attn = np.zeros(50, dtype=np.int64)
    attn[:3] = 1
    W = word_span_mask(attn)
    assert W.sum() == 1 and W[1] and not W[0] and not W[2]
    attn2 = np.zeros(50, dtype=np.int64)
    attn2[:2] = 1
    assert word_span_mask(attn2).sum() == 0  # L=2 -> empty (never occurs; guard)
    print("T7 PASS: word span edge cases (L=3 single position, L<3 empty)")


if __name__ == "__main__":
    test_a3_signature()
    test_damage_sets_replay()
    test_augment_invariants()
    test_damage_statistics()
    test_fixed_seed_determinism()
    test_standardizer()
    test_word_span_edges()
    print("ALL TESTS PASSED")
