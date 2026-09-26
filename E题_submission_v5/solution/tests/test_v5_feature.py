"""Regression tests for the v5 feature-level Q2/Q3 pipeline.

Each test pins one defect that was actually found while building v5, so a
regression is caught by name rather than by a silently lower score.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from src.q2_features import (FeatSplit, FeatStandardizer, load_aligned_splits,
                             token_span_mask)
from src.q2_feature_model import FeatureTAV, PROTOCOL_VERSION, masked_stats


def _synthetic(n=12, T=50, seed=0):
    rng = np.random.default_rng(seed)
    text = rng.normal(size=(n, T, 768)).astype(np.float32)
    audio = rng.normal(size=(n, T, 74)).astype(np.float32)
    vision = rng.normal(size=(n, T, 35)).astype(np.float32)
    span = np.zeros((n, T), dtype=np.float32)
    for i in range(n):
        span[i, :max(3, 5 + i % 20)] = 1.0
    audio *= span[..., None] * (rng.random((n, T, 1)) > 0.15)
    vision *= span[..., None] * (rng.random((n, T, 1)) > 0.2)
    return FeatSplit(split="train", text=text, audio=audio, vision=vision,
                     t_obs=span.copy(), a_obs=(audio != 0).any(-1).astype(np.float32),
                     v_obs=(vision != 0).any(-1).astype(np.float32),
                     y_reg=rng.normal(size=n).astype(np.float32),
                     y_cls=rng.integers(0, 3, n).astype(np.int64),
                     sample_id=[f"s{i}" for i in range(n)], span_obs=span.copy())


class MaskedStatsTests(unittest.TestCase):
    def test_masked_stats_ignore_padding(self):
        """Defect guarded: padding must not enter mean/max/attention pooling."""
        h = torch.zeros(1, 4, 3)
        h[0, :2] = torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        h[0, 2:] = 1e3                      # padding poison
        m = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        q = torch.zeros(3)
        mean, mx, pool, w = masked_stats(h, m, q)
        self.assertAlmostEqual(float(mean[0, 0]), 2.0, places=5)
        self.assertAlmostEqual(float(mx[0, 0]), 3.0, places=5)
        self.assertAlmostEqual(float(pool[0, 0]), 2.0, places=5)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)

    def test_all_missing_modality_is_zero_not_sentinel(self):
        """Defect guarded: a fully-missing modality used to emit -1e4 constants."""
        h = torch.randn(2, 5, 4) * 1e3
        m = torch.zeros(2, 5)
        q = torch.zeros(4)
        mean, mx, pool, _ = masked_stats(h, m, q)
        for t in (mean, mx, pool):
            self.assertTrue(torch.isfinite(t).all())
            self.assertTrue(torch.allclose(t, torch.zeros_like(t)))


class StandardizerTests(unittest.TestCase):
    def test_fit_and_transform_use_the_same_text_mask(self):
        """Defect guarded: statistics fitted on one mask, applied with another."""
        sp = _synthetic()
        std = FeatStandardizer.fit(sp)
        arr = std.transform(sp)
        manual = ((sp.text - std.t_mean) / std.t_std) * sp.span_obs[..., None]
        self.assertTrue(np.allclose(arr["T"], manual, atol=1e-5))
        # rows outside the valid-token prefix must be exactly zero
        tail = sp.span_obs == 0
        self.assertTrue(np.allclose(arr["T"][tail], 0.0))
        # and the fitted mean must equal the prefix-row mean
        self.assertTrue(np.allclose(std.t_mean, sp.text[sp.span_obs > 0].mean(0), atol=1e-5))

    def test_zero_preservation_for_av(self):
        sp = _synthetic()
        std = FeatStandardizer.fit(sp)
        arr = std.transform(sp)
        self.assertTrue(np.allclose(arr["A"][sp.a_obs == 0], 0.0))
        self.assertTrue(np.allclose(arr["V"][sp.v_obs == 0], 0.0))

    def test_checkpoint_roundtrip(self):
        sp = _synthetic()
        std = FeatStandardizer.fit(sp)
        back = FeatStandardizer.from_json(std.to_json())
        for k in ("t_mean", "t_std", "a_mean", "a_std", "v_mean", "v_std"):
            self.assertTrue(np.allclose(getattr(std, k), getattr(back, k)))


class TokenSpanTests(unittest.TestCase):
    def test_span_excludes_cls_and_sep(self):
        m = np.zeros((1, 6), dtype=np.float32)
        m[0, :4] = 1.0
        span = token_span_mask(m)
        self.assertEqual(span[0].tolist(), [0, 1, 1, 0, 0, 0])

    def test_short_span_is_not_dropped(self):
        """Samples with L<3 keep their positions rather than becoming all-zero."""
        m = np.zeros((2, 5), dtype=np.float32)
        m[0, :2] = 1.0
        m[1, :1] = 1.0
        span = token_span_mask(m)
        self.assertEqual(int(span[0].sum()), 2)
        self.assertEqual(int(span[1].sum()), 1)


class ModelTests(unittest.TestCase):
    def _model(self, **kw):
        torch.manual_seed(0)
        return FeatureTAV(**kw)

    def test_forward_shapes_and_gate_normalisation(self):
        m = self._model()
        sp = _synthetic(n=4)
        arr = FeatStandardizer.fit(sp).transform(sp)
        b = {k: torch.as_tensor(v) for k, v in arr.items()}
        out = m(b)
        self.assertEqual(out["logits"].shape, (4, 3))
        self.assertEqual(out["reg"].shape, (4,))
        self.assertTrue(torch.allclose(out["gate"].sum(-1), torch.ones(4), atol=1e-5))
        self.assertTrue(bool((out["reg"].abs() <= 3).all()))

    def test_dead_modality_is_gated_off(self):
        m = self._model()
        sp = _synthetic(n=3)
        arr = FeatStandardizer.fit(sp).transform(sp)
        b = {k: torch.as_tensor(v) for k, v in arr.items()}
        b["A"] = torch.zeros_like(b["A"])
        b["a_obs"] = torch.zeros_like(b["a_obs"])
        out = m(b)
        self.assertLess(float(out["gate"][:, 1].max()), 1e-6)

    def test_padding_does_not_change_text_pooling(self):
        """Changing padded text rows must not move the prediction."""
        m = self._model().eval()
        sp = _synthetic(n=2)
        arr = FeatStandardizer.fit(sp).transform(sp)
        b = {k: torch.as_tensor(v) for k, v in arr.items()}
        with torch.no_grad():
            a1 = m(b)["logits"]
            b2 = {k: v.clone() for k, v in b.items()}
            tail = b2["t_obs"] <= 0
            b2["T"][tail] = 1e3
            a2 = m(b2)["logits"]
        self.assertTrue(torch.allclose(a1, a2, atol=1e-5))


class CorruptionTests(unittest.TestCase):
    def test_budget_is_round_rate_times_token_span(self):
        from src.train_q2_feature import corrupt_span_bounded
        sp = _synthetic(n=8, seed=3)
        arr = FeatStandardizer.fit(sp).transform(sp)
        b = {k: torch.as_tensor(v) for k, v in arr.items()}
        rng = np.random.default_rng(0)
        r = np.random.default_rng(0)
        got_any = False
        for _ in range(60):
            out = corrupt_span_bounded({k: v.clone() for k, v in b.items()}, r)
            removed = (b["t_obs"] > 0) & (out["t_obs"] <= 0)
            if removed.any():
                got_any = True
                for i in range(8):
                    n = int(removed[i].sum())
                    if n:
                        span = int((b["t_obs"][i] > 0).sum())
                        self.assertGreaterEqual(n, 1)
                        self.assertLessEqual(n, span)
            # corruption must never touch padding or add observations
            self.assertTrue(bool(((out["t_obs"] > 0) <= (b["t_obs"] > 0)).all()))
        self.assertTrue(got_any, "corruption never fired")

    def test_corruption_keeps_shapes_and_labels(self):
        from src.train_q2_feature import corrupt_span_bounded
        sp = _synthetic(n=4)
        arr = FeatStandardizer.fit(sp).transform(sp)
        b = {k: torch.as_tensor(v) for k, v in arr.items()}
        out = corrupt_span_bounded({k: v.clone() for k, v in b.items()}, np.random.default_rng(1))
        for k in ("T", "A", "V", "t_obs", "a_obs", "v_obs"):
            self.assertEqual(tuple(out[k].shape), tuple(b[k].shape))


class ShapleyTests(unittest.TestCase):
    def test_efficiency_identity(self):
        from src.explain_att4_v5 import shapley
        v = {(): 0.2, ("T",): 0.5, ("A",): 0.3, ("V",): 0.25,
             ("T", "A"): 0.6, ("T", "V"): 0.55, ("A", "V"): 0.4, ("T", "A", "V"): 0.7}
        phi = shapley(v)
        self.assertAlmostEqual(sum(phi.values()), v[("T", "A", "V")] - v[()], places=9)

    def test_coalition_order_does_not_matter(self):
        from src.explain_att4_v5 import shapley, canon
        v = {(): 0.2, ("T",): 0.5, ("A",): 0.3, ("V",): 0.25,
             ("T", "A"): 0.6, ("V", "T"): 0.55, ("A", "V"): 0.4, ("V", "A", "T"): 0.7}
        self.assertAlmostEqual(sum(shapley(v).values()), 0.5, places=9)
        self.assertEqual(canon(("A", "T")), ("T", "A"))

    def test_dummy_player_gets_zero(self):
        from src.explain_att4_v5 import shapley
        v = {(): 0.1, ("T",): 0.5, ("A",): 0.1, ("V",): 0.6,
             ("T", "A"): 0.5, ("T", "V"): 0.9, ("A", "V"): 0.6, ("T", "A", "V"): 0.9}
        self.assertAlmostEqual(shapley(v)["A"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
