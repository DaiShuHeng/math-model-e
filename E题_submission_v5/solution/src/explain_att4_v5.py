"""Q3 deliverable: 附件4 predictions + exact Shapley explanation for the v5 ensemble.

Explains the *ensemble*: for every coalition S the value v(S) is the mean over the
selected checkpoints of the predicted probability of the ensemble's full-input
class, so Σ_m φ_m = v(M) − v(∅) still holds exactly for the explained object.

Run (from `solution/`):
  python -m src.explain_att4_v5 --ckpt weights/v5_deploy/c3_s2026.pt ... \\
      --out results/附件4_预测与解释_v5.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from . import config
from .q2_features import FeatStandardizer, load_aligned_splits, load_attachment4
from .q2_feature_model import load_checkpoint

MODS = ("T", "A", "V")
COALS = [tuple(c) for k in range(4) for c in itertools.combinations(MODS, k)]
W = {0: 1 / 3, 1: 1 / 6, 2: 1 / 3}
POLARITY = {0: "Negative", 1: "Neutral", 2: "Positive"}


def mask_keep(arr: dict, keep: tuple) -> dict:
    out = dict(arr)
    for m in MODS:
        if m in keep:
            continue
        out[m] = np.zeros_like(arr[m])
        out[f"{m.lower()}_obs"] = np.zeros_like(arr[f"{m.lower()}_obs"])
    return out


def canon(c) -> tuple:
    """Canonical coalition key: order fixed by MODS, so ('A','T') == ('T','A')."""
    return tuple(m for m in MODS if m in c)


def shapley(v: dict) -> dict:
    """Exact Shapley for 3 players; keys of v are canonicalised through canon()."""
    v = {canon(k): val for k, val in v.items()}
    phi = {}
    full = canon(MODS)
    for m in MODS:
        o = [x for x in MODS if x != m]
        phi[m] = (W[0] * (v[(m,)] - v[()])
                  + W[1] * (v[canon((m, o[0]))] - v[(o[0],)])
                  + W[1] * (v[canon((m, o[1]))] - v[(o[1],)])
                  + W[2] * (v[full] - v[canon(o)]))
    return phi


def _set_effect(models, one: dict, cls: int, base: float, positions) -> float:
    """Class-probability drop when the listed positions are removed TOGETHER."""
    if not positions:
        return 0.0
    mod = {k: v.copy() for k, v in one.items()}
    for j in positions:
        mod["T"][0, j] = 0.0
        mod["t_obs"][0, j] = 0.0
    after = float(np.mean([m.predict(mod)["prob"][0, cls] for m in models]))
    return base - after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--evidence-k", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")

    models = [load_checkpoint(p, device=a.device)[0] for p in a.ckpt]
    std = FeatStandardizer.from_json(load_checkpoint(a.ckpt[0])[1]["normalizer"])

    # sanity: the ensemble must reproduce the valid numbers before explaining it
    va = load_aligned_splits(("valid",))["valid"]
    av = std.transform(va)
    valid_prob = np.mean([m.predict(av)["prob"] for m in models], 0)
    valid_reg = np.mean([m.predict(av)["reg"] for m in models], 0)
    from .metrics import cls_metrics, reg_metrics
    valid = {**cls_metrics(va.y_cls, valid_prob.argmax(-1)), **reg_metrics(va.y_reg, valid_reg)}
    print("SANITY valid:", {k: round(float(v), 4) for k, v in valid.items()}, flush=True)

    sp = load_attachment4()
    arr = std.transform(sp)
    full_p = np.mean([m.predict(arr)["prob"] for m in models], 0)
    full_r = np.mean([m.predict(arr)["reg"] for m in models], 0)
    pred = full_p.argmax(-1)

    rows = []
    for i in range(len(sp)):
        one = {k: v[i:i + 1] for k, v in arr.items()}
        coalition_cls, coalition_reg = {}, {}
        for keep in COALS:
            mk = mask_keep(one, keep)
            coalition_cls[keep] = float(np.mean([m.predict(mk)["prob"][0, pred[i]] for m in models]))
            coalition_reg[keep] = float(np.mean([m.predict(mk)["reg"][0] for m in models]))
        pc, pr = shapley(coalition_cls), shapley(coalition_reg)
        full = canon(MODS)
        share = {m: abs(pc[m]) / max(sum(abs(x) for x in pc.values()), 1e-12) for m in MODS}
        # ---- local evidence: full occlusion sweep, no attention circularity ----
        # A candidate list built from attention weights would prejudge the very
        # ranking the explanation is supposed to justify.  Instead every word
        # position of the sample is deleted on its own and the drop in the
        # explained class probability is measured (one batched forward pass over
        # the deleted variants).  Attention is kept as a separate column so the
        # two rankings can be compared; they agree only partially.
        base = float(np.mean([m.predict(one)["prob"][0, pred[i]] for m in models]))
        word_idx = [int(j) for j in np.flatnonzero(one["word_span"][0] > 0)]
        attn_w = np.mean([m.predict(one)["text_attn"][0] for m in models], 0)
        effects = {}
        if word_idx:
            rep = len(word_idx)
            var = {k: np.repeat(v, rep, axis=0) for k, v in one.items()}
            for r, j in enumerate(word_idx):
                var["T"][r, j] = 0.0
                var["t_obs"][r, j] = 0.0
            pv = np.mean([m.predict(var)["prob"][:, pred[i]] for m in models], 0)
            effects = {j: base - float(pv[r]) for r, j in enumerate(word_idx)}
        k = min(a.evidence_k, len(effects))
        top_by_effect = sorted(effects, key=lambda j: -effects[j])[:k]
        top_by_attn = sorted(word_idx, key=lambda j: -attn_w[j])[:k]
        set_effect = {name: _set_effect(models, one, pred[i], base, pos)
                      for name, pos in (("effect", top_by_effect), ("attn", top_by_attn))}
        rows.append({
            "sample_id": sp.sample_id[i], "pred_class": int(pred[i]),
            "pred_polarity": POLARITY[int(pred[i])],
            "pred_intensity": round(float(full_r[i]), 4),
            "pred_prob": round(float(full_p[i, pred[i]]), 4),
            "prob_neg": round(float(full_p[i, 0]), 4), "prob_neu": round(float(full_p[i, 1]), 4),
            "prob_pos": round(float(full_p[i, 2]), 4),
            "a_obs_rate": round(float(one["a_obs"].mean()), 4),
            "v_obs_rate": round(float(one["v_obs"].mean()), 4),
            "phi_text_cls": round(pc["T"], 6), "phi_audio_cls": round(pc["A"], 6),
            "phi_vision_cls": round(pc["V"], 6),
            "main_modality": max(MODS, key=lambda m: abs(pc[m])),
            "share_text": round(share["T"], 4), "share_audio": round(share["A"], 4),
            "share_vision": round(share["V"], 4),
            "phi_text_reg": round(pr["T"], 6), "phi_audio_reg": round(pr["A"], 6),
            "phi_vision_reg": round(pr["V"], 6),
            "efficiency_cls": round(coalition_cls[full] - coalition_cls[()], 6), "sum_phi_cls": round(sum(pc.values()), 6),
            "efficiency_reg": round(coalition_reg[full] - coalition_reg[()], 6), "sum_phi_reg": round(sum(pr.values()), 6),
            "v_empty_cls": round(coalition_cls[()], 6), "v_full_cls": round(coalition_cls[full], 6),
            "evidence_positions": ";".join(map(str, top_by_effect)),
            "evidence_individual_drop": ";".join(f"{effects[j]:.4f}" for j in top_by_effect),
            "evidence_attention": ";".join(f"{attn_w[j]:.4f}" for j in top_by_effect),
            "attention_top_positions": ";".join(map(str, top_by_attn)),
            "attention_top_attention": ";".join(f"{attn_w[j]:.4f}" for j in top_by_attn),
            "occlusion_positions_tested": len(word_idx),
            "effect_ranked_set_drop": round(set_effect["effect"], 6),
            "attention_ranked_set_drop": round(set_effect["attn"], 6),
            "rankings_agree": int(set(top_by_effect) == set(top_by_attn)),
            "positive_individual_effects": int(sum(e > 0 for e in effects.values())),
        })
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    eff = max(abs(r["efficiency_cls"] - r["sum_phi_cls"]) for r in rows)
    summary = {"rows": len(rows), "checkpoints": a.ckpt, "valid_ensemble": valid,
               "max_efficiency_error": eff,
               "main_modality_counts": {m: sum(r["main_modality"] == m for r in rows) for m in MODS},
               "class_counts": {POLARITY[c]: sum(r["pred_class"] == c for r in rows) for c in range(3)}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
