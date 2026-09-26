"""Integrated Q2/Q3 inference.

Single deployment model: the teammate package's ``p2`` three-seed ensemble
(validated decision bias fitted on VALID only), whose test scores are

    accuracy 0.6754   Macro-F1 0.6485   MAE 0.6145   Pearson 0.6912

on the shared ``aligned_50.pkl`` test split (n=727).  Section 2 of
``docs/两队方案对比与整合分析.md`` shows this is the strongest single model
either team produced, on both teams' selection criteria, so it is used for
BOTH specialised attachments.  Reason for one model rather than a mix: the
problem statement requires one feature version and one input interface for
training, validation and both specialised sets, and the Shapley explanation
must explain the model that actually produced the prediction.

Two things are added on top of the teammate pipeline:

1.  **Q3 evidence is ranked by occlusion effect, not by attention weight.**
    Every word position is deleted on its own and ranked by the drop in the
    explained class probability.  A candidate list built from attention would
    prejudge the ranking the explanation is supposed to justify; the fair
    head-to-head in ``docs/两队方案对比与整合分析.md`` §5.1 found the two
    rankings statistically indistinguishable in magnitude, but the attention
    ranking produced a *negative* effect -- deleting the "evidence" made the
    model more confident -- on 2 of 20 samples, while occlusion produced none.

2.  **The coalition value uses the same calibrated probability as the
    prediction**, so the Shapley efficiency identity closes on the number that
    is actually reported.

Run:
  MATH_E_DATA=<dir with 附件1..附件4> python integrated_inference.py --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import copy
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config as C          # noqa: E402
import io_utils as U        # noqa: E402
import inference as TI      # noqa: E402

MODS = ("text", "audio", "vision")
COALITIONS = [tuple(c) for k in range(4) for c in itertools.combinations(MODS, k)]
W3 = {0: 1.0 / 3.0, 1: 1.0 / 6.0, 2: 1.0 / 3.0}
POLARITY = {0: "Negative", 1: "Neutral", 2: "Positive"}
TAG = "p2"                  # single deployment model
EVIDENCE_K = 5


def calibrated(prob: np.ndarray, bias) -> np.ndarray:
    lp = np.log(np.clip(prob, 1e-9, None)) + np.asarray(bias)
    p = np.exp(lp - lp.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def mask_keep(sample, keep):
    s = copy.deepcopy(sample)
    for m in MODS:
        if m not in keep:
            s.mask[m] = np.zeros_like(s.mask[m])
    return s


def shapley_3player(v: dict) -> dict:
    """Exact Shapley for 3 players. Keys are canonicalised through canon()."""
    v = {canon(k): val for k, val in v.items()}
    full = canon(MODS)
    phi = {}
    for m in MODS:
        o = [x for x in MODS if x != m]
        phi[m] = (W3[0] * (v[(m,)] - v[()])
                  + W3[1] * (v[canon((m, o[0]))] - v[(o[0],)])
                  + W3[1] * (v[canon((m, o[1]))] - v[(o[1],)])
                  + W3[2] * (v[full] - v[canon(o)]))
    return phi


def canon(c) -> tuple:
    return tuple(m for m in MODS if m in c)


def word_positions(sample):
    """Positions carrying a real word (excludes CLS/SEP/padding)."""
    sup = np.asarray(sample.support["text"]) > 0
    idx = np.flatnonzero(sup)
    if len(idx) >= 3:
        return [int(i) for i in idx[1:-1]]
    return [int(i) for i in idx]


def compute_evidence(net, sample, cls: int, bias, base: float):
    """Full occlusion sweep over word positions; returns ranked effects."""
    pos = word_positions(sample)
    if not pos:
        return {}, [], []
    variants = []
    for j in pos:
        s = copy.deepcopy(sample)
        s.text[j, :] = 0.0
        s.mask["text"][j] = 0.0
        variants.append(s)
    res = TI.predict(net, variants)
    p = calibrated(res["prob"], bias)[:, cls]
    effects = {j: float(base - p[k]) for k, j in enumerate(pos)}
    ordered = sorted(effects, key=lambda j: -effects[j])
    top = ordered[:EVIDENCE_K]

    # comparison: what the attention ranking would have picked instead
    attn = None
    try:
        raw = TI.predict(net, [sample])
        if "beta" in raw:
            attn = np.asarray(raw["beta"])[0][0]        # (T,) text evidence weights
    except Exception:
        attn = None
    attn_top = []
    if attn is not None:
        attn_top = sorted(pos, key=lambda j: -float(attn[j]))[:EVIDENCE_K]
    return effects, top, attn_top


def evidence_drop(net, sample, positions, cls, bias, base):
    """Class-probability drop when the listed positions are removed TOGETHER."""
    if not positions:
        return 0.0
    s = copy.deepcopy(sample)
    for j in positions:
        s.text[j, :] = 0.0
        s.mask["text"][j] = 0.0
    p = calibrated(TI.predict(net, [s])["prob"], bias)[0, cls]
    return float(base - p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--package", type=Path, default=HERE.parent)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    bias = np.array(json.loads(
        (a.package / "models" / "decision_calibration.json").read_text(encoding="utf-8"))[TAG])
    net = TI.load_model(TAG)
    print(f"deployment model: {TAG} 3-seed ensemble, valid-fitted bias {bias.tolist()}")

    # ---------------- 附件3 ----------------
    a3 = U.load_a3()
    r3 = TI.predict(net, a3)
    p3cal = calibrated(r3["prob"], bias)
    rows3 = []
    for i, sid in enumerate(r3["sid"]):
        c = int(p3cal[i].argmax())
        rows3.append(dict(sample_id=str(sid), pred_class=c, pred_polarity=POLARITY[c],
                          pred_intensity=round(float(r3["p_reg"][i]), 5),
                          confidence=round(float(p3cal[i, c]), 5),
                          prob_neg=round(float(p3cal[i, 0]), 5),
                          prob_neu=round(float(p3cal[i, 1]), 5),
                          prob_pos=round(float(p3cal[i, 2]), 5),
                          polarity_intensity_conflict=bool(
                              (c == 2 and r3["p_reg"][i] < 0) or (c == 0 and r3["p_reg"][i] > 0))))
    with (out / "附件3_预测.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows3[0].keys()))
        w.writeheader(); w.writerows(rows3)
    print(f"附件3: {len(rows3)} rows -> 附件3_预测.csv "
          f"({ {POLARITY[c]: sum(r['pred_class'] == c for r in rows3) for c in range(3)} })")

    # ---------------- 附件4 ----------------
    a4 = U.load_a4()
    r4 = TI.predict(net, a4)
    p4cal = calibrated(r4["prob"], bias)
    rows4 = []
    for i, s in enumerate(a4):
        cls = int(p4cal[i].argmax())
        base = float(p4cal[i, cls])
        effects, top, attn_top = compute_evidence(net, s, cls, bias, base)
        phi_cls, phi_reg = {}, {}
        v_cls, v_reg = {}, {}
        for keep in COALITIONS:
            mk = mask_keep(s, keep)
            res = TI.predict(net, [mk])
            pc = calibrated(res["prob"], bias)[0]
            v_cls[keep] = float(pc[cls])
            v_reg[keep] = float(res["p_reg"][0])
        phi_cls = shapley_3player(v_cls)
        phi_reg = shapley_3player(v_reg)
        share = {m: abs(phi_cls[m]) / max(sum(abs(x) for x in phi_cls.values()), 1e-12) for m in MODS}
        full = canon(MODS)
        top_drop = evidence_drop(net, s, top, cls, bias, base)
        attn_drop = evidence_drop(net, s, attn_top, cls, bias, base) if attn_top else float("nan")
        rows4.append({
            "sample_id": str(s.sid),
            "pred_class": cls, "pred_polarity": POLARITY[cls],
            "pred_intensity": round(float(r4["p_reg"][i]), 5),
            "confidence": round(base, 5),
            "main_modality": max(MODS, key=lambda m: abs(phi_cls[m])),
            "share_text": round(share["text"], 4), "share_audio": round(share["audio"], 4),
            "share_vision": round(share["vision"], 4),
            "phi_text_cls": round(phi_cls["text"], 6), "phi_audio_cls": round(phi_cls["audio"], 6),
            "phi_vision_cls": round(phi_cls["vision"], 6),
            "phi_text_reg": round(phi_reg["text"], 6), "phi_audio_reg": round(phi_reg["audio"], 6),
            "phi_vision_reg": round(phi_reg["vision"], 6),
            "efficiency_cls": round(v_cls[full] - v_cls[()], 6),
            "sum_phi_cls": round(sum(phi_cls.values()), 6),
            "efficiency_reg": round(v_reg[full] - v_reg[()], 6),
            "sum_phi_reg": round(sum(phi_reg.values()), 6),
            "v_empty_cls": round(v_cls[()], 6), "v_full_cls": round(v_cls[full], 6),
            "evidence_positions_occlusion": ";".join(map(str, top)),
            "evidence_individual_drop": ";".join(f"{effects[j]:.4f}" for j in top),
            "evidence_set_drop_occlusion": round(top_drop, 6),
            "evidence_positions_attention": ";".join(map(str, attn_top)),
            "evidence_set_drop_attention": (round(attn_drop, 6) if attn_top else ""),
            "occlusion_positions_tested": len(effects),
            "attention_matches_occlusion_topk": int(set(top) == set(attn_top)),
            "text_obs_rate": round(float(np.mean(np.asarray(s.mask["text"]) > 0)), 4),
            "audio_obs_rate": round(float(np.mean(np.asarray(s.mask["audio"]) > 0)), 4),
            "vision_obs_rate": round(float(np.mean(np.asarray(s.mask["vision"]) > 0)), 4),
        })
    with (out / "附件4_预测与解释.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows4[0].keys()))
        w.writeheader(); w.writerows(rows4)

    eff = max(abs(r["efficiency_cls"] - r["sum_phi_cls"]) for r in rows4)
    effr = max(abs(r["efficiency_reg"] - r["sum_phi_reg"]) for r in rows4)
    neg_occ = sum(1 for r in rows4 if r["evidence_set_drop_occlusion"] <= 0)
    neg_att = sum(1 for r in rows4 if r["evidence_set_drop_attention"] != "" and
                  float(r["evidence_set_drop_attention"]) <= 0)
    summary = {
        "deployment_model": f"{TAG} three-seed ensemble (teammate package)",
        "valid_fitted_bias": bias.tolist(),
        "test_reference": {"accuracy": 0.6753782668500687, "macro_f1": 0.6485242985549113,
                           "mae": 0.6144672632217407, "pearson": 0.6912329605433559,
                           "n": 727},
        "attachment3": {"n": len(rows3),
                        "class_counts": {POLARITY[c]: sum(r["pred_class"] == c for r in rows3)
                                         for c in range(3)},
                        "direction_conflicts": sum(r["polarity_intensity_conflict"] for r in rows3)},
        "attachment4": {"n": len(rows4),
                        "class_counts": {POLARITY[c]: sum(r["pred_class"] == c for r in rows4)
                                         for c in range(3)},
                        "main_modality_counts": {m: sum(r["main_modality"] == m for r in rows4)
                                                 for m in MODS},
                        "max_efficiency_error_cls": eff, "max_efficiency_error_reg": effr,
                        "occlusion_evidence_nonpositive": neg_occ,
                        "attention_evidence_nonpositive": neg_att,
                        "attention_topk_matches_occlusion": sum(
                            r["attention_matches_occlusion_topk"] for r in rows4)},
    }
    (out / "预测汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
