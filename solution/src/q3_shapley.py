"""Q3 exact 3-modality Shapley attribution for 附件4 (20 unlabeled samples).

For each sample and each coalition S ⊆ {T,A,V} (8 coalitions), the teammate's
frozen p3 3-seed ensemble is evaluated with the complementary modalities
masked off (their designed absent-modality semantics: mask=0 -> missing token,
gate -1e4). The value v(S) is the calibrated probability of the FULL-model
predicted class (and separately the regression output). Exact Shapley values
for 3 players:

    φ_m = Σ_{S ⊆ M\{m}} |S|!(2-|S|)!/3! * [v(S∪{m}) - v(S)]

so φ_T+φ_A+φ_V = v(TAV) - v(∅) exactly (verified per sample and written to
the CSV). Signed φ (for the predicted class) = direction & magnitude of each
modality's contribution; |φ|/Σ|φ| = normalized importance share. Also reports
the decomposition of the regression output and cross-checks the main modality
against the teammate's deletion-based main_reference_modality.

附件4 remains inference-only: no labels exist, none are read.

Run:
  cd solution && python -m src.q3_shapley            # writes results/附件4_shapley.csv
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

SOL = Path(__file__).resolve().parents[1]
TM = Path("/home/daishuheng/math_competition/math_model_e_optimized")
os.environ.setdefault("MATH_E_DATA", "/home/daishuheng/math_competition/E题/E题数据")
os.environ.setdefault("MATH_E_BERT", "/home/daishuheng/math_competition/models/bert-base-uncased")
sys.path.insert(0, str(TM / "code"))

import io_utils as U                     # noqa: E402  (teammate's modules)
import inference as TI                   # noqa: E402

MODS = ("text", "audio", "vision")
COALITIONS = [tuple(c) for k in range(4) for c in itertools.combinations(MODS, k)]
W = {  # |S|! (2-|S|)! / 3!  for 3 players
    0: 1.0 / 3.0,
    1: 1.0 / 6.0,
    2: 1.0 / 3.0,
}
P3_BIAS = np.array([0.2, -0.2, 0.0])     # models/decision_calibration.json p3
POLARITY = {0: "Negative", 1: "Neutral", 2: "Positive"}


def coalition_value(net, samples, keep, bias):
    """Evaluate the ensemble with only modalities in `keep` available."""
    masked = []
    for s in samples:
        mask = {m: (s.mask[m].copy() if m in keep else np.zeros_like(s.mask[m]))
                for m in MODS}
        masked.append(replace(s, mask=mask))
    res = TI.predict(net, masked)
    logp = np.log(res["prob"].clip(1e-9)) + bias
    prob = np.exp(logp - logp.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    return prob, res["p_reg"], res["alpha"]


def shapley_3player(v: dict[tuple, float], m: str) -> float:
    others = [x for x in MODS if x != m]
    φ = W[0] * (v[key((m,))] - v[key(())])
    φ += W[1] * (v[key((m, others[0]))] - v[key((others[0],))])
    φ += W[1] * (v[key((m, others[1]))] - v[key((others[1],))])
    φ += W[2] * (v[key(MODS)] - v[key(others)])
    return φ


def key(c):
    """Canonical coalition key: modality order as in MODS (matches COALITIONS)."""
    return tuple(sorted(set(c), key=MODS.index))


def main():
    samples = U.load_a4()
    net = TI.load_model("p3")

    # coalition values per sample: v[coalition_key] = arrays over samples
    probs, regs, alphas = {}, {}, {}
    for c in COALITIONS:
        p, r, a = coalition_value(net, samples, set(c), P3_BIAS)
        probs[key(c)] = p
        regs[key(c)] = r
        alphas[key(c)] = a

    full = probs[key(tuple(MODS))]
    pred = full.argmax(1)

    rows = []
    for j, s in enumerate(samples):
        c = int(pred[j])
        # v(.) for the predicted class probability and for the regression head
        v_cls = {key(co): float(probs[key(co)][j, c]) for co in COALITIONS}
        v_reg = {key(co): float(regs[key(co)][j]) for co in COALITIONS}
        φ_cls = {m: shapley_3player(v_cls, m) for m in MODS}
        φ_reg = {m: shapley_3player(v_reg, m) for m in MODS}
        # exactness check: Σφ = v(TAV) - v(∅)
        sum_cls = sum(φ_cls.values())
        eff_cls = v_cls[key(tuple(MODS))] - v_cls[key(tuple())]
        sum_reg = sum(φ_reg.values())
        eff_reg = v_reg[key(tuple(MODS))] - v_reg[key(tuple())]
        abs_cls = {m: abs(φ_cls[m]) for m in MODS}
        total_abs = sum(abs_cls.values()) + 1e-12
        main_mod = max(abs_cls, key=abs_cls.get)
        rows.append({
            "sample_id": s.sid,
            "pred_class": c,
            "pred_polarity": POLARITY[c],
            "pred_intensity": round(float(full[j].argmax() * 0 + regs[key(tuple(MODS))][j]), 4),
            "pred_prob": round(float(full[j, c]), 4),
            "main_modality_shapley": main_mod,
            "phi_text_cls": round(φ_cls["text"], 5),
            "phi_audio_cls": round(φ_cls["audio"], 5),
            "phi_vision_cls": round(φ_cls["vision"], 5),
            "share_text": round(abs_cls["text"] / total_abs, 4),
            "share_audio": round(abs_cls["audio"] / total_abs, 4),
            "share_vision": round(abs_cls["vision"] / total_abs, 4),
            "phi_text_reg": round(φ_reg["text"], 5),
            "phi_audio_reg": round(φ_reg["audio"], 5),
            "phi_vision_reg": round(φ_reg["vision"], 5),
            "efficiency_cls": round(eff_cls, 5),
            "sum_phi_cls": round(sum_cls, 5),
            "efficiency_reg": round(eff_reg, 5),
            "sum_phi_reg": round(sum_reg, 5),
            "v_empty_cls": round(v_cls[key(tuple())], 5),
            "v_T_cls": round(v_cls[key(("text",))], 5),
            "v_A_cls": round(v_cls[key(("audio",))], 5),
            "v_V_cls": round(v_cls[key(("vision",))], 5),
            "v_TA_cls": round(v_cls[key(("text", "audio"))], 5),
            "v_TV_cls": round(v_cls[key(("text", "vision"))], 5),
            "v_AV_cls": round(v_cls[key(("audio", "vision"))], 5),
            "v_TAV_cls": round(v_cls[key(tuple(MODS))], 5),
            "alpha_full": np.round(alphas[key(tuple(MODS))][j], 4).tolist(),
        })
        print(f"{s.sid}: pred {POLARITY[c]:<8} main={main_mod:<6} "
              f"φ_cls T {φ_cls['text']:+.3f} A {φ_cls['audio']:+.3f} V {φ_cls['vision']:+.3f} "
              f"| Σφ {sum_cls:+.4f} vs eff {eff_cls:+.4f}", flush=True)

    out = SOL / "results" / "附件4_shapley.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    max_err = max(abs(r["sum_phi_cls"] - r["efficiency_cls"]) for r in rows)
    print(f"\nwrote {out}")
    print(f"exactness max |Σφ - efficiency| = {max_err:.2e} (float rounding only)")

    # cross-check vs teammate's deletion-based main modality
    try:
        import pandas as pd
        # dtype=str: "01" must stay a string or the sid join silently fails
        # (pandas coerces it to int 1) — first run reported 0/20 on this bug.
        theirs = pd.read_csv(TM / "results" / "附件4_优化预测与解释.csv", dtype={0: str})
        theirs = theirs.set_index(theirs.columns[0])
        agree = 0
        for r in rows:
            sid = r["sample_id"]
            if sid in theirs.index:
                tm_main = str(theirs.loc[sid, "main_reference_modality"])
                if tm_main == r["main_modality_shapley"]:
                    agree += 1
        print(f"main-modality agreement with deletion method: {agree}/{len(rows)}")
    except Exception as e:  # noqa: BLE001
        print("cross-check skipped:", e)


if __name__ == "__main__":
    main()
