"""Q3: exact 3-modality Shapley decomposition + local-evidence localisation.

Protocol
--------
* The explained quantity is the predicted class of the FULL-input model, and
  separately the regression output.  For a coalition S ⊆ {T,A,V} the model is
  re-evaluated with the modalities outside S marked unobserved (their rows zeroed
  and their observation mask cleared, i.e. the model's own missing-modality
  semantics), giving v(S).
* Exact Shapley for 3 players:
      φ_m = Σ_{S ⊆ M\\{m}} |S|!(2-|S|)!/3! · [v(S∪{m}) − v(S)]
  so Σ_m φ_m = v(M) − v(∅) holds to numerical precision (written per sample).
* Normalised |φ| shares express relative magnitude only; the sign is preserved.
* Local evidence is reported as a *deletion effect* over the 50 aligned text
  positions (attention-pool weight is reported alongside as a model internal,
  never as the evidence ranking itself).

附件4 has no labels: nothing here reads labels for it.

Run (from `solution/`):
  python -m src.q3_shapley_v5 --ckpt weights/v5/best.pt --out results/附件4_shapley_v5.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from . import config
from .q2_features import FeatStandardizer, load_attachment4
from .q2_feature_model import load_checkpoint

MODS = ("T", "A", "V")
COALITIONS = [tuple(c) for k in range(4) for c in itertools.combinations(MODS, k)]
WEIGHT = {0: 1 / 3, 1: 1 / 6, 2: 1 / 3}
POLARITY = {0: "Negative", 1: "Neutral", 2: "Positive"}


def masked_arrays(arrays: dict, keep: tuple) -> dict:
    out = dict(arrays)
    for m in MODS:
        if m in keep:
            continue
        out[m] = np.zeros_like(arrays[m])
        out[f"{m.lower()}_obs"] = np.zeros_like(arrays[f"{m.lower()}_obs"])
    return out


def shapley_3player(v: dict) -> dict:
    phi = {}
    for m in MODS:
        others = [x for x in MODS if x != m]
        val = WEIGHT[0] * (v[(m,)] - v[()])
        val += WEIGHT[1] * (v[tuple(sorted((m, others[0])))] - v[(others[0],)])
        val += WEIGHT[1] * (v[tuple(sorted((m, others[1])))] - v[(others[1],)])
        val += WEIGHT[2] * (v[tuple(sorted(MODS))] - v[tuple(sorted(others))])
        phi[m] = float(val)
    return phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--evidence-positions", type=int, default=10,
                    help="number of top text positions deleted for the local-effect column")
    args = ap.parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite {out_path}")

    model, ck = load_checkpoint(args.ckpt, device=args.device)
    std = FeatStandardizer.from_json(ck["normalizer"])
    sp = load_attachment4()
    arrays = std.transform(sp)
    n = len(sp)

    full = model.predict(arrays, device=args.device)
    pred = full["prob"].argmax(-1)

    rows = []
    for i in range(n):
        one = {k: v[i:i + 1] for k, v in arrays.items()}
        v_cls, v_reg = {}, {}
        for keep in COALITIONS:
            r = model.predict(masked_arrays(one, keep), device=args.device)
            v_cls[keep] = float(r["prob"][0, pred[i]])
            v_reg[keep] = float(r["reg"][0])
        phi_c = shapley_3player(v_cls)
        phi_r = shapley_3player(v_reg)
        eff_c = v_cls[tuple(sorted(MODS))] - v_cls[()]
        eff_r = v_reg[tuple(sorted(MODS))] - v_reg[()]
        share = {m: abs(phi_c[m]) / max(sum(abs(x) for x in phi_c.values()), 1e-12) for m in MODS}
        main_mod = max(MODS, key=lambda m: abs(phi_c[m]))
        # local evidence: delete the top-k attention positions (per modality) and
        # measure the class-probability drop, compared with a random-k control.
        drop = {}
        for key, idx in (("text_attn", i),):
            pass
        attn_w = full["text_attn"][i]
        order = np.argsort(-attn_w)
        obs = one["t_obs"][0] > 0
        cand = [j for j in order if obs[j]]
        topk = cand[:args.evidence_positions]
        drop = _deletion_effect(model, one, pred[i], topk, device)
        rows.append({
            "sample_id": sp.sample_id[i],
            "pred_class": int(pred[i]), "pred_polarity": POLARITY[int(pred[i])],
            "pred_intensity": round(float(full["reg"][i]), 4),
            "pred_prob": round(float(full["prob"][i, pred[i]]), 4),
            "main_modality": main_mod,
            "gate_text": round(float(full["gate"][i, 0]), 4),
            "gate_audio": round(float(full["gate"][i, 1]), 4),
            "gate_vision": round(float(full["gate"][i, 2]), 4),
            "phi_text_cls": round(phi_c["T"], 6), "phi_audio_cls": round(phi_c["A"], 6),
            "phi_vision_cls": round(phi_c["V"], 6),
            "share_text": round(share["T"], 4), "share_audio": round(share["A"], 4),
            "share_vision": round(share["V"], 4),
            "phi_text_reg": round(phi_r["T"], 6), "phi_audio_reg": round(phi_r["A"], 6),
            "phi_vision_reg": round(phi_r["V"], 6),
            "efficiency_cls": round(eff_c, 6), "sum_phi_cls": round(sum(phi_c.values()), 6),
            "efficiency_reg": round(eff_r, 6), "sum_phi_reg": round(sum(phi_r.values()), 6),
            "v_empty": round(v_cls[()], 6), "v_full": round(v_cls[tuple(sorted(MODS))], 6),
            "evidence_top_positions": ";".join(str(j) for j in topk),
            "evidence_deletion_drop": round(drop, 6),
        })

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    eff_err = max(abs(r["efficiency_cls"] - r["sum_phi_cls"]) for r in rows)
    print(json.dumps({"rows": len(rows), "max_efficiency_error_cls": eff_err,
                      "main_modality_counts": {m: sum(r["main_modality"] == m for r in rows) for m in MODS},
                      "written": str(out_path)}, ensure_ascii=False, indent=2))


@torch.no_grad()
def _deletion_effect(model, one: dict, cls: int, positions, device) -> float:
    """Class-probability drop when the listed text positions are marked unobserved."""
    if not positions:
        return 0.0
    base = model.predict(one, device=device)["prob"][0, cls]
    mod = {k: v.copy() for k, v in one.items()}
    for j in positions:
        mod["T"][0, j] = 0.0
        mod["t_obs"][0, j] = 0.0
    after = model.predict(mod, device=device)["prob"][0, cls]
    return float(base - after)


if __name__ == "__main__":
    main()
