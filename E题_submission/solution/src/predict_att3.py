"""Q2 final inference on 附件3 (30 unlabeled locally-damaged samples).

Protocol: features are used EXACTLY as delivered (corruption is already inside
the data — no augmentation here); observation masks come from non-zero rows;
standardization uses TRAIN-fitted stats. Multiple checkpoints are ensembled by
averaging class probabilities and regression outputs.

Output CSV columns:
  file, n_words, pred_class, polarity, intensity, prob_neg, prob_neu, prob_pos,
  gate_text, gate_audio, gate_vision

Run:
  cd solution && python -m src.predict_att3 --ckpt weights/q2_ft/s*/best.pt \
      --out results/att3_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from . import config
from .data_adapter import Standardizer, load_splits, read_pickle
from .models import Q2Model

POLARITY = {0: "Negative", 1: "Neutral", 2: "Positive"}


def load_att3_samples():
    samples = []
    for path in sorted(config.ATT3_DIR.glob("*.pkl")):
        d = read_pickle(path)
        d = d.get("test", d)
        tb = np.asarray(d["text_bert"], dtype=np.int64).reshape(3, 50)
        audio = np.asarray(d["audio"], dtype=np.float32).reshape(50, 74)
        vision = np.asarray(d["vision"], dtype=np.float32).reshape(50, 35)
        sid = str(np.asarray(d["id"]).reshape(-1)[0]) if "id" in d else path.stem
        samples.append({"file": path.stem, "id": sid, "ids": tb[0], "attn": tb[1],
                        "audio": audio, "vision": vision})
    return samples


@torch.no_grad()
def predict(checkpoints: list[Path], samples) -> list[dict]:
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    models = []
    for cp in checkpoints:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        m = Q2Model(bert_dir=ck["args"].get("bert_dir", str(config.BERT_DIR)),
                    freeze_text=bool(ck["args"].get("freeze_text", 1))).to(device).eval()
        m.load_state_dict(ck["state_dict"])
        models.append(m)

    std = Standardizer.fit(load_splits()["train"])
    rows = []
    for s in samples:
        audio_obs = ~np.isclose(s["audio"], 0).all(axis=1)
        vision_obs = ~np.isclose(s["vision"], 0).all(axis=1)
        a_z = std.transform_audio(s["audio"][None], audio_obs[None])[0]
        v_z = std.transform_vision(s["vision"][None], vision_obs[None])[0]
        batch = {
            "ids": torch.from_numpy(s["ids"][None]).to(device),
            "attn": torch.from_numpy(s["attn"][None].astype(np.int64)).to(device),
            "audio": torch.from_numpy(a_z[None]).to(device),
            "vision": torch.from_numpy(v_z[None]).to(device),
            "audio_obs": torch.from_numpy(audio_obs[None].astype(np.float32)).to(device),
            "vision_obs": torch.from_numpy(vision_obs[None].astype(np.float32)).to(device),
        }
        probs, regs, gates = [], [], []
        for m in models:
            out = m(batch)
            probs.append(torch.softmax(out["logits"], -1)[0].float().cpu().numpy())
            regs.append(out["reg"][0].float().cpu().numpy())
            gates.append(out["gate"][0].float().cpu().numpy())
        p = np.mean(probs, axis=0)
        reg = float(np.mean(regs))
        gate = np.mean(gates, axis=0)
        rows.append({
            "file": s["file"], "id": s["id"], "n_words": int(s["attn"].sum() - 2),
            "pred_class": int(p.argmax()), "polarity": POLARITY[int(p.argmax())],
            "intensity": round(reg, 4),
            "prob_neg": round(float(p[0]), 4), "prob_neu": round(float(p[1]), 4),
            "prob_pos": round(float(p[2]), 4),
            "gate_text": round(float(gate[0]), 4), "gate_audio": round(float(gate[1]), 4),
            "gate_vision": round(float(gate[2]), 4),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="+", required=True)
    ap.add_argument("--out", type=str, default="results/att3_predictions.csv")
    args = ap.parse_args()
    ckpts = [Path(c) for c in args.ckpt]
    samples = load_att3_samples()
    print(f"附件3: {len(samples)} samples, ensemble of {len(ckpts)} checkpoints")
    rows = predict(ckpts, samples)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print("class distribution:", dict(sorted(Counter(r["pred_class"] for r in rows).items())))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
