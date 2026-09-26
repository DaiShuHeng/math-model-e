"""Q2 inference for 附件3 (and, when 768-d text features exist, 附件4).

The deployable model is feature-level: it consumes the competition's aligned
modality features and an observation mask per modality.  For 附件3 the provided
files contain 语音/视觉 and `text_bert` but **no** `text` field (see
docs/缺失结构审计.md §5); the text branch therefore needs an encoder, and the
command accepts either

  --text-source provided   use the 768-d `text` field when the file has it
                           (附件2/附件4).  Fails loudly on 附件3.
  --text-source encoder    encode `text_bert` with a BERT-class checkpoint,
                           mirroring exactly how the organisers produced `text`.

The two paths share one interface: (T, A, V, t_obs, a_obs, v_obs) at 50 aligned
positions.  Which path was used is recorded in the output CSV header row.

Run (from `solution/`):
  python -m src.predict_q2_feature --ckpt weights/v5/best.pt --att 3 \\
      --text-encoder ../models/bert-base-uncased --out results/att3_v5.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from . import config
from .q2_features import (FeatSplit, FeatStandardizer, load_attachment3, load_attachment4)
from .q2_feature_model import load_checkpoint

POLARITY = {0: "Negative", 1: "Neutral", 2: "Positive"}


def encode_text_with_bert(sp: FeatSplit, bert_dir: Path, max_len: int = 50) -> np.ndarray:
    """Produce (N, 50, 768) text features from `text_bert` token ids.

    Mirrors the organisers' pipeline: a pretrained BERT-class encoder applied to
    the aligned token positions.  Requires `transformers` plus a local
    checkpoint; it is *not* needed when the provided `text` field exists.
    """
    from transformers import AutoModel, AutoTokenizer  # local import: optional dep
    tok = AutoTokenizer.from_pretrained(str(bert_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(bert_dir), local_files_only=True).eval()
    if model.config.hidden_size != 768:
        raise ValueError(f"Encoder hidden size {model.config.hidden_size} != 768; "
                         "cannot reproduce the provided text feature space")
    ids_all, attn_all = load_raw_text_bert(sp)
    out = []
    with torch.no_grad():
        for i in range(0, len(ids_all), 64):
            ids = torch.as_tensor(ids_all[i:i + 64].astype(np.int64))
            attn = torch.as_tensor(attn_all[i:i + 64].astype(np.int64))
            h = model(input_ids=ids, attention_mask=attn).last_hidden_state.numpy()
            if h.shape[1] < max_len:
                h = np.pad(h, ((0, 0), (0, max_len - h.shape[1]), (0, 0)))
            out.append(h[:, :max_len].astype(np.float32))
    return np.concatenate(out)


def load_raw_text_bert(sp: FeatSplit):
    """Re-read the per-file `text_bert` for the specialised sets (they are flat)."""
    from .q2_features import read_pickle
    d = config.ATT3_DIR if sp.split == "att3" else config.ATT4_DIR
    ids, attn = [], []
    for p in sorted(Path(d).glob("*.pkl")):
        rec = read_pickle(p)
        rec = rec["test"] if "test" in rec and isinstance(rec["test"], dict) else rec
        tb = np.asarray(rec["text_bert"], dtype=np.float32)
        if tb.ndim == 3:
            tb = tb[0]
        ids.append(tb[0]); attn.append(tb[1])
    return np.stack(ids), np.stack(attn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--att", type=int, required=True, choices=(3, 4))
    ap.add_argument("--text-source", default="provided", choices=("provided", "encoder"))
    ap.add_argument("--text-encoder", type=str)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite {out_path}; choose a new --out")

    model, ck = load_checkpoint(args.ckpt, device=args.device)
    std = FeatStandardizer.from_json(ck["normalizer"])
    sp = load_attachment3() if args.att == 3 else load_attachment4()

    have_text = not np.isclose(sp.text, 0).all()
    if args.text_source == "provided" and not have_text:
        raise SystemExit(f"附件{args.att} files carry no 768-d `text` field; "
                         "re-run with --text-source encoder --text-encoder <bert dir>")
    if args.text_source == "encoder":
        if not args.text_encoder:
            raise SystemExit("--text-source encoder requires --text-encoder")
        sp = FeatSplit(**{**sp.__dict__, "text": encode_text_with_bert(sp, Path(args.text_encoder))})
        sp = FeatSplit(**{**sp.__dict__, "t_obs": (~np.isclose(sp.text, 0).all(-1)).astype(np.float32)})

    arrays = std.transform(sp)
    res = model.predict(arrays, device=args.device)
    prob, reg, gate = res["prob"], res["reg"], res["gate"]
    pred = prob.argmax(-1)

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "id", "n_words", "pred_class", "polarity", "intensity",
                    "prob_neg", "prob_neu", "prob_pos", "gate_text", "gate_audio", "gate_vision",
                    "text_source"])
        for i, sid in enumerate(sp.sample_id):
            w.writerow([sid, sid, int(sp.t_obs[i].sum()), int(pred[i]), POLARITY[int(pred[i])],
                        f"{reg[i]:.4f}", f"{prob[i,0]:.4f}", f"{prob[i,1]:.4f}", f"{prob[i,2]:.4f}",
                        f"{gate[i,0]:.4f}", f"{gate[i,1]:.4f}", f"{gate[i,2]:.4f}",
                        args.text_source])
    summary = {"checkpoint": str(args.ckpt), "attachment": args.att, "n": len(sp),
               "text_source": args.text_source,
               "class_counts": {POLARITY[c]: int((pred == c).sum()) for c in range(3)},
               "mean_intensity": float(reg.mean()),
               "gate_mean": gate.mean(0).round(4).tolist()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (out_path.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
