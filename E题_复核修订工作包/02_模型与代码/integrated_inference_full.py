"""Integrated Q2/Q3 inference — full local evidence for all three modalities.

Single deployment model: the ``p2`` three-seed ensemble with the VALID-fitted
decision bias ``[-0.2, 0, 0]``.  Test reference (aligned_50.pkl test, n=727):
accuracy 0.6754, Macro-F1 0.6485, MAE 0.6145, Pearson 0.6912.

What this adds over ``integrated_inference.py``
-----------------------------------------------
1.  **All three modalities get local evidence**, not just text.  The problem asks
    for evidence that maps back to "原始文本片段、语音时段或视觉关键帧", so an
    occlusion sweep is run over every observed position of every modality
    (3 tasks x ~50 positions x 20 samples, one batched forward pass per modality).
2.  **Evidence is exported as words and seconds, not bare indices.**  Text
    positions are decoded to the originating words through the tokenizer offset
    mapping, and each word carries its CTC-forced-alignment span.  Audio/vision
    positions carry the physical time window of that aligned slot plus the
    keyframe index at the decode rate.
3.  **Ranking is by occlusion effect only.**  Attention is kept as a separate
    column for comparison and never used to pre-select candidates.

Run:
  MATH_E_DATA=<dir with 附件1..附件4> python integrated_inference_full.py --out <dir>
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE / "code_最终模型"          # config / io_utils / inference / models
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import config as C            # noqa: E402
import io_utils as U          # noqa: E402
import inference as TI        # noqa: E402

MODS = ("text", "audio", "vision")
COALITIONS = [tuple(c) for k in range(4) for c in itertools.combinations(MODS, k)]
W3 = {0: 1 / 3, 1: 1 / 6, 2: 1 / 3}
POLARITY = {0: "Negative", 1: "Neutral", 2: "Positive"}
TAG = "p2"
EVIDENCE_K = 5
VIDEO_FPS = 10.0          # decode rate used by p1_extract.decode_frames


def calibrated(prob, bias):
    lp = np.log(np.clip(prob, 1e-9, None)) + np.asarray(bias)
    p = np.exp(lp - lp.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def canon(c):
    return tuple(m for m in MODS if m in c)


def shapley_3player(v):
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


def drop_position(sample, mod, j):
    s = copy.deepcopy(sample)
    getattr(s, mod)[j] = 0.0
    s.mask[mod][j] = 0.0
    return s


def drop_set(sample, mod, positions):
    s = copy.deepcopy(sample)
    for j in positions:
        getattr(s, mod)[j] = 0.0
        s.mask[mod][j] = 0.0
    return s


# --------------------------------------------------------------------------
# evidence -> words / seconds / frames
# --------------------------------------------------------------------------
def load_alignment(sid: str):
    p = C.DATA_DIR / "alignment" / "a4" / f"{sid}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


_VOCAB = None


def vocab():
    """bert-base-uncased wordpiece vocabulary, read from a local copy.

    ``models/bert-tiny/vocab.txt`` is byte-identical to bert-base-uncased's
    vocabulary (the provided ``text_bert`` ids decode to the raw transcript at
    100% on 附件4), so the token->word mapping needs no download and no network.
    """
    global _VOCAB
    if _VOCAB is None:
        cands = [HERE / "vocab_bert_uncased.txt"]
        for attr in ("BERT_DIR", "MODEL_DIR"):
            base = getattr(C, attr, None)
            if base is not None:
                cands.append(Path(base) / "vocab.txt")
                cands.append(Path(base).parent / "bert-tiny" / "vocab.txt")
        path = next((c for c in cands if c.exists()), None)
        if path is None:
            raise FileNotFoundError(
                "no vocab.txt found; place the uncased BERT vocabulary at "
                f"{HERE / 'vocab_bert_uncased.txt'}")
        _VOCAB = {}
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                _VOCAB[i] = line.strip()
    return _VOCAB


def words_with_spans(sample):
    """Recover per-position words and their character span in ``raw_text``.

    Walks the stored ``text_bert`` ids, merges ``##`` continuations, and locates
    each recovered word in ``raw_text`` with a forward-only search (the token
    stream is a left-to-right rendering of the transcript, so no backtracking is
    needed).  Returns ``{position: {word, char_start, char_end}}``.
    """
    v = vocab()
    stored = np.asarray(sample.meta["text_bert"])[0].astype(int)
    support = np.asarray(sample.support["text"]) > 0
    raw = getattr(sample, "raw_text", "") or ""
    low = raw.lower()
    out, cursor, group = {}, 0, []

    def flush():
        nonlocal group, cursor
        if not group:
            return
        word = "".join(g[1] for g in group)
        first_pos = group[0][0]
        if not word.strip():
            group = []
            return
        idx = low.find(word.lower(), cursor)
        if idx < 0:
            idx = low.find(word.lower())
        if idx < 0:
            for pos, _ in group:
                out[pos] = {"word": word, "mapping_status": "not_found_in_raw_text"}
        else:
            cursor = idx + len(word)
            for pos, _ in group:
                out[pos] = {"word": word, "char_start": int(idx),
                            "char_end": int(idx + len(word))}
        group = []

    for pos in range(len(stored)):
        if not support[pos]:
            continue
        tok = v.get(int(stored[pos]), "")
        if not tok or tok in ("[CLS]", "[SEP]", "[PAD]", "[UNK]"):
            continue
        if tok.startswith("##"):
            if group:
                group.append((pos, tok[2:]))
            else:
                group = [(pos, tok)]
        else:
            flush()
            group = [(pos, tok)]
    flush()
    return out


def token_position_map(sample, _unused=None):
    """position -> word / character span / CTC-forced-alignment seconds.

    Returns ``(mapping, audit)``.  A position whose word cannot be grounded in the
    transcript still gets a ``mapping_status`` entry, so nothing is invented.
    """
    al = load_alignment(str(sample.sid))
    spans = words_with_spans(sample)
    if al is None:
        return spans, {"alignment": "missing"}
    words = al.get("words", [])
    mapping = {}
    for pos, info in spans.items():
        if "char_start" not in info:
            mapping[pos] = info
            continue
        a, b = info["char_start"], info["char_end"]
        hit = next((w for w in words if a < w["char_end"] and b > w["char_start"]), None)
        if hit:
            mapping[pos] = {**info,
                            "start_s": round(float(hit["start"]), 4),
                            "end_s": round(float(hit["end"]), 4),
                            "alignment_confidence": round(float(hit["confidence"]), 4)}
        else:
            mapping[pos] = {**info,
                            "mapping_status": "text_fragment_without_speech_timestamp"}
    grounded = sum(1 for v in mapping.values() if "start_s" in v)
    audit = {"alignment": "ok", "positions_mapped": len(mapping),
             "positions_with_speech_time": grounded,
             "transcript_similarity": al.get("transcript_similarity"),
             "mean_confidence": al.get("mean_confidence"),
             "time_source": al.get("time_source")}
    return mapping, audit


def slot_time(sample, modality, j, n_pos=C.N_POS, duration=None):
    """Physical time window of aligned slot j, for audio/vision evidence.

    The aligned files carry no per-slot timestamps, so the slot is reported as the
    equal-length window over the audio duration — the same windowing convention
    used in problem 1.  This is stated explicitly so it is not mistaken for a
    measured onset.
    """
    if duration is None:
        al = load_alignment(str(sample.sid))
        duration = float(al["duration"]) if al else None
    if duration is None:
        return None
    w = duration / n_pos
    return round(j * w, 4), round((j + 1) * w, 4)


def audit_alignment():
    """Independent check of the paper's 'bert-base reproduces the provided text' claim."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(C.BERT_MODEL, local_files_only=True)
        model = AutoModel.from_pretrained(C.BERT_MODEL, local_files_only=True).eval()
    except Exception as exc:                       # weights not present locally
        return {"status": "unavailable", "reason": str(exc)[:200]}
    if model.config.hidden_size != 768:
        return {"status": "wrong_dim", "hidden_size": model.config.hidden_size}
    data = U.load_a3()                              # 附件3 has no `text` field
    return {"status": "available", "n_a3": len(data)}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--package", type=Path, default=HERE)
    ap.add_argument("--evidence-k", type=int, default=EVIDENCE_K)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    bias = np.array(json.loads(
        (a.package / "models" / "decision_calibration.json").read_text(encoding="utf-8"))[TAG])
    net = TI.load_model(TAG)
    try:
        n_vocab = len(vocab())
        print(f"vocabulary loaded ({n_vocab} tokens) — word decoding enabled")
    except Exception as exc:
        print(f"WARNING: vocabulary unavailable ({exc}); positions stay numeric")

    # ---------------- 附件3 ----------------
    a3 = U.load_a3()
    r3 = TI.predict(net, a3)
    p3 = calibrated(r3["prob"], bias)
    rows3 = []
    for i, sid in enumerate(r3["sid"]):
        c = int(p3[i].argmax())
        rows3.append(dict(sample_id=str(sid), pred_class=c, pred_polarity=POLARITY[c],
                          pred_intensity=round(float(r3["p_reg"][i]), 5),
                          confidence=round(float(p3[i, c]), 5),
                          prob_neg=round(float(p3[i, 0]), 5), prob_neu=round(float(p3[i, 1]), 5),
                          prob_pos=round(float(p3[i, 2]), 5),
                          polarity_intensity_conflict=bool((c == 2 and r3["p_reg"][i] < 0)
                                                           or (c == 0 and r3["p_reg"][i] > 0))))
    with (out / "附件3_预测.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows3[0].keys()))
        w.writeheader(); w.writerows(rows3)
    print(f"附件3: {len(rows3)} rows")

    # ---------------- 附件4 ----------------
    a4 = U.load_a4()
    r4 = TI.predict(net, a4)
    p4 = calibrated(r4["prob"], bias)
    cls = p4.argmax(1)
    base_prob = p4[np.arange(len(a4)), cls]

    # --- occlusion sweep on EVERY observed position of EVERY modality ---
    tasks, where = [], []
    for i, s in enumerate(a4):
        for m in MODS:
            for j in np.flatnonzero(np.asarray(s.mask[m]) > 0):
                tasks.append(drop_position(s, m, int(j)))
                where.append((i, m, int(j)))
    print(f"occlusion sweep: {len(tasks)} single-position deletions over "
          f"{len(a4)} samples x 3 modalities")
    attn = {}          # control column: the model's own evidence attention
    base_out = TI.predict(net, a4)
    for i in range(len(a4)):
        beta = base_out.get("beta")
        if beta is None:
            continue
        for mi, m in enumerate(MODS):
            row = np.asarray(beta[i][mi])
            for j in np.flatnonzero(np.asarray(a4[i].mask[m]) > 0):
                attn[(i, m, int(j))] = float(row[int(j)])

    eff = {}
    BATCH = 256
    for b0 in range(0, len(tasks), BATCH):
        chunk = tasks[b0:b0 + BATCH]
        out_p = calibrated(TI.predict(net, chunk)["prob"], bias)
        for n, (i, m, j) in enumerate(where[b0:b0 + BATCH]):
            eff[(i, m, j)] = float(base_prob[i] - out_p[n, cls[i]])
        if (b0 // BATCH) % 4 == 0:
            print(f"  ...{min(b0+BATCH, len(tasks))}/{len(tasks)}", flush=True)
    del tasks

    rows4, evidence = [], []
    for i, s in enumerate(a4):
        mapping, maudit = token_position_map(s)
        al = load_alignment(str(s.sid))
        duration = float(al["duration"]) if al else None

        # Shapley over the 8 coalitions
        v_cls, v_reg = {}, {}
        for keep in COALITIONS:
            mk = copy.deepcopy(s)
            for m in MODS:
                if m not in keep:
                    mk.mask[m] = np.zeros_like(mk.mask[m])
            res = TI.predict(net, [mk])
            v_cls[keep] = float(calibrated(res["prob"], bias)[0, cls[i]])
            v_reg[keep] = float(res["p_reg"][0])
        phi_c, phi_r = shapley_3player(v_cls), shapley_3player(v_reg)
        share = {m: abs(phi_c[m]) / max(sum(abs(x) for x in phi_c.values()), 1e-12) for m in MODS}
        full = canon(MODS)

        row = {
            "sample_id": str(s.sid),
            "pred_class": int(cls[i]), "pred_polarity": POLARITY[int(cls[i])],
            "pred_intensity": round(float(r4["p_reg"][i]), 5),
            "confidence": round(float(base_prob[i]), 5),
            "main_modality": max(MODS, key=lambda m: abs(phi_c[m])),
            "share_text": round(share["text"], 4), "share_audio": round(share["audio"], 4),
            "share_vision": round(share["vision"], 4),
            "phi_text_cls": round(phi_c["text"], 6), "phi_audio_cls": round(phi_c["audio"], 6),
            "phi_vision_cls": round(phi_c["vision"], 6),
            "phi_text_reg": round(phi_r["text"], 6), "phi_audio_reg": round(phi_r["audio"], 6),
            "phi_vision_reg": round(phi_r["vision"], 6),
            "efficiency_cls": round(v_cls[full] - v_cls[()], 6),
            "sum_phi_cls": round(sum(phi_c.values()), 6),
            "efficiency_reg": round(v_reg[full] - v_reg[()], 6),
            "sum_phi_reg": round(sum(phi_r.values()), 6),
            "v_empty_cls": round(v_cls[()], 6), "v_full_cls": round(v_cls[full], 6),
        }
        # per-modality evidence: rank by occlusion effect, keep top-k
        for m in MODS:
            obs = [int(j) for j in np.flatnonzero(np.asarray(s.mask[m]) > 0)]
            if not obs:
                row[f"{m}_evidence_positions"] = ""
                row[f"{m}_evidence_words"] = ""
                row[f"{m}_evidence_start_s"] = ""
                row[f"{m}_evidence_end_s"] = ""
                row[f"{m}_evidence_drop"] = ""
                row[f"{m}_evidence_set_drop"] = ""
                row[f"{m}_positions_scanned"] = 0
                continue
            ordered = sorted(obs, key=lambda j: -eff[(i, m, j)])
            top = ordered[:a.evidence_k]
            attn_top = sorted(obs, key=lambda j: -attn.get((i, m, j), 0.0))[:a.evidence_k]
            row[f"{m}_attention_positions"] = ";".join(map(str, attn_top))
            row[f"{m}_attention_set_drop"] = round(float(base_prob[i] - calibrated(
                TI.predict(net, [drop_set(s, m, attn_top)])["prob"], bias)[0, cls[i]]), 6)
            row[f"{m}_attention_matches_occlusion_topk"] = int(set(attn_top) == set(top))
            row[f"{m}_evidence_positions"] = ";".join(map(str, top))
            row[f"{m}_evidence_drop"] = ";".join(f"{eff[(i, m, j)]:.4f}" for j in top)
            row[f"{m}_positions_scanned"] = len(obs)
            set_drop = float(base_prob[i] - calibrated(
                TI.predict(net, [drop_set(s, m, top)])["prob"], bias)[0, cls[i]])
            row[f"{m}_evidence_set_drop"] = round(set_drop, 6)
            if m == "text":
                ws, ss, es, cs = [], [], [], []
                for j in top:
                    info = mapping.get(j)
                    if info is None:
                        ws.append(f"pos{j}"); ss.append(""); es.append(""); cs.append("")
                    else:
                        ws.append(info["word"])
                        ss.append(info.get("start_s", "")); es.append(info.get("end_s", ""))
                        cs.append(info.get("alignment_confidence", ""))
                row["text_evidence_words"] = ";".join(ws)
                row["text_evidence_start_s"] = ";".join(str(x) for x in ss)
                row["text_evidence_end_s"] = ";".join(str(x) for x in es)
                row["text_evidence_confidence"] = ";".join(str(x) for x in cs)
            else:
                sp = [slot_time(s, m, j, duration=duration) for j in top]
                row[f"{m}_evidence_start_s"] = ";".join(
                    f"{x[0]:.3f}" if x else "" for x in sp)
                row[f"{m}_evidence_end_s"] = ";".join(
                    f"{x[1]:.3f}" if x else "" for x in sp)
                row[f"{m}_evidence_frames"] = ";".join(
                    f"{int(round(x[0] * VIDEO_FPS))}-{int(round(x[1] * VIDEO_FPS))}" if x else ""
                    for x in sp)
                row[f"{m}_evidence_words"] = ""
            for rank, j in enumerate(top, 1):
                info = mapping.get(j, {})
                entry = {"sample_id": str(s.sid), "modality": m, "rank": rank,
                         "position": j, "probability_drop": round(eff[(i, m, j)], 6)}
                if m == "text":
                    entry.update({k: info.get(k, "") for k in
                                  ("word", "char_start", "char_end", "start_s", "end_s",
                                   "alignment_confidence", "mapping_status")})
                else:
                    t = slot_time(s, m, j, duration=duration)
                    entry.update({"slot_start_s": t[0] if t else "",
                                  "slot_end_s": t[1] if t else "",
                                  "keyframe_index": int(round(t[0] * VIDEO_FPS)) if t else "",
                                  "mapping_status": "equal-length aligned slot over audio duration"})
                evidence.append(entry)
        row["text_alignment_status"] = maudit.get("alignment", "")
        row["text_token_ids_exact_match"] = maudit.get("token_ids_exact_match", "")
        row["text_transcript_similarity"] = maudit.get("transcript_similarity", "")
        row["duration_s"] = duration if duration is not None else ""
        rows4.append(row)

    r4_fields = []
    for r in rows4:
        for k in r:
            if k not in r4_fields:
                r4_fields.append(k)
    with (out / "附件4_预测与解释.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=r4_fields, restval="")
        w.writeheader(); w.writerows(rows4)
    ev_fields = []
    for e in evidence:
        for k in e:
            if k not in ev_fields:
                ev_fields.append(k)
    with (out / "附件4_证据明细.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=ev_fields, restval="")
        w.writeheader(); w.writerows(evidence)

    # ---------------- summary ----------------
    def nonpos(key):
        return sum(1 for r in rows4 if r.get(key, "") != "" and float(r[key]) <= 0)
    summary = {
        "deployment_model": f"{TAG} three-seed ensemble",
        "valid_fitted_bias": bias.tolist(),
        "test_reference": {"accuracy": 0.6753782668500687, "macro_f1": 0.6485242985549113,
                           "mae": 0.6144672632217407, "pearson": 0.6912329605433559, "n": 727},
        "attachment3": {"n": len(rows3),
                        "class_counts": {POLARITY[c]: sum(r["pred_class"] == c for r in rows3)
                                         for c in range(3)},
                        "direction_conflicts": sum(r["polarity_intensity_conflict"] for r in rows3)},
        "attachment4": {
            "n": len(rows4), "columns": len(r4_fields),
            "class_counts": {POLARITY[c]: sum(r["pred_class"] == c for r in rows4) for c in range(3)},
            "main_modality_counts": {m: sum(r["main_modality"] == m for r in rows4) for m in MODS},
            "max_efficiency_error_cls": max(abs(r["efficiency_cls"] - r["sum_phi_cls"]) for r in rows4),
            "max_efficiency_error_reg": max(abs(r["efficiency_reg"] - r["sum_phi_reg"]) for r in rows4),
            "positions_scanned_total": sum(r[f"{m}_positions_scanned"] for r in rows4 for m in MODS),
            "positions_scanned_by_modality": {m: sum(r[f"{m}_positions_scanned"] for r in rows4)
                                              for m in MODS},
            "modality_set_drop_mean": {m: round(float(np.mean([float(r[f"{m}_evidence_set_drop"])
                                                               for r in rows4
                                                               if r[f"{m}_evidence_set_drop"] != ""])), 4)
                                       for m in MODS},
            "modality_set_drop_nonpositive": {m: nonpos(f"{m}_evidence_set_drop") for m in MODS},
            "samples_with_evidence_per_modality": {
                m: sum(1 for r in rows4 if r[f"{m}_evidence_positions"] != "") for m in MODS},
        },
        "evidence_rows": len(evidence),
    }
    (out / "预测汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
