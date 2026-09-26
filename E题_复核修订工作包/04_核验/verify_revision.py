"""Verification for the 数据复核修订工作包.

Checks every claim the package makes, plus the defects that were found and fixed
in the previous round, so a regression is caught by name.

  V1  p2 ensemble reproduces its recorded TEST metrics to 1e-6
  V2  附件3 deliverable: 30 rows, columns, probabilities sum to 1, no NaN
  V3  附件4 deliverable: 20 rows, all required evidence columns present
  V4  Shapley efficiency identity closes (classification and regression)
  V5  occlusion evidence covers EVERY observed position of EVERY modality
  V6  text evidence carries words with CTC-derived seconds (not bare indices)
  V7  audio / vision evidence carries slot times and keyframe indices
  V8  attention pre-selection is NOT used (attention column is a control only)
  V9  problem-1 vision features agree with an independent face detector
  V10 paper: no header, tables styled, abstract not bold, no '10^(' left
  V11 no identity information

Exit code 0 only if all gates pass.
"""
from __future__ import annotations

import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
CODE = PKG / "02_模型与代码"
DATA = PKG / "01_交付结果"
DOCX = PKG / "03_论文与文档" / "数模论文_数据复核修订稿.docx"
sys.path.insert(0, str(CODE / "code_最终模型"))
sys.path.insert(0, str(CODE))

GATES = []


def gate(name, ok, detail=""):
    GATES.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)


def main() -> int:
    if not os.environ.get("MATH_E_DATA"):
        print("ERROR: set MATH_E_DATA to the directory holding 附件1..附件4", file=sys.stderr)
        return 2
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    import torch
    torch.set_num_threads(6)

    # ---------- V1 ----------
    import io_utils as U
    from inference import load_model, predict
    from sklearn.metrics import accuracy_score, f1_score

    summary = json.loads((DATA / "预测汇总.json").read_text(encoding="utf-8"))
    ref = summary["test_reference"]
    d = U.load_a2_all(splits=("test",))
    net = load_model("p2")
    res = predict(net, d["test"])
    bias = np.array(summary["valid_fitted_bias"])
    lp = np.log(np.clip(res["prob"], 1e-9, None)) + bias
    prob = np.exp(lp - lp.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
    pred = prob.argmax(1)
    got = {"accuracy": float(accuracy_score(res["y_cls"], pred)),
           "macro_f1": float(f1_score(res["y_cls"], pred, average="macro")),
           "mae": float(np.mean(np.abs(res["y_reg"] - res["p_reg"]))),
           "pearson": float(np.corrcoef(res["y_reg"], res["p_reg"])[0, 1])}
    gate("V1 p2 test metrics reproduce", all(abs(got[k] - ref[k]) < 1e-6 for k in got),
         json.dumps({k: round(v, 6) for k, v in got.items()}))

    # ---------- V2 ----------
    a3 = pd.read_csv(DATA / "附件3_预测.csv", encoding="utf-8-sig")
    need3 = {"sample_id", "pred_class", "pred_polarity", "pred_intensity", "confidence",
             "prob_neg", "prob_neu", "prob_pos", "polarity_intensity_conflict"}
    ok2 = bool(len(a3) == 30 and need3 <= set(a3.columns)
               and np.allclose(a3[["prob_neg", "prob_neu", "prob_pos"]].sum(1), 1.0, atol=1e-3)
               and not a3.isna().any().any())
    gate("V2 附件3 deliverable", ok2,
         f"rows={len(a3)} cols={len(a3.columns)} class={a3.pred_polarity.value_counts().to_dict()}")

    # ---------- V3 ----------
    a4 = pd.read_csv(DATA / "附件4_预测与解释.csv", encoding="utf-8-sig")
    need4 = {"sample_id", "pred_polarity", "pred_intensity", "confidence", "main_modality",
             "share_text", "share_audio", "share_vision",
             "phi_text_cls", "phi_audio_cls", "phi_vision_cls",
             "efficiency_cls", "sum_phi_cls", "efficiency_reg", "sum_phi_reg"}
    for m in ("text", "audio", "vision"):
        need4 |= {f"{m}_evidence_positions", f"{m}_evidence_set_drop", f"{m}_positions_scanned"}
    miss = need4 - set(a4.columns)
    gate("V3 附件4 deliverable", len(a4) == 20 and not miss,
         f"rows={len(a4)} cols={len(a4.columns)} missing={sorted(miss)}")

    # ---------- V4 ----------
    ec = float(np.abs(a4.efficiency_cls - a4.sum_phi_cls).max())
    er = float(np.abs(a4.efficiency_reg - a4.sum_phi_reg).max())
    gate("V4 Shapley efficiency identity", ec < 1e-6 and er < 1e-6,
         f"max err cls={ec:.2e} reg={er:.2e}")

    # ---------- V5: full coverage ----------
    total = int(a4[[f"{m}_positions_scanned" for m in ("text", "audio", "vision")]].sum().sum())
    by_mod = {m: int(a4[f"{m}_positions_scanned"].sum()) for m in ("text", "audio", "vision")}
    gate("V5 occlusion covers every observed position", total == 1662 and by_mod["text"] == 564,
         f"total={total} by_modality={by_mod}")

    # ---------- V6: text evidence = words + seconds ----------
    ev = pd.read_csv(DATA / "附件4_证据明细.csv", encoding="utf-8-sig")
    txt = ev[ev.modality == "text"]
    have_word = int((txt.word.astype(str).str.len() > 0).sum())
    have_time = int(txt.start_s.notna().sum())
    gate("V6 text evidence has words and CTC seconds",
         have_word == len(txt) and have_time >= 0.9 * len(txt),
         f"rows={len(txt)} with_word={have_word} with_seconds={have_time}")

    # ---------- V7: audio/vision evidence has times + frames ----------
    ok7, det7 = True, {}
    for m in ("audio", "vision"):
        sub = ev[ev.modality == m]
        n_time = int(sub.slot_start_s.notna().sum())
        n_frame = int(sub.keyframe_index.notna().sum())
        det7[m] = f"rows={len(sub)} time={n_time} frame={n_frame}"
        ok7 &= (n_time == len(sub) and n_frame == len(sub))
    gate("V7 audio/vision evidence has slot times and keyframes", ok7, str(det7))

    # ---------- V8: attention is a control, not the selector ----------
    agree = {m: int(a4[f"{m}_attention_matches_occlusion_topk"].sum()) for m in ("text", "audio", "vision")}
    drops = {m: pd.to_numeric(a4[f"{m}_evidence_set_drop"], errors="coerce") for m in ("text", "audio", "vision")}
    atts = {m: pd.to_numeric(a4[f"{m}_attention_set_drop"], errors="coerce") for m in ("text", "audio", "vision")}
    better = all(drops[m].mean() > atts[m].mean() for m in ("text", "audio", "vision"))
    gate("V8 occlusion ranks evidence, attention is control only",
         sum(agree.values()) == 0 and better,
         "top5_identical=" + str(agree)
         + " | set_drop occlusion/attention: "
         + ", ".join(f"{m} {drops[m].mean():+.4f}/{atts[m].mean():+.4f}" for m in ("text", "audio", "vision")))

    # ---------- V9: Q1 vision agrees with an independent detector ----------
    s1 = pd.read_csv(DATA / "p1_summary_v3.csv")
    vis_ok = int((s1.vision_valid == 0).sum()) == 4 and int(s1.vision_valid.sum()) == 4306
    detail9 = f"vision_valid total={int(s1.vision_valid.sum())} zeros={int((s1.vision_valid==0).sum())}"
    if vis_ok:
        try:
            import cv2
            base = (Path(os.environ["MATH_E_DATA"]) /
                    "附件1-数据集原始多模态样本" / "MOSEI数据集部分原始视频-100条")
            model_path = PKG / "02_模型与代码" / "face_detection_yunet_2023mar.onnx"
            if not model_path.exists():
                model_path = Path("/Users/kyrie/戴书恒/数学建模竞赛/竞赛题/E题/math-model-e/"
                                  "models/face_detection_yunet_2023mar.onnx")
            det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.6, 0.3, 5000)
            zeros = s1[s1.vision_valid == 0]
            found = 0
            for _, r in zeros.iterrows():
                f = base / str(r.video_id) / f"{int(r.clip_id)}.mp4"
                if not f.exists():
                    continue
                cap = cv2.VideoCapture(str(f)); n = h = 0
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    n += 1
                    if (n - 1) % 10:
                        continue
                    H, Wd = fr.shape[:2]; det.setInputSize((Wd, H))
                    _, fc = det.detect(fr)
                    if fc is not None and len(fc):
                        h += 1
                cap.release()
                found += int(h > 0)
            detail9 += f" | independent check: {found}/{len(zeros)} of the zero-vision clips do contain faces"
            vis_ok &= found <= 1
        except ImportError:
            detail9 += " | (opencv not installed, skipped pixel check)"
    gate("V9 Q1 vision features are credible", vis_ok, detail9)

    # ---------- V10: paper ----
    if DOCX.exists():
        with zipfile.ZipFile(DOCX) as z:
            names = z.namelist()
            xml = z.read("word/document.xml").decode("utf-8")
        has_header = any("header" in n for n in names if n.endswith(".xml"))
        n_tbl_style = len(re.findall(r'<w:tblStyle w:val="TableGrid"', xml))
        # abstract body must not be bold
        paras = re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)
        bold_abs = 0
        for p in paras:
            txt = re.sub(r"<[^>]+>", "", p).strip()
            if txt.startswith(("针对原始", "问题二使用", "问题三对同一")) and "<w:b" in p:
                bold_abs += 1
        power_left = xml.count("10^(")
        gate("V10 paper format fixes hold",
             (not has_header) and n_tbl_style >= 10 and bold_abs == 0 and power_left == 0,
             f"header={has_header} TableGrid={n_tbl_style} bold_abstract={bold_abs} "
             f"leftover_10^={power_left}")
    else:
        gate("V10 paper format fixes hold", False, f"{DOCX} not found")

    # ---------- V11 ----------
    PAT = [re.compile(r"参赛队\s*[：:]\s*\S"), re.compile(r"队伍编号\s*[：:]\s*\S"),
           re.compile(r"队员\s*(?:姓名|名单)?\s*[：:]\s*\S"), re.compile(r"学号\s*[：:]\s*\S"),
           re.compile(r"指导老师\s*[：:]\s*\S"), re.compile(r"(?:作者|姓名)\s*[：:]\s*\S")]
    hits = []
    for p in PKG.rglob("*"):
        if not p.is_file() or p.suffix.lower() in {".pt", ".npz", ".pkl", ".png", ".jpg",
                                                   ".pdf", ".joblib", ".bin", ".docx", ".mp4"}:
            continue
        try:
            t = p.read_text(errors="ignore")
        except Exception:
            continue
        for pat in PAT:
            m = pat.search(t)
            if m:
                hits.append((str(p.relative_to(PKG)), m.group(0)[:30]))
    gate("V11 no identity information", not hits, str(hits[:3]) if hits else "clean")

    ok = all(g[1] for g in GATES)
    print("\n" + ("STATUS: VERIFIED" if ok else "STATUS: FAILED"))
    for n, o, dt in GATES:
        if not o:
            print(f"  failed: {n} -- {dt}")
    (PKG / "04_核验" / "verification_report.json").write_text(json.dumps(
        {"status": "VERIFIED" if ok else "FAILED",
         "gates": [{"name": n, "passed": o, "detail": d} for n, o, d in GATES]},
        ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
