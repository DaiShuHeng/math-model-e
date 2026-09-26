"""Build checked, human-readable prediction tables from packaged outputs.

This does not alter model predictions. Disagreements between independent
classification and regression heads are retained and visibly flagged.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def check_ids(frame, expected):
    ids = frame["sample_id"].astype(str)
    assert len(frame) == expected and ids.nunique() == expected
    assert set(ids) == {f"{i:02d}" for i in range(1, expected + 1)}


def polarity_conflict(name, intensity):
    return bool((name == "Negative" and intensity > 0) or
                (name == "Positive" and intensity < 0))


def main():
    p2 = pd.read_csv(R / "附件3_优化预测.csv", dtype={"sample_id": str})
    p2["sample_id"] = p2["sample_id"].str.extract(r"(\d{2})$")[0]
    check_ids(p2, 30)
    p2["极性强度方向冲突"] = [polarity_conflict(c, y) for c, y in
                             zip(p2.pred_polarity, p2.pred_intensity)]
    p2 = p2.rename(columns={"pred_polarity": "预测极性", "pred_intensity": "预测情感强度",
                            "confidence": "分类置信度"})
    p2[["sample_id", "预测极性", "预测情感强度", "分类置信度", "极性强度方向冲突"]].to_csv(
        R / "附件3_最终预测.csv", index=False, encoding="utf-8-sig")

    base = pd.read_csv(R / "附件4_优化预测与解释.csv", dtype={"sample_id": str})
    shap = pd.read_csv(R / "附件4_shapley.csv", dtype={"sample_id": str})
    check_ids(base, 20)
    check_ids(shap, 20)
    merged = base.merge(shap, on="sample_id", suffixes=("", "_shap"), validate="one_to_one")
    assert (merged.pred_polarity == merged.pred_polarity_shap).all()
    assert np.max(np.abs(merged.pred_intensity - merged.pred_intensity_shap)) < 0.0001
    assert np.max(np.abs(merged.sum_phi_cls - merged.efficiency_cls)) < 1e-4
    merged["极性强度方向冲突"] = [polarity_conflict(c, y) for c, y in
                                 zip(merged.pred_polarity, merged.pred_intensity)]
    merged["视觉输入状态"] = np.where(merged.sample_id == "13", "对齐版视觉全零", "有非零视觉特征")
    audit = pd.read_csv(R / "evidence_mapping_audit.csv", dtype={"sample_id": str})
    merged = merged.merge(audit[["sample_id", "alignment_similarity", "alignment_confidence",
                                 "token_ids_exact_match"]], on="sample_id", validate="one_to_one")
    evidence = pd.read_csv(R / "evidence_time_mapping.csv", dtype={"sample_id": str})
    e = evidence[(evidence.modality == "text") & (evidence["rank"] == 1)]
    e = e[["sample_id", "word", "start_s", "end_s", "alignment_confidence"]].rename(
        columns={"word": "首位文本证据词", "start_s": "估计起点秒", "end_s": "估计终点秒",
                 "alignment_confidence": "证据对齐置信度"})
    merged = merged.merge(e, on="sample_id", how="left", validate="one_to_one")
    cols = ["sample_id", "pred_polarity", "pred_intensity", "confidence", "main_modality_shapley",
            "share_text", "share_audio", "share_vision", "phi_text_cls", "phi_audio_cls",
            "phi_vision_cls", "首位文本证据词", "估计起点秒", "估计终点秒", "证据对齐置信度",
            "alignment_similarity", "alignment_confidence", "token_ids_exact_match", "视觉输入状态", "极性强度方向冲突",
            "text_evidence", "audio_evidence", "vision_evidence"]
    merged[cols].rename(columns={"pred_polarity": "预测极性", "pred_intensity": "预测情感强度",
                                 "confidence": "分类置信度", "main_modality_shapley": "主要参考模态",
                                 "share_text": "文本贡献份额", "share_audio": "语音贡献份额",
                                 "share_vision": "视觉贡献份额"}).to_csv(
        R / "附件4_最终预测与解释.csv", index=False, encoding="utf-8-sig")
    print("附件3：", len(p2), "条，方向冲突", int(p2["极性强度方向冲突"].sum()))
    print("附件4：", len(merged), "条，方向冲突", int(merged["极性强度方向冲突"].sum()))


if __name__ == "__main__":
    main()
