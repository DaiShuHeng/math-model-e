"""Generate the paper's key figures from frozen result artifacts.

fig_rate_curve.png  — champion vs teammate-p2 Acc/mF1 vs damage rate (valid+test)
fig_shapley.png     — 附件4 exact Shapley φ (predicted class) stacked bars
fig_q1_vision.png   — Q1 vision coverage & probe before/after YuNet

Run: cd solution && python -m src.make_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess

SOL = Path(__file__).resolve().parents[1]
FIG = SOL / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# register the system CJK fallback font (matplotlib's cache misses it)
_droid = subprocess.run(["fc-match", "-f", "%{file}", "Droid Sans Fallback"],
                        capture_output=True, text=True).stdout.strip()
if _droid:
    fm.fontManager.addfont(_droid)
# explicit concrete-name list (NOT the "sans-serif" alias): mpl only builds the
# per-glyph fallback chain from font.family — via the alias it resolves to a
# single file, and Droid has zero ASCII glyphs (boxed axis numbers).
matplotlib.rcParams["font.family"] = ["Droid Sans Fallback", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def rate_curve_fig():
    # champion per-seed rate curves (valid)
    rc = json.loads((SOL / "logs" / "rate_curve_base_cw.json").read_text())
    rates = next(iter(rc.values()))["rates"]
    champ = {k: [] for k in ("Acc", "MacroF1")}
    for p, curve in rc.items():
        for pt, m in zip(curve["points"], rates):
            pass
    # average across the three checkpoints (they share the rates list)
    for i, r in enumerate(rates):
        champ["Acc"].append(np.mean([c["points"][i]["Accuracy"] for c in rc.values()]))
        champ["MacroF1"].append(np.mean([c["points"][i]["MacroF1"] for c in rc.values()]))

    # teammate points from head-to-head valid (clean, pool, r20, r40)
    h2h = json.loads((SOL / "logs" / "head2head.json").read_text())
    tm_x = [0.0, None, 0.2, 0.4]           # pool has no single rate; drawn at ~0.14 (empirical mean)
    tm_rates = [0.0, 0.14, 0.2, 0.4]
    tm_acc = [h2h[c]["theirs"]["Accuracy"] for c in ("clean", "pool", "r20", "r40")]
    tm_f1 = [h2h[c]["theirs"]["MacroF1"] for c in ("clean", "pool", "r20", "r40")]
    ch_acc_h = [h2h[c]["champ"]["Accuracy"] for c in ("clean", "pool", "r20", "r40")]
    ch_f1_h = [h2h[c]["champ"]["MacroF1"] for c in ("clean", "pool", "r20", "r40")]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]
    ax.plot(rates, champ["Acc"], "o-", color="#1f6fb4", label="本文模型（逐率评估）")
    ax.plot(tm_rates, tm_acc, "s--", color="#c2504d", label="通用增强基线（p2）")
    ax.scatter([0.14], [h2h["pool"]["champ"]["Accuracy"]], marker="*", s=120, color="#1f6fb4", zorder=5)
    ax.annotate("率池协议", (0.14, h2h["pool"]["champ"]["Accuracy"]), textcoords="offset points",
                xytext=(6, 7), fontsize=8, color="#1f6fb4")
    ax.annotate("率池协议", (0.14, h2h["pool"]["theirs"]["Accuracy"]), textcoords="offset points",
                xytext=(6, -12), fontsize=8, color="#c2504d")
    ax.set(xlabel="词级损坏率 r", ylabel="Accuracy (valid)", title="(a) 分类准确率 vs 损坏率")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.3)

    ax = axes[1]
    ax.plot(rates, champ["MacroF1"], "o-", color="#1f6fb4", label="本文模型")
    ax.plot(tm_rates, tm_f1, "s--", color="#c2504d", label="通用增强基线（p2）")
    ax.scatter([0.14], [h2h["pool"]["champ"]["MacroF1"]], marker="*", s=120, color="#1f6fb4", zorder=5)
    ax.set(xlabel="词级损坏率 r", ylabel="Macro F1 (valid)", title="(b) 宏平均 F1 vs 损坏率")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "fig_rate_curve.png", dpi=200)
    plt.close(fig)


def shapley_fig():
    df = pd.read_csv(SOL / "results" / "附件4_shapley.csv", dtype={0: str})
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.bar(x, df.phi_text_cls, color="#1f6fb4", label="文本 φ_T")
    ax.bar(x, df.phi_audio_cls, bottom=df.phi_text_cls, color="#5799ad", label="音频 φ_A")
    ax.bar(x, df.phi_vision_cls, bottom=df.phi_text_cls + df.phi_audio_cls, color="#df9c4c", label="视觉 φ_V")
    ax.set_xticks(x); ax.set_xticklabels(df.sample_id, fontsize=8)
    ax.axhline(0, color="k", lw=.8)
    # annotate predicted polarity
    for i, (sid, pol) in enumerate(zip(df.sample_id, df.pred_polarity)):
        ax.text(i, ax.get_ylim()[1] * 0.97, pol[0], ha="center", fontsize=7, color="dimgray")
    ax.set(xlabel="附件4 样本", ylabel="对预测类概率的 Shapley 值",
           title="附件4 各样本三模态精确 Shapley 归因（Σφ = v(TAV) − v(∅)，逐样本精确成立）")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(FIG / "fig_shapley.png", dpi=200)
    plt.close(fig)


def q1_vision_fig():
    v2 = pd.read_csv("/home/daishuheng/math_competition/math_model_e_optimized/data/p1_summary_v2.csv")
    v3 = pd.read_csv(SOL / "data" / "p1_summary_v3.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax = axes[0]
    ax.hist(v2.face_detection_rate.clip(0, 1), bins=20, alpha=.65, color="#c2504d", label=f"Haar (均值 {v2.face_detection_rate.mean():.2f})")
    ax.hist(v3.face_detection_rate.clip(0, 1), bins=20, alpha=.65, color="#1f6fb4", label=f"YuNet (均值 {v3.face_detection_rate.mean():.2f})")
    ax.set(xlabel="人脸检出率", ylabel="视频数", title="(a) 100 个附件1视频的人脸检出率分布")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    v2z = int((v2.vision_valid == 0).sum()); v3z = int((v3.vision_valid == 0).sum())
    bars = ax.bar(["视觉全无效视频", "平均有效视觉时间窗"],
                  [v2z / 100, v2.vision_valid.mean() / 50], width=.45, color="#c2504d")
    bars2 = ax.bar(["视觉全无效视频", "平均有效视觉时间窗"],
                   [v3z / 100, v3.vision_valid.mean() / 50], width=.45,
                   bottom=[0, 0], align="edge", color="#1f6fb4")
    ax.set(ylabel="比例", title="(b) 视觉通道覆盖：Haar vs YuNet")
    ax.set_xticks([0.11, 0.61]); ax.set_xticklabels(["视觉全无效视频", "平均有效视觉时间窗"])
    ax.text(0.11 - .11, v2z / 100 + .01, f"{v2z}/100 → {v3z}/100", ha="center", fontsize=9)
    ax.text(0.61 + .23, v3.vision_valid.mean() / 50 + .01,
            f"{v2.vision_valid.mean():.1f}/50 → {v3.vision_valid.mean():.1f}/50", ha="center", fontsize=9)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#c2504d"), plt.Rectangle((0, 0), 1, 1, color="#1f6fb4")],
              labels=["Haar (v2)", "YuNet (v3)"], frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_q1_vision.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    rate_curve_fig(); print("fig_rate_curve.png")
    shapley_fig(); print("fig_shapley.png")
    q1_vision_fig(); print("fig_q1_vision.png")
