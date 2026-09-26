"""Evidence-backed extended chapters and audit appendices for the final paper."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def tab(headers, rows):
    def clean(v):
        return str(v).replace("|", "\\|").replace("$", "\\$").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(clean, row)) + " |" for row in rows])


def f(x, n=3):
    return "—" if pd.isna(x) else f"{float(x):.{n}f}"


def make_figures(root, p1, align, robust, p2details, shap):
    figdir = root / "results" / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].boxplot([p1[c].to_numpy() for c in ["text_valid", "audio_valid", "vision_valid"]],
                  tick_labels=["文本", "语音", "视觉"])
    ax[0].set_ylabel("有效时间窗 / 50")
    ax[0].set_ylim(-2, 53)
    ax[1].hist(p1["vision_valid"], bins=np.arange(0, 55, 5), color="#5479a3", edgecolor="white")
    ax[1].set_xlabel("视觉有效窗数")
    ax[1].set_ylabel("视频片段数")
    fig.tight_layout()
    fig.savefig(figdir / "paper_completeness.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for ds, grp in align.groupby("dataset"):
        ax[0].hist(grp["confidence"], bins=np.linspace(0, 1, 11), alpha=.6, label=ds)
        ax[1].scatter(grp["confidence"], grp["transcript_similarity"], label=ds, alpha=.65, s=23)
    ax[0].set_xlabel("CTC 路径置信指标")
    ax[0].set_ylabel("样本数")
    ax[1].set_xlabel("CTC 路径置信指标")
    ax[1].set_ylabel("转写相似度")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(figdir / "paper_alignment_audit.png", dpi=180)
    plt.close(fig)

    rr = robust[robust.family.eq("rate")].copy()
    modalities = ["text", "audio", "vision", "ta", "tv", "av", "tav"]
    rates = sorted(rr.rate.unique())
    matrix = np.array([[rr.loc[rr.modality.eq(m) & rr.rate.eq(rate), "f1_macro"].iloc[0]
                        for rate in rates] for m in modalities])
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=.5, vmax=.68)
    ax.set_yticks(np.arange(len(modalities)), modalities)
    ax.set_xticks(np.arange(len(rates)), [f"{x:.1f}" for x in rates])
    ax.set_xlabel("新增置零比例")
    ax.set_ylabel("受损模态组合")
    for i in range(len(modalities)):
        for j in range(len(rates)):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="验证集宏平均 F1")
    fig.tight_layout()
    fig.savefig(figdir / "paper_robustness_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
    ax[0].scatter(p2details.true_intensity, p2details.pred_intensity, s=13, alpha=.38)
    ax[0].plot([-3, 3], [-3, 3], "k--", lw=1)
    ax[0].set(xlabel="真实强度", ylabel="预测强度", xlim=(-3.1, 3.1), ylim=(-3.1, 3.1))
    for cls, name in [(0, "负向"), (1, "中性"), (2, "正向")]:
        ax[1].hist(p2details.loc[p2details.true_class.eq(cls), "absolute_error"], bins=np.linspace(0, 4, 25),
                   histtype="step", lw=1.8, label=name)
    ax[1].set(xlabel="绝对误差", ylabel="样本数")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(figdir / "paper_validation_errors.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.3))
    x = np.arange(len(shap))
    for m, offset, color in [("text", -.25, "#386cb0"), ("audio", 0, "#f28e2b"), ("vision", .25, "#59a14f")]:
        ax.bar(x + offset, shap[f"phi_{m}_cls"], width=.25, label=m, color=color)
    ax.axhline(0, color="black", lw=.7)
    ax.set_xticks(x, shap.sample_id.astype(str), rotation=90)
    ax.set(xlabel="附件四样本编号", ylabel="预测类别概率的有符号 Shapley 值")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(figdir / "paper_shapley_signed.png", dpi=180)
    plt.close(fig)


def extended_main(root):
    r = root / "results"
    p1 = pd.read_csv(root / "data" / "p1_summary_v2.csv")
    align = pd.read_csv(r / "alignment_quality.csv", dtype={"sample_id": str})
    pad = pd.read_csv(r / "padding_and_observation_audit.csv", dtype={"id": str})
    robust = pd.read_csv(r / "robustness_valid.csv")
    mask_audit = pd.read_csv(r / "robustness_mask_audit.csv")
    p2 = pd.read_csv(r / "p2_validation_details.csv", dtype={"sample_id": str})
    p3 = pd.read_csv(r / "p3_validation_details.csv", dtype={"sample_id": str})
    q3 = pd.read_csv(r / "附件4_最终预测与解释.csv", dtype={"sample_id": str}, encoding="utf-8-sig")
    shap = pd.read_csv(r / "附件4_shapley.csv", dtype={"sample_id": str})
    runs = json.loads((r / "validation_training_runs.json").read_text(encoding="utf-8"))
    probe = json.loads((r / "text_interface_probe.json").read_text(encoding="utf-8"))
    make_figures(root, p1, align, robust, p2, shap)
    out = []
    def add(s): out.append(s.strip() + "\n")

    add("""## 5 数据接口、对齐和观测审计

### 5.1 题目数据与三类位置

附件二与专项附件都以长度 50 的数组存储序列，但“存储位置”“内容位置”“实际观测位置”并不相同。存储位置包括 [CLS]、[SEP] 与补零；内容位置是题目提供的有效文本词元位置；实际观测位置还要求所对应模态数值有限且不是全零。若把补零直接当成缺失，就会把短语句误判为严重损坏；若把模态特征全零直接当成补零，则会把真正丢失的音视频片段排除在分析之外。因此数据读取先根据文本词元构造内容支持掩码，再逐模态检查观测掩码。训练统计量、注意力掩码和鲁棒性实验均使用同一套定义。

附件三的一条样本可能只剩少量文本词元，但这不说明被删除的原句有多长。以“当前可见文本支持序列”为分母可以描述其现有音视频缺口，却不能报告“原始文本缺失率”。附件四是另一类专项任务，虽然有视频和特征，也没有情感标签，不可用于补充训练集。审计表覆盖附件三 30 条和附件四 20 条、每条三模态，共 150 行；正文附录保留逐条数值。

下表的内容位与观测位为每条样本的平均数，均以 50 位存储轴为上限。附件三语音和视觉的平均观测位较少，且是当前可见内容上的统计；附件四第 13 条的视觉全零提示某些模态根本没有可解释的实体证据。""")
    grouped = pad.groupby(["dataset", "modality"], sort=True)[["content_positions", "observed_positions", "zero_within_content"]].mean().reset_index()
    add(tab(["专项附件", "模态", "平均内容位", "平均观测位", "内容内平均全零位"],
            [[x.dataset, x.modality, f(x.content_positions, 2), f(x.observed_positions, 2),
              f(x.zero_within_content, 2)] for x in grouped.itertuples()]))
    add("""### 5.2 文本连续接口校验

附件三提供 `text_bert`，必须用与附件二 `text` 一致的冻结文本编码接口转成连续特征。我们在训练与验证集中共选 8 条样本，把词元编号直接送入冻结 BERT，并与题目提供的 `text` 在有效位置上逐维比较；另对“沿词元轴重采样再插值”的做法做诊断。直接编码 RMSE 的量级约为 $10^{-6}$，插值后的误差高出多个数量级。这一核查仅验证选取的 8 条样本接口，不能证明所有文本都绝对正确；实际推理仍保留词元逐位核对和异常记录。""")
    add(tab(["划分", "行号", "有效位置数", "直接编码 RMSE", "插值 RMSE", "直接相关", "插值相关"],
            [[x["split"], x["i"], x["n"], f(x["direct_rmse"], 8), f(x["resampled_rmse"], 4),
              f(x["direct_corr"], 6), f(x["resample_corr"], 4)] for x in probe]))

    add("""### 5.3 物理时间与位置时间的边界

问题一的 50 个窗覆盖视频真实时间，边界由解码时长计算；问题二和三的 50 个位置是官方 aligned 特征的词元存储轴。这两套横轴不具有简单的一一对应关系。将第 $j$ 个词元误解释为视频第 $j$ 个等长时间窗，会产生系统性的证据定位错误。本研究仅用 CTC 对齐后的词起止秒数标注证据，而不从词元序号线性推算时间。对齐失败时保留缺失时间标志，词元到转写词的编号不吻合时标记人工复核。

对齐质量是自动算法产生的路径置信指标和转写相似度，并非人工核实的“时间准确率”。100 条问题一片段中 24 条被标为优先回查；20 条附件四片段中 3 条被标为优先回查。绝对阈值只用于排序人工复核工作，不参与情感模型的参数选择。下图显示两项质量指标的分布与关系。""")
    add("![自动对齐质量审计](../results/figures/paper_alignment_audit.png)")
    add(tab(["数据部分", "片段数", "路径指标均值", "相似度均值", "优先回查数"],
            [[ds, len(g), f(g.confidence.mean(), 3), f(g.transcript_similarity.mean(), 3),
              int(g.needs_review.sum())] for ds, g in align.groupby("dataset")]))

    add("""### 5.4 问题一的视觉可用性

三模态数组都生成成功，不等于三模态在每条视频中都含有可靠观测。100 条片段的语音物理窗均有样本，文本平均 37.3 个有效窗，视觉平均 17.6 个有效窗；39 条片段的视觉有效窗为零。视觉检出受脸部方向、画面裁剪、亮度和检测器能力影响。模型在这些样本上的视觉通道应视为不可用，而不应把零特征当成“面部没有情绪”。问题一融合交叉验证的宏平均 F1 为 0.4492；样本量有限且弱视觉较多，不能把融合表现归因于精确表情识别。""")
    add("![问题一观测位分布](../results/figures/paper_completeness.png)")
    byvideo = p1.groupby("video_id", sort=True).agg(clips=("sample_id", "size"), duration=("duration_s", "mean"),
                text=("text_valid", "mean"), visual=("vision_valid", "mean"), face=("face_detection_rate", "mean")).reset_index()
    add("按原视频聚合后仍可看到视觉可用性差异。下表只用于描述数据覆盖情况，不能作为原视频的情感标签或独立测试成绩。")
    add(tab(["原视频 ID", "片段数", "平均时长/s", "文本窗均值", "视觉窗均值", "人脸检出率均值"],
            [[x.video_id, x.clips, f(x.duration, 2), f(x.text, 1), f(x.visual, 1), f(x.face, 2)]
             for x in byvideo.itertuples()]))

    add("""## 6 训练选择、鲁棒性与误差结构

### 6.1 特征与模型复杂度

对照模型使用已观测位置的模态均值表示，训练集上拟合标准化，逻辑回归和 Ridge 的超参数只用验证集选择。其作用是给复杂网络一个简单、可复现的参照，不能预先假设 Transformer 必然占优。文本均值模型在验证集的宏平均 F1 为 0.5905，三模态均值模型为 0.5757；这提醒我们在当前特征质量下，融合未必自动改善所有类别。主模型包含 462546 个可训练参数，三个模态分别编码再门控融合；三个种子集成带来更高推理成本。""")
    linear = pd.read_csv(r / "linear_baselines_valid.csv")
    add(tab(["输入", "分类 C", "回归 α", "Accuracy", "宏平均 F1", "MAE", "Pearson"],
            [[x.model, x.C, x.ridge_alpha, f(x.acc,4), f(x.f1_macro,4), f(x.mae,4), f(x.pearson,4)]
             for x in linear.itertuples()]))
    add("""### 6.2 训练轮次与随机种子

每个专项模型固定三个训练种子。下表记录各自依据验证目标选择的最佳轮次及验证指标；没有根据测试集重新选择种子。单种子指标有差异，说明单次训练偶然性不小。仅对问题二的 2026 种子做过一次去掉重建项的消融，其宏平均 F1 与带重建项近似，不能从这一个种子得出“重建显著有效”的结论。三个种子与单种子消融也不是等计算量比较。""")
    add(tab(["任务/消融", "种子", "最佳轮次", "Accuracy", "宏平均 F1", "MAE", "Pearson"],
            [[x["tag"], x["seed"], x["best_epoch"], f(x["acc"],4), f(x["f1_macro"],4),
              f(x["mae"],4), f(x["pearson"],4)] for x in runs]))
    add("""### 6.3 类别结构与混淆矩阵

整体准确率会受到正向类别较多的影响，宏平均 F1 则对三类等权。测试集的真实负向、中性、正向样本数分别为 207、158、362；中性类数量较少，而且靠近强度零边界，常与弱正向或弱负向相混。下面四张混淆矩阵按真实标签为行、预测标签为列展示，不应与回归强度预测直接等同。尤其问题三测试集中性类 F1 仅 0.3704，应以此约束解释结论：能解释模型为何预测，不表示这个预测一定正确。""")
    for model in ["p2", "p3"]:
        for split, zh in [("valid", "验证"), ("test", "测试")]:
            c = pd.read_csv(r / f"{model}_confusion_{split}.csv")
            label = c.columns[0]
            add(f"**{model.upper()} {zh}集混淆矩阵。**")
            add(tab(["真实/预测", "Negative", "Neutral", "Positive"],
                    [[row[label], row["Negative"], row["Neutral"], row["Positive"]]
                     for _, row in c.iterrows()]))

    add("""### 6.4 独立测试结果的抽样不确定性

对固定模型在 727 条测试片段的结果做普通样本 bootstrap，可估计“在同类独立样本假设下”指标的样本波动。题目片段可能来自相同原视频，而该重采样没有按视频分组，因此区间可能偏窄。区间用于说明本次测试数值的有限样本波动，不能用来宣称方法之间存在显著差异，也不能推导附件三、四的无标签正确率。""")
    intervals = pd.read_csv(r / "test_bootstrap_intervals.csv")
    add(tab(["模型", "指标", "2.5%", "97.5%"],
            [[x.model, x.metric, f(x.ci95_low,4), f(x.ci95_high,4)] for x in intervals.itertuples()]))
    add("""### 6.5 缺失率、模态组合与位置

鲁棒实验在验证集上运行，分类与回归指标仍使用固定的模型、训练统计量和验证集选定的分类偏置。人工缺失是对当前已观测位置再置零；率为 0 时对应原输入。图中各格是不同模态组合和缺失率下的宏平均 F1。即使目标率相同，不同样本的有效位置数和取整会让实际删除比例略有差别；对照 `robustness_mask_audit.csv` 可查看实际率和段数。连续段在“有效位置序列”上定义，遇到天然空洞时在存储轴上可能断成更多段。""")
    add("![问题二验证集人工缺失热力图](../results/figures/paper_robustness_matrix.png)")
    add("完整的 53 组验证情境如下。`rate` 行改变受损比例；`segments` 行固定比例改变连续段数；`position` 行固定比例改变遮蔽在前、中、后的位置。标准差仅对产生随机遮蔽的情境有意义；确定性位置重复种子不增加独立证据。")
    add(tab(["实验族", "模态", "目标率", "段数", "位置", "Accuracy", "宏平均 F1", "MAE"],
            [[x.family, x.modality, f(x.rate,1), x.segments, x.position, f(x.acc,4),
              f(x.f1_macro,4), f(x.mae,4)] for x in robust.itertuples()]))
    grouped_mask = mask_audit.groupby(["family", "modality", "target_rate", "segments", "position"], sort=False).agg(
        actual=("mean_actual_rate", "mean"), actual_segments=("mean_actual_segments", "mean"),
        max_rounding=("max_rounding_error", "max")).reset_index()
    add("下面按三个随机种子汇总实际遮蔽比例。最大取整误差按单样本记录，其量级取决于短序列长度；目标率与平均实际率接近并不说明每条样本都恰好达到目标率。")
    add(tab(["实验族", "模态", "目标率", "段数", "位置", "平均实际率", "平均实际段数", "最大取整误差"],
            [[x.family, x.modality, f(x.target_rate,1), x.segments, x.position,
              f(x.actual,3), f(x.actual_segments,2), f(x.max_rounding,2)] for x in grouped_mask.itertuples()]))

    add("""### 6.6 回归残差与逐例误差

仅看平均 MAE 会掩盖强情绪片段上的大错误。下图显示问题二验证集真实强度与预测强度以及不同真实极性下的绝对误差分布。部分片段的转写字面极性、语调和视频语境可能不一致；本研究不依据个别验证失败反向修改测试集结果，而是保留原始句子和误差以便复核。下表列出问题二和问题三各自验证集绝对误差最大的 8 条；完整转写保存在逐例 CSV 中，表内不截断或改写原句。""")
    add("![问题二验证集回归残差](../results/figures/paper_validation_errors.png)")
    for name, frame in [("问题二", p2), ("问题三", p3)]:
        add(f"**{name}验证集高误差样本。**")
        top = frame.sort_values("absolute_error", ascending=False).head(8)
        add(tab(["样本", "真类", "预测类", "真强度", "预测强度", "绝对误差"],
                [[x.sample_id, x.true_class, x.pred_class, f(x.true_intensity,3),
                  f(x.pred_intensity,3), f(x.absolute_error,3)] for x in top.itertuples()]))
    add("""从逐例错误可见，最大误差不全是“模型只输出接近零”造成：一些真值为强正向的片段被回归到负值，另一些真值负向片段被回归到正值。这类跨符号错误同时伤害回归 MAE 和三分类结果，单纯微调分类偏置无法修复回归头。验证集中的相关转写保留在 `p2_validation_details.csv` 与 `p3_validation_details.csv`；如果要判明是讽刺、语境、转写偏差还是特征噪声，必须逐条回看原视频，而不能只凭表中词句推断原因。正文因此把“可复核错误位置”与“已确认错误原因”明确区分。""")

    add(r"""## 7 解释值、词证据与复核条件

### 7.1 模态 Shapley 的定义与数值性质

Shapley 值以“完整输入预测的类别概率”为被解释量，在所有 8 个保留模态联盟上使用同一冻结模型。对三个模态，单个模态加入空集与加入两个模态联盟的权重都为 $1/3$，加入单模态联盟的两个边际作用权重各为 $1/6$。该加权平均会把交互作用分摊给相关模态，避免只看一次“单独删除”时把交互全部压到某一个通道。完整结果的有符号概率值相加等于全模态与全缺失预测之差；回归头也单独作相同分解。若某模态的 $\phi_m<0$，它在联盟平均意义下反而压低了最终预测类别的概率，绝对份额仍显示其敏感性大小，所以“份额大”并不总表示支持预测。

下图和表保留 20 条样本的有符号分类贡献。样本 02 的语音绝对作用略高于文本，因而主模态为语音；这不意味着其语音必然表达相同的情绪。样本 13 视觉输入全零，视觉 Shapley 接近零，不能解释成面部中性。""")
    add("![附件四有符号模态贡献](../results/figures/paper_shapley_signed.png)")
    add(tab(["编号", "预测极性", "文本 φ", "语音 φ", "视觉 φ", "分类效率差", "回归 φ 和"],
            [[x.sample_id, x.pred_polarity, f(x.phi_text_cls,4), f(x.phi_audio_cls,4),
              f(x.phi_vision_cls,4), f(x.efficiency_cls,4), f(x.sum_phi_reg,4)] for x in shap.itertuples()]))
    add("""### 7.2 局部删除与位置解释

对有效词元逐个把某一模态的该位置设为不可用，计算完整输入预测类别概率的变化。这与 Shapley 的“三个完整模态联盟”是不同粒度：前者回答哪些局部位置让模型当前预测更有把握，后者回答在各种模态组合中某个模态平均改变预测多少。局部作用取正值排序用于候选词展示；负值在机读结果中保留，不能把其绝对值直接称为支持证据。注意力权重只用于减少候选计算或提供辅助排序，不被当作解释本身。

附件四每条样本的文本、语音、视觉至多各保存三个局部候选。词时间来自 CTC 对齐；对没有可靠语言对应关系的标点、词元不一致或视觉全零位置，时间或实体证据可以为空。总计 177 条候选记录，其中 167 条有自动起止秒数；附录列出全部记录。以下三类信号需要人工复核：情感极性和强度方向冲突、文本词元与转写对应不足、自动对齐质量差。前两类并不自动否定预测，只说明系统无法给出足够一致的机器证据。""")
    audit = pd.read_csv(r / "evidence_mapping_audit.csv", dtype={"sample_id": str})
    add(tab(["编号", "词元编号一致", "内容位置", "文本可映射位", "路径置信指标", "转写相似度"],
            [[x.sample_id, "是" if x.token_ids_exact_match else "否", x.content_positions,
              x.text_mapped_positions, f(x.alignment_confidence,3), f(x.alignment_similarity,3)]
             for x in audit.itertuples()]))
    add("""### 7.3 不确定性陈述

分类最大概率是模型内部归一化输出，尚未做概率校准。比如 0.8 不能直接解读为“正确概率 80%”。Shapley 的效率恒等式仅检验数值计算符合定义；不同遮蔽基准、特征扰动或输入噪声可能改变归因排序。CTC 路径置信指标也不是时间戳正确率。因此本论文把三类量分开表述：预测值用于交付题目要求，贡献值用于描述模型响应，质量标志用于安排人工回看。""")
    return "\n".join(out)


def appendices(root):
    r = root / "results"
    align = pd.read_csv(r / "alignment_quality.csv", dtype={"sample_id": str})
    pad = pd.read_csv(r / "padding_and_observation_audit.csv", dtype={"id": str})
    ev = pd.read_csv(r / "evidence_time_mapping.csv", dtype={"sample_id": str})
    sections = []
    sections.append("""# 附录：逐例审计数据

以下表格属于论文正文结果的核对材料，按样本编号给出输入质量和解释定位。它们不是额外训练样本，也没有作为超参数选择依据。为阅读方便仅展示最关键列；原始 CSV 另保留全部数值。图表与逐例表的单位和掩码定义均与正文一致。

## 附录 A：120 条自动对齐质量

`a1` 为问题一 100 条视频，`a4` 为附件四 20 条视频。“优先回查”是自动审计标志，不是人工已判错。相似度低或路径指标低时应结合视频实际发音复核词时间。""")
    sections.append(tab(["部分", "样本编号", "CTC 路径指标", "转写相似度", "优先回查"],
                        [[x.dataset, x.sample_id, f(x.confidence,3), f(x.transcript_similarity,3),
                          "是" if x.needs_review else "否"] for x in align.itertuples()]))
    sections.append("""## 附录 B：50 条专项输入的三模态观测审计

每条专项样本分别列出文本、语音、视觉。内容位来自当前词元接口；观测位还要排除全零和异常数值。缺失率只相对当前内容位计算，尤其附件三不能据此推测原始句子被删除了多少词。""")
    sections.append(tab(["部分", "样本编号", "模态", "内容位", "观测位", "内容内全零位", "当前缺失率"],
                        [[x.dataset, x.id, x.modality, x.content_positions, x.observed_positions,
                          x.zero_within_content, f(x.missing_rate_within_available_token_sequence,3)]
                         for x in pad.itertuples()]))
    sections.append("""## 附录 C：177 条附件四局部证据与自动时间

`Δp` 是删除该位置后原预测类别概率的下降量；负值表示删除后该类概率反而上升。自动时间为空表示词元与语音时间不能可靠对应，不应把空值补成推测秒数。视觉全零的记录不应被描述为观测到的表情证据。""")
    def interval(x):
        return "—" if pd.isna(x.start_s) or pd.isna(x.end_s) else f"{f(x.start_s,2)}–{f(x.end_s,2)}"
    sections.append(tab(["编号", "模态", "排序", "位置", "候选词", "自动时间/s", "Δp", "对齐指标"],
                        [[x.sample_id, x.modality, x.rank, x.position, x.word, interval(x),
                          f(x.probability_drop,4), f(x.alignment_confidence,3)] for x in ev.itertuples()]))
    sections.append(r"""## 附录 D：复现步骤与结果校验规则

1. 核对官方四份附件目录，训练/验证/测试划分保持原样；专项附件的无标签内容只用于固定模型最终推理。
2. 问题一从原视频与音轨提取三模态物理时间窗特征，逐条保存输入路径、时长、有效掩码、词时间和异常状态；用原视频 ID 分组做 5 折交叉验证。
3. 问题二、三从附件二 aligned 版本读取连续特征，支持掩码先于观测掩码构造；只从训练观测估计标准化参数。训练三个固定种子，验证集选择轮次与类别偏置，保存全部权重。
4. 固定权重后在独立测试集计算三分类与回归指标；鲁棒性置零实验在验证集完成，记录目标率、实际率、段数与随机种子。
5. 附件三、四只执行推理。核对 30 与 20 个样本编号无重复无遗漏，极性和强度分别原样保存，方向冲突单独标注。
6. 附件四枚举 8 个模态联盟，逐样本验证 $\sum_m\phi_m=v(TAV)-v(\varnothing)$；局部证据先核对词元位置，再附上自动词时间。无可靠映射不补造时间。
7. 由同一结果 CSV 生成正文表格、图和 PDF，避免手工抄写产生数字差异；文件大小与压缩包完整性独立检查。

这些规则是本研究的复现路径。独立测试集、附件三与附件四的用途彼此不同；任何在专项无标签输入上新决定的阈值、缺失率或模型结构都应视为新的探索，不能追认成本论文的验证结论。""")
    return "\n\n".join(sections)
