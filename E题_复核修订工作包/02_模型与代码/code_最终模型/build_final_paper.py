"""Regenerate the submission paper from verified packaged results."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
P = ROOT / "paper"


def table(headers, rows):
    rows = [[str(v).replace("|", "\\|").replace("$", "\\$").replace("\n", " ") for v in row] for row in rows]
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(row) + " |" for row in rows])


def fmt(x, n=3):
    return "—" if pd.isna(x) else f"{float(x):.{n}f}"


def main():
    metrics = json.loads((R / "final_metrics.json").read_text(encoding="utf-8"))
    p1 = pd.read_csv(ROOT / "data" / "p1_summary_v2.csv")
    q2 = pd.read_csv(R / "附件3_最终预测.csv", dtype={"sample_id": str}, encoding="utf-8-sig")
    q3 = pd.read_csv(R / "附件4_最终预测与解释.csv", dtype={"sample_id": str}, encoding="utf-8-sig")
    shap = pd.read_csv(R / "附件4_shapley.csv", dtype={"sample_id": str}, encoding="utf-8-sig")
    cv = pd.read_csv(R / "p1_grouped_cv.csv")
    robust = pd.read_csv(R / "robustness_valid.csv")
    expl = pd.read_csv(R / "explanation_validation.csv")
    assert len(p1) == 100 and len(q2) == 30 and len(q3) == 20
    assert q3["sample_id"].equals(shap["sample_id"])

    parts = []
    def add(s):
        parts.append(s.strip() + "\n")

    add(r"""# 基于有效位置建模与精确模态归因的多模态情感分析

## 摘要

针对视频多模态特征构造、局部模态缺失下的情感预测以及可回查的预测解释，建立统一的输入审计和预测流程。问题一将冻结文本编码、声学描述符和视觉变化描述符映射到 50 个物理时间窗，使用 CTC 强制对齐记录词与视频秒数的对应关系；100 条视频均生成三模态特征及有效位置掩码。问题二在官方训练集学习模态专属时序表示和有效比例门控，通过局部遮蔽训练处理不完整输入；独立测试集的三分类准确率为 0.6754、宏平均 F1 为 0.6485，情感强度 MAE 为 0.6145。问题三使用同一数据协议训练可解释专项模型，对文本、语音、视觉三个模态的 8 个联盟进行精确 Shapley 分解，并用逐位置删除作用及自动词时间定位给出可回查证据；专项模型独立测试集准确率为 0.6327、宏平均 F1 为 0.5874。给出附件三 30 条预测和附件四 20 条预测及解释。模型选择仅使用验证集；无标签专项附件仅用于最终推理。解释反映已训练模型对输入扰动的响应，不等同于情绪形成的因果机制。

**关键词：** 多模态情感分析；局部缺失；有效位置掩码；时间对齐；Shapley 归因

## 1 问题与数据协议

附件一包含 100 个视频片段，源自 37 个原视频；附件二提供 aligned 特征，训练、验证、测试分别为 3395、728、727 条；附件三有 30 条无标签缺失样本，附件四有 20 条无标签解释样本。情感强度记为 $y\in[-3,3]$，极性按 $y<0$、$y=0$、$y>0$ 分为 Negative、Neutral、Positive。主评价指标为三分类准确率、宏平均 F1、强度 MAE 和 Pearson 相关系数。模型在训练集拟合，在验证集选择训练轮次和分类决策偏置；测试集只用于固定模型的最终评价。附件三、四不参与参数估计、缺失生成规则选择、阈值选择或模型筛选。

为避免错误合并不同特征接口，问题一的自建物理时间窗特征单独评估；问题二、三直接使用附件二及相同接口下的专项数据。附件二中的 `text` 为 768 维连续特征，`text_bert` 是词元索引和掩码接口，不是第四个情感模态。冻结的 BERT 与 wav2vec2 仅用于特征提取和时间定位，不在其他情感数据上训练。

基本假设是：aligned 版本的模态序列保持题目给定的位置关系；提供的转写大体对应音轨；局部置零可模拟一部分观测丢失。最后一项只覆盖“信息不可用”的情形，不代表噪声、遮挡、识别错误等所有真实损坏。填充位置和观测缺失分别记录，不把填充计入缺失率。

## 2 问题一：三模态特征与时间组织

### 2.1 特征构造

文本使用冻结 BERT 的 768 维最后层表示，以字符偏移将子词聚合到原转写词。音频以 16 kHz 单声道解码，25 ms 窗、10 ms 步提取 74 维描述符，含 MFCC 及差分、能量、过零率、基频、谱形、谱对比、色度与谱统计。视觉以 10 fps 抽帧并检测人脸，在样本内固定人脸裁剪区域上构造 7×5 网格的相对像素变化，共 35 维。该视觉量是外观变化描述符，可能同时受表情、头部运动和光照影响，不作为精确面部动作单元解释。

使用冻结 wav2vec2 的 CTC 输出对已知转写做单调强制对齐。令 $D(t,s)$ 为声学帧 $t$ 到字符状态 $s$ 的最优对数路径分数，递推为 $D(t,s)=\log p_t(c_s)+\max\{D(t-1,s),D(t-1,s-1),D(t-1,s-2)\}$，最后一项只在非空白且相邻状态不重复时允许。回溯得到词起止秒数。对时长 $T$ 建立 $I_j=[jT/50,(j+1)T/50)$，音视频特征在窗内对有效帧求均值，文本按词区间和时间窗的交叠长度加权。无有效观测的窗数值置零且掩码置零，未用复制值伪造观测。

每条样本保存形状为 `(50,768)`、`(50,74)`、`(50,35)` 的特征及三个掩码、51 个时间边界和元数据。100 条均完成，处理失败数为 0；实际解码音轨时长范围 2.228～29.110 秒。自动词时间未逐词人工校核，因此仅用于定位候选片段。典型样本的原素材、波形、词区间和特征窗如下。

![问题一时间对齐示例](../results/figures/p1_alignment_example.png)

### 2.2 特征有效性检验

按原视频 `video_id` 分组进行 5 折 StratifiedGroupKFold，使用种子 2026、2027、2028；同一原视频不会跨训练和验证折。每折仅从训练折估计标准化参数，再训练逻辑回归和 Ridge。下表是每个种子的折外预测指标再取平均，反映 100 条自建特征的内部预测信息，不代表附件二官方特征的性能。""")

    g = cv.groupby("feature", sort=False)[["acc", "f1_macro", "mae", "pearson"]].mean()
    add(table(["特征", "Accuracy", "宏平均 F1", "MAE", "Pearson"],
              [[k, *[fmt(v, 4) for v in row]] for k, row in g.iterrows()]))
    add("![按原视频分组的交叉验证](../results/figures/p1_grouped_cv.png)\n\n### 2.3 全量样本特征清单\n\n下表的有效窗数直接对应已提交的 100 个 NPZ。三模态固定形状如上；检测率为视频抽帧中的人脸检出比例，并非情感识别准确率。")
    add(table(["样本编号", "解码时长/s", "文本/语音/视觉有效窗", "人脸检测率"],
              [[r["sample_id"], fmt(r["duration_s"], 2),
                f"{int(r['text_valid'])}/{int(r['audio_valid'])}/{int(r['vision_valid'])}",
                fmt(r["face_detection_rate"], 2)] for _, r in p1.iterrows()]))

    add(r"""## 3 问题二：局部缺失下的极性与强度预测

### 3.1 缺失定义与模型

对模态 $m$、位置 $j$，用 $S_{mj}$ 表示内容支持、$O_{mj}$ 表示实际观测、$A_{mj}$ 表示训练时人工保留，输入掩码为 $M_{mj}=S_{mj}O_{mj}A_{mj}$。标准化均值和方差只从训练集已观测位置计算；缺失位置在标准化后仍为零。人工缺失率以遮蔽前可见位置数为分母，连续段在有序有效位置上抽取，实际删除数和存储轴段数另行审计。附件三已丢失内容的原始长度未知，故不能反推其真实原始缺失率。

三个模态分别投影到 96 维，经一层四头 Transformer 和注意力池化得到 $z_m$。有效观测比例 $a_m=\sum_j M_{mj}/\sum_j S_{mj}$ 参与门控：$\alpha_m=\operatorname{softmax}_m(g(z_m)+\log a_m)$，完全不可用模态在融合时屏蔽，融合向量 $z=\sum_m\alpha_mz_m$。分类头输出三类概率，回归头输出 $3\tanh(f(z)/3)$。训练目标为 $L=L_{CE}+0.6L_{Huber}+0.1L_{rec}$；重建损失仅作用于人工遮蔽而原本可见的位置，分别按位置与特征维数归一化。

优化器为 AdamW，学习率 $6\times10^{-4}$，权重衰减 $10^{-4}$，batch 64，最多 35 轮，验证分数为宏平均 F1 减去 MAE/6，7 轮无改善提前停止。问题二训练样本以 0.6 概率加 0.1～0.8 的局部缺失，随机选择模态组合及 1～3 个连续段。2026、2027、2028 三个种子等权集成分类概率与回归值；分类对数概率偏置由验证集选择，问题二为 `[-0.2,0,0]`。测试集不参与上述选择。

### 3.2 验证与独立测试

下表是冻结后的问题二模型结果。宏平均 F1 对每类等权，能揭示中性类识别不足；独立测试集中中性类 F1 为 0.4871，负向和正向分别为 0.7070、0.7514。分类概率尚未做可靠性校准，表中 Accuracy 不能直接当作逐例置信度。""")
    add(table(["数据划分", "样本数", "Accuracy", "宏平均 F1", "MAE", "Pearson"],
              [[name, n, *[fmt(v[k], 4) for k in ["acc", "f1_macro", "mae", "pearson"]]]
               for name, n, v in [("验证集", 728, metrics["p2"]["validation"]), ("测试集", 727, metrics["p2"]["test"])]]))
    add("![问题二验证集混淆矩阵](../results/figures/confusion_valid.png)\n\n在验证集上固定新增缺失率 0.4，以下为不同受损模态组合的宏平均 F1 和 MAE。此实验只验证预设的人工置零机制，不把附件三的无标签样本用于估计损坏分布。")
    rr = robust[(robust["family"] == "rate") & (robust["rate"] == 0.4)].copy()
    order = {"text": 0, "audio": 1, "vision": 2, "ta": 3, "tv": 4, "av": 5, "tav": 6}
    rr["ord"] = rr["modality"].map(order)
    rr = rr.sort_values("ord")
    add(table(["人工受损模态", "宏平均 F1", "MAE"],
              [[r["modality"], fmt(r["f1_macro"], 4), fmt(r["mae"], 4)] for _, r in rr.iterrows()]))
    add("图中另给出 0.2 和 0.8 缺失率；缺失位置、段数和实际掩码另见结果 CSV。\n\n![不同人工缺失率下的验证性能](../results/figures/robustness.png)\n\n### 3.3 附件三的 30 条最终预测\n\n极性由分类头给出，强度由独立回归头给出；中性类的回归强度可以非零。方向冲突标志只在正类却回归负值或负类却回归正值时为“是”，不擅自覆盖原预测。附件三无标签，不能计算准确率。")
    add(table(["编号", "极性", "强度", "最大分类概率", "方向冲突"],
              [[r["sample_id"], r["预测极性"], fmt(r["预测情感强度"], 4), fmt(r["分类置信度"], 3),
                "是" if r["极性强度方向冲突"] else "否"] for _, r in q2.iterrows()]))

    add(r"""## 4 问题三：预测依据的定量解释

### 4.1 预测模型与两级归因

问题三沿用三模态编码和训练集/验证集协议，局部缺失训练概率为 0.1，关闭重建项；使用三种子等权集成，验证集选出的分类偏置为 `[0.2,-0.2,0]`。其冻结模型在验证集 728 条上 Accuracy 0.6401、宏平均 F1 0.6242、MAE 0.5764、Pearson 0.6665；独立测试集 727 条相应为 0.6327、0.5874、0.6322、0.6803。中性类测试 F1 为 0.3704，是该解释模型的主要预测短板。

对每条附件四样本，先固定完整输入的预测类别 $c$，把保留模态集合 $S\subseteq\{T,A,V\}$ 的该类概率记为 $v(S)=p(c\mid x_S)$。对于模态 $m$，精确三参与者 Shapley 值为

$$\phi_m=\sum_{S\subseteq M\setminus\{m\}}\frac{|S|!(2-|S|)!}{3!}[v(S\cup\{m\})-v(S)].$$

枚举全部 8 个联盟并使用模型定义的缺失掩码，主参考模态取 $|\phi_m|$ 最大者，展示份额为 $|\phi_m|/\sum_k|\phi_k|$；有符号值保留在结果 CSV。每例满足 $\sum_m\phi_m=v(TAV)-v(\varnothing)$，浮点显示精度内残差为零。这个恒等式只是归因算法的内部核验，不证明真实因果忠实度。对每个有效词元另做单位置删除，记录原预测类概率变化 $\Delta_j=p(c\mid x)-p(c\mid x\setminus j)$；正值表示该位置支持当前预测。再把通过词元编号校验的位置映射到自动 CTC 词时间。没有可靠对应的位置留空，不虚构秒数。

### 4.2 解释有效性与质量控制

在固定的 128 条验证样本上，把注意力排序靠前的 10%、20%、40% 位置删除，并与 5 次等量随机删除配对。下表的概率下降差值及样本 bootstrap 区间只说明注意力可作为候选位置筛选；不等于对 Shapley 的独立因果验证。""")
    add(table(["删除比例", "样本数", "注意力候选下降", "随机下降", "配对差", "95%区间"],
              [[fmt(r["fraction"], 1), int(r["n"]), fmt(r["attention_drop"], 4),
                fmt(r["random_drop"], 4), fmt(r["paired_difference"], 4),
                f"[{fmt(r['ci95_low'],4)}, {fmt(r['ci95_high'],4)}]"] for _, r in expl.iterrows()]))
    main_counts = q3["主要参考模态"].value_counts().to_dict()
    shares = {m: q3[f"{m}贡献份额"].mean() for m in ["文本", "语音", "视觉"]}
    add(f"附件四 20 例中，Shapley 主参考模态为文本 {main_counts.get('text',0)} 例、语音 {main_counts.get('audio',0)} 例、视觉 {main_counts.get('vision',0)} 例；文本、语音、视觉的平均绝对贡献份额分别为 {shares['文本']:.3f}、{shares['语音']:.3f}、{shares['视觉']:.3f}。该比例仅描述这 20 条模型响应，不代表所有视频中的情感原因。样本 13 的对齐视觉特征全零，视觉证据不应作实体表情解释。自动语音识别与给定转写相似度不足的样本须优先人工回看。\n\n![典型样本的局部删除证据](../results/figures/explanation_card.png)")
    add("### 4.3 附件四的 20 条最终预测与解释\n\n下表展示全部样本的极性、强度、主参考模态和三模态贡献份额。份额是绝对 Shapley 值归一化结果，方向须查看完整 CSV 中的有符号 `phi_*_cls`。局部词证据为删除作用，可能与全模态归因的主模态不同。")
    def span(r):
        if pd.isna(r["估计起点秒"]) or pd.isna(r["估计终点秒"]):
            return "待核对"
        return f"{fmt(r['估计起点秒'],2)}–{fmt(r['估计终点秒'],2)}"
    add(table(["编号", "极性", "强度", "主模态", "文本份额", "语音份额", "视觉份额"],
              [[r["sample_id"], r["预测极性"], fmt(r["预测情感强度"], 3), r["主要参考模态"],
                fmt(r["文本贡献份额"], 3), fmt(r["语音贡献份额"], 3), fmt(r["视觉贡献份额"], 3)]
               for _, r in q3.iterrows()]))
    add("首位文本候选词与自动定位如下。完整的文本、语音、视觉候选位置、删除作用值和时间均见结果 CSV；自动时间仅供回看定位。")
    add(table(["编号", "首位文本候选词", "自动时间/s", "证据对齐置信度", "备注"],
              [[r["sample_id"], "—" if pd.isna(r["首位文本证据词"]) else r["首位文本证据词"],
                span(r), fmt(r["证据对齐置信度"], 3),
                "视觉全零" if r["sample_id"] == "13" else ("方向冲突" if r["极性强度方向冲突"] else "—")]
               for _, r in q3.iterrows()]))

    add("""## 5 结论与适用边界

三部分的最终输出分别为：100 条带物理时间窗与有效掩码的三模态特征；30 条缺失输入的极性、强度与冲突提示；20 条极性、强度、精确模态归因和可回查的局部时间证据。独立测试性能是固定模型在题目划分上的结果，不能外推为附件三或四的未知准确率。

目前主要限制有三点。第一，100 条自建样本量小，视觉像素变化易受头动与光照干扰；其分组交叉验证不能证明跨数据集泛化。第二，局部置零仅模拟观测丢失，附件三真实损坏机制未知；其无标签内容不应用于反推调参。第三，CTC 时间为自动估计，Shapley 与删除作用均依赖模型的遮蔽语义和预测函数，不能解释为真实情绪成因。对不一致类别/强度、低对齐相似度和视觉全零样本保留显式标志，供人工复查。

## 6 复现与提交文件

`data/p1_features` 保存 100 条 NPZ，`results/附件3_最终预测.csv` 和 `results/附件4_最终预测与解释.csv` 为两项专项结果。`results/附件4_shapley.csv` 保存 8 联盟推理得到的有符号归因及恒等式核验值。训练统计量、验证偏置及 6 份专项模型权重保存在项目中；完整运行命令与依赖见 README。预训练特征提取器体积较大，提供下载脚本而不纳入提交包。所有专题结果表均在正文展示，CSV 供机读和逐例审计。

## 参考文献

1. 赛题组，2026 年研究生数学建模 E 题题面与四份官方附件。
2. Zadeh A, et al. Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph. ACL, 2018. https://aclanthology.org/P18-1208/
3. Devlin J, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL, 2019. https://aclanthology.org/N19-1423/
4. Baevski A, et al. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. NeurIPS, 2020. https://arxiv.org/abs/2006.11477
5. Jain S, Wallace B C. Attention is not Explanation. NAACL, 2019. https://aclanthology.org/N19-1357/
6. Lundberg S M, Lee S-I. A Unified Approach to Interpreting Model Predictions. NeurIPS, 2017. https://arxiv.org/abs/1705.07874
7. Guo Z, et al. MPLMM: Multi-Modal Prompt Learning for Missing Modality. ACL, 2024. https://github.com/zrguo/MPLMM （方法背景；本文未使用其权重、训练数据或实验数字。）
""")
    from paper_extra import extended_main, appendices
    body = "\n".join(parts)
    body = body.replace("## 5 结论与适用边界", extended_main(ROOT) + "\n\n## 8 结论与适用边界", 1)
    body = body.replace("## 6 复现与提交文件", "## 9 复现与提交文件", 1)
    body += "\n\n" + appendices(ROOT) + "\n"
    out = P / "论文终稿.md"
    out.write_text(body, encoding="utf-8")
    pandoc = shutil.which("pandoc") or str(Path(sys.prefix) / "Library" / "bin" / "pandoc.exe")
    html = P / "论文终稿.html"
    if Path(pandoc).exists():
        subprocess.run([pandoc, str(out), "-o", str(html), "--mathml", "-s", "--toc",
                        "-V", "lang=zh-CN"], check=True)
    else:
        try:
            import markdown
            body = markdown.markdown(out.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "sane_lists"])
            css = "body{font-family:'Microsoft YaHei',sans-serif;max-width:980px;margin:40px auto;line-height:1.75;color:#222}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #bbb;padding:5px}th{background:#eef2f5}img{max-width:100%}"
            html.write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><style>'+css+'</style><body>'+body+'</body></html>', encoding="utf-8")
        except ImportError:
            pass
    print(out, "chars", len(out.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
