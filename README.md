# 2026 华为杯 E 题 · 复杂场景下多模态情感预测

CMU-MOSEI 多模态情感（附件 1–4）。本仓库为**私有**队内工作仓库：代码、结果、论文、提交包。
竞赛期间严禁公开（评审后按组委会规定处理）。

## 仓库内容（clone 后即得）

| 目录 | 内容 |
|---|---|
| `solution/` | 主求解仓库：Q1 YuNet 加固、Q2 冠军模型（机制匹配增强+门控融合）、Q3 精确 Shapley、`paper/论文.md` + 三图、全部 `logs/*.json` 结果 |
| `math_model_e_optimized/` | 基础流水线仓库：CTC 对齐、特征提取、p2/p3 基线（含小模型头权重 ~1.9MB×9） |
| `E题_submission/` | 最终提交包（16.1MB，`solution/src/build_submission.py` 一键再生） |
| `E题_队友评审包/` | 给队友的轻量评审包（论文+数字+导读） |
| `models/` | YuNet onnx / blaze_face tflite（bert 等大权重不入库，见下） |

## clone 后还需要自行准备的（`.gitignore` 已排除）

1. **比赛原始数据**（版权原因不入库）：`E题/E题数据/` —— 从官方渠道解压到本路径；
2. **预训练权重** → 放 `models/`：
   - `bert-base-uncased/`（422MB）：ModelScope `AI-ModelScope/bert-base-uncased` 或 HF `google-bert/bert-base-uncased`
   - `bert-tiny/`：HF `google/bert_uncased_L-2_H-128_A-2`
   - wav2vec2-base-960h（CTC 对齐用）：torchaudio bundle 或 HF
   - YuNet onnx 已在库内（sha256 见 `solution/提交说明.md`）
3. **冠军权重**（3×420MB）：本仓库 GitHub **Release `weights-v1`** 下载，
   放回 `solution/weights/q2_base_cw/s{2026,7,42}/best.pt`，即可直接推理，无需重训；
   其余消融配置权重可按 `solution/提交说明.md` 命令重训。

## 快速上手（新机器）

```bash
git clone <本仓库> && cd math_competition
# 数据与预训练权重按上节放好
cd solution && conda activate mosei
python -m src.final_test_eval        # 复现测试集一次评估（需冠军权重）
python -m src.q3_shapley            # 复现附件4 Shapley 归因
```

环境、固定版本下载表、完整复现顺序见 `solution/提交说明.md`。
