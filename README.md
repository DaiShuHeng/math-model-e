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

## clone 后还需要自行准备的（`.gitignore` 已排除，一键下载）

```bash
TOKEN=<你的GitHub PAT> bash download_all.sh
```

脚本从本仓库两个 Release（私有资产，必须走 API 端点）下载并自动放置：

| Release | 资产 | 大小 | 放置位置 |
|---|---|---|---|
| data-v1 | E题数据.zip | 1.8G | 解压到 `E题/E题数据/` |
| data-v1 | bert-base-uncased.zip | 408M | 解压到 `models/bert-base-uncased/` |
| data-v1 | bert-tiny.zip | 17M | 解压到 `models/bert-tiny/` |
| data-v1 | cache_misc.zip | 39M | 解压回 `solution/cache/` 等原路径 |
| weights-v1 | q2_base_cw 3 种子 best.pt | 3×420M | `solution/weights/q2_base_cw/<seed>/` |
| weights-v1 | q2_ablation_weights.zip | 1.6G | 解压到 `solution/weights/`（9 组消融配置） |

全部带 SHA256 校验。仅冠军推理可只跑 `download_weights.sh`（体积小得多）。
wav2vec2-base-960h（CTC 对齐用）不入 Release：torchaudio bundle 会自动下载。
YuNet onnx 已在 git 库内（sha256 见 `solution/提交说明.md`）。

## 快速上手（新机器）

```bash
git clone <本仓库> && cd math_competition
# 数据与预训练权重按上节放好
cd solution && conda activate mosei
python -m src.final_test_eval        # 复现测试集一次评估（需冠军权重）
python -m src.q3_shapley            # 复现附件4 Shapley 归因
```

环境、固定版本下载表、完整复现顺序见 `solution/提交说明.md`。
