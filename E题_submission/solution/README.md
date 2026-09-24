# E题求解代码仓库

多模态情感预测（CMU-MOSEI 派生数据）：Q1 特征提取与对齐 / Q2 局部缺失鲁棒预测 / Q3 可解释预测。

## 目录结构

```
solution/
├── src/
│   ├── config.py          # 路径、常量（UNK_ID=100、词表 30522）、30 个经验缺失率池、P_FILE=0.9
│   ├── data_adapter.py    # 安全反序列化(numpy白名单) → 数组 + 可见性掩码；训练集拟合的标准化器(保零)
│   ├── augment.py         # 词级 T+A+V 同步破坏增强（逆向工程附件3损伤机制），含类型/位置/时长变体
│   ├── datasets.py        # torch Dataset：在线增强 → 重算可见掩码 → 标准化；固定种子评测协议
│   ├── metrics.py         # Acc/MacroF1/WeightedF1(labels=[0,1,2]) + MAE/Pearson + 双头一致性
│   ├── models.py          # 冻结/微调 BERT-Tiny + 各模态 BiGRU 分支 + 掩码注意力池化 + 可用性门控融合 + 双头
│   ├── train_q2.py        # 训练：train 修参数、valid 选模型/早停、test 从不触碰；clean+corrupt20 双评测
│   ├── eval_q2.py         # 缺失率扫描 (0~0.7, 固定种子777) —— 鲁棒性曲线图数据
│   ├── eval_variants.py   # 缺失类型/位置/时长影响规律网格 —— 论文 Q2 消融数据
│   └── predict_att3.py    # 附件3 (30文件) 最终推理：多种子集成 → 预测CSV(极性+强度+门控)
├── tests/run_tests.py     # 7 项验收测试（附件3签名30/30、损伤集重放、不变量、统计、确定性…）
├── weights/               # checkpoints (best.pt + metrics.json + rate_curve.json)
├── results/               # 附件3/4 预测 CSV
├── cache/                 # aligned_arrays.npz 数组缓存
└── logs/                  # 训练/评测日志与 JSON 结果
```

## 关键协议（竞赛纪律）

1. **只用 train 拟合任何统计量**（标准化器）；valid 仅用于模型选择与超参；test 不参与任何决策。
2. **鲁棒性评测固定种子** (base 777)：所有配置/缺失率看到相同的损坏副本，性能差可比较。
3. **附件3/4 无标签、仅最终推理**：预测时特征原样使用（真实损伤已在数据内），绝不回查干净输入。
4. 标准化保零：缺失行变换后仍为精确 0，模型掩码语义不被破坏。
5. pkl 经 numpy 白名单反序列化器读取（安全）。

## 复现命令

```bash
cd solution
conda activate mosei && export CUDA_VISIBLE_DEVICES=0

python -m tests.run_tests                    # 验收测试（全部通过）
# 基线（冻结 BERT-Tiny）
python -m src.train_q2 --seed 2026 --out weights/q2/s2026
# 微调 + 类别加权 CE（当前最优配置）
python -m src.train_q2 --seed 2026 --freeze-text 0 --cls-weight 1 --out weights/q2_ft_cw/s2026
# 消融：无增强
python -m src.train_q2 --seed 2026 --freeze-text 0 --no-aug --out weights/q2_ft_noaug/s2026
# 缺失率曲线 / 影响规律网格 / 附件3 推理
python -m src.eval_q2 --ckpt weights/q2_ft_cw/s*/best.pt --out logs/rate_curve_cw.json
python -m src.eval_variants --ckpt weights/q2_ft_cw/s*/best.pt
python -m src.predict_att3 --ckpt weights/q2_ft_cw/s*/best.pt --out results/att3_predictions.csv
```

## 当前结果（valid, 3 种子均值±std）

| 配置 | clean Acc/mF1/r | corrupt20 Acc/mF1 | 说明 |
|---|---|---|---|
| 冻结 BERT-Tiny + 增强 | 0.575 / 0.524 / 0.501 | 0.560 / 0.500 | 线性探针≈完整模型 → 冻结是瓶颈 |
| 微调 + 增强 | 0.601 / 0.559 / 0.566 | 0.581 / 0.532 | 微调 +2.6 Acc |
| 微调 + 无增强（消融） | 0.595 / 0.563 / 0.562 | **0.533** / 0.530 | 证明增强专门买到鲁棒性 |
| **微调 + 增强 + 类别加权** | **0.594 / 0.569 / 0.569** | **0.573 / 0.545** | 当前冠军 |

- 多数类基线 valid Acc=0.464 / mF1=0.211。
- 缺失率 0→0.7：冠军模型 Acc 仅 0.594→~0.53，门控自动从文本(0.70→0.61)向 A/V 转移。
- 影响规律：文本缺失主导；尾部缺失对 Pearson 伤害最大；连续块比散点更伤连续性指标。

## 环境依赖

conda env `mosei`（Python 3.11, torch 2.14+cu130, transformers 5.17, sklearn）；
BERT-Tiny 权重在 `models/bert-tiny/`（fp16 8.8MB，满足 50MB 附件限制）；
GPU 0（A100 80G，GPU 1 被他人占用）。中文网络：HF 用 hf-mirror.com，pip 用清华源。
